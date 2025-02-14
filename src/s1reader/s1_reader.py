from __future__ import annotations

import datetime
import glob
import os
import warnings
import lxml.etree as ET
import zipfile
from typing import Union

from types import SimpleNamespace
from packaging import version

import isce3
import numpy as np
import shapely

from scipy.interpolate import InterpolatedUnivariateSpline
from nisar.workflows.stage_dem import check_dateline

from s1reader import s1_annotation  # to access __file__
from s1reader.s1_annotation import (
    RFI_INFO_AVAILABLE_FROM,
    CalibrationAnnotation,
    AuxCal,
    BurstCalibration,
    BurstEAP,
    BurstNoise,
    BurstExtendedCoeffs,
    NoiseAnnotation,
    ProductAnnotation,
    SwathRfiInfo,
    SwathMiscMetadata,
)

from s1reader.s1_burst_slc import Doppler, Sentinel1BurstSlc
from s1reader.s1_burst_id import S1BurstId
from s1reader.s1_orbit import T_ORBIT, PADDING_SHORT, merge_osv_list

# Tolerance of the ascending node crossing time in seconds
ASCENDING_NODE_TIME_TOLERANCE_IN_SEC = 1.0
esa_track_burst_id_file = (
    f"{os.path.dirname(os.path.realpath(__file__))}/data/sentinel1_track_burst_id.txt"
)


# TODO evaluate if it make sense to combine below into a class
def as_datetime(t_str):
    """Parse given time string to datetime.datetime object.

    Parameters:
    ----------
    t_str : string
        Time string to be parsed. (e.g., "2021-12-10T12:00:0.0")

    Returns:
    ------
    _ : datetime.datetime
        datetime.datetime object parsed from given time string.
    """
    return datetime.datetime.fromisoformat(t_str)


def parse_polynomial_element(elem, poly_name):
    """Parse azimuth FM (Frequency Modulation) rate element to reference time and poly1d tuples.

    Parameters
    ----------
    elem : Element
        Element containing coefficients.
    poly_name : string
        Name of element containing azimuth time and polynomial coefficients.

    Returns:
    ------
    _ : tuple
        Tuple of time and Poly1d constructed from azimuth time and coefficients.
    """
    ref_time = as_datetime(elem.find("azimuthTime").text)

    half_c = 0.5 * isce3.core.speed_of_light
    r0 = half_c * float(elem.find("t0").text)

    # NOTE Format of the azimuth FM rate polynomials has changed
    # when IPF version was somewhere between 2.36 and 2.82
    if elem.find(poly_name) is None:  # before the format change i.e. older IPF
        coeffs = [float(x.text) for x in elem[2:]]
    else:  # after the format change i.e. newer IPF
        coeffs = [float(x) for x in elem.find(poly_name).text.split()]

    poly1d = isce3.core.Poly1d(coeffs, r0, half_c)
    return (ref_time, poly1d)


def get_nearest_polynomial(t_mid, time_poly_pair):
    """Find polynomial closest to given sensing mid and return associated poly1d.

    Parameters
    ----------
    t_mid : datetime.datetime
        Middle of the burst
    time_poly_pair: list(tuple)
        List of tuples of time and associated Poly1d

    Returns:
    ------
    nearest_poly: list
        Polynomial coefficients associated with nearest time.
    """
    # lambda calculating absolute time difference
    get_abs_dt = lambda t_mid, t_new: np.abs((t_mid - t_new).total_seconds())

    # calculate 1st dt and polynomial
    dt = get_abs_dt(t_mid, time_poly_pair[0][0])
    nearest_poly = time_poly_pair[0][1]

    # loop thru remaining time, polynomial pairs
    for x in time_poly_pair[1:]:
        temp_dt = get_abs_dt(t_mid, x[0])

        # stop looping if dt starts growing
        if temp_dt > dt:
            break

        # set dt and polynomial for next iteration
        dt, nearest_poly = temp_dt, x[1]

    return nearest_poly


def doppler_poly1d_to_lut2d(
    doppler_poly1d, starting_slant_range, slant_range_res, shape, az_time_interval
):
    """Convert doppler poly1d to LUT2d.

    Parameters
    ----------
    doppler_poly1d : poly1d
        Poly1d object to be converted.
    starting_slant_range : float
        Starting slant range of the burst.
    slant_range_res : float
        Slant-range pixel spacing of the burst.
    shape : tuple
        Tuples holding number of lines and samples of the burst.
    az_time_interval : float
        Azimth time interval of the burst.

    Returns:
    ------
    _ : LUT2d
        LUT2d calculated from poly1d.
    """
    (n_lines, n_samples) = shape
    # calculate all slant ranges in grid
    slant_ranges = starting_slant_range + np.arange(n_samples) * slant_range_res

    # no az dependency, but LUT2d required, so ensure all az coords covered
    # offset by -2 days in seconds (reference epoch)
    offset_ref_epoch = 2 * 24 * 3600
    az_times = offset_ref_epoch + np.array([0, n_lines * az_time_interval])

    # calculate frequency for all slant range
    freq_1d = doppler_poly1d.eval(slant_ranges)
    # freq_1d = np.array([doppler_poly1d.eval(t) for t in slant_ranges])

    # init LUT2d (vstack freq since no az dependency) and return
    return isce3.core.LUT2d(slant_ranges, az_times, np.vstack((freq_1d, freq_1d)))


def get_burst_orbit(sensing_start, sensing_stop, osv_list: ET.Element):
    """Init and return ISCE3 orbit.

    Parameters:
    -----------
    sensing_start : datetime.datetime
        Sensing start of burst; taken from azimuth time
    sensing_stop : datetime.datetime
        Sensing stop of burst
    osv_list : xml.etree.ElementTree.Element
        ElementTree containing orbit state vectors

    Returns:
    --------
    _ : datetime
        Sensing mid as datetime object.
    """
    orbit_sv = []
    # add start & end padding to ensure sufficient number of orbit points
    pad = datetime.timedelta(seconds=PADDING_SHORT)
    for osv in osv_list:
        t_orbit = as_datetime(osv[1].text[4:])

        if t_orbit > sensing_stop + pad:
            break

        if t_orbit > sensing_start - pad:
            pos = [float(osv[i].text) for i in range(4, 7)]
            vel = [float(osv[i].text) for i in range(7, 10)]
            orbit_sv.append(
                isce3.core.StateVector(isce3.core.DateTime(t_orbit), pos, vel)
            )

    # use list of stateVectors to init and return isce3.core.Orbit
    time_delta = datetime.timedelta(days=2)
    ref_epoch = isce3.core.DateTime(sensing_start - time_delta)

    return isce3.core.Orbit(orbit_sv, ref_epoch)


def calculate_centroid(lons, lats):
    """Calculate burst centroid from boundary longitude/latitude points.

    Parameters:
    -----------
    lons : list
        Burst longitudes (degrees)
    lats : list
        Burst latitudes (degrees)

    Returns:
    --------
    _ : shapely.geometry.Point
        Burst center in degrees longitude and latitude
    """
    proj = isce3.core.Geocent()

    # convert boundary points to geocentric
    xyz = [
        proj.forward([np.deg2rad(lon), np.deg2rad(lat), 0])
        for lon, lat in zip(lons, lats)
    ]

    # get mean of corners as centroid
    xyz_centroid = np.mean(np.array(xyz), axis=0)

    # convert back to LLH
    llh_centroid = [np.rad2deg(x) for x in proj.inverse(xyz_centroid)]

    return shapely.geometry.Point(llh_centroid[:2])


def get_burst_centers_and_boundaries(tree, num_bursts=None):
    """Parse grid points list and calculate burst center lat and lon

    Parameters
    ----------
    tree : Element
        Element containing geolocation grid points.
    num_bursts: int or None
        Expected number of bursts in the subswath.
        To check if the # of polygon is the same as the parsed polygons.
        When it's None, the number of burst is the same as # polygons found in this function.

    Returns
    -------
    center_pts : list
        List of burst centroids ass shapely Points
    boundary_pts : list
        List of burst boundaries as shapely Polygons
    """
    # find element tree
    grid_pt_list = tree.find("geolocationGrid/geolocationGridPointList")

    # read in all points
    n_grid_pts = int(grid_pt_list.attrib["count"])
    lines = np.empty(n_grid_pts)
    pixels = np.empty(n_grid_pts)
    lats = np.empty(n_grid_pts)
    lons = np.empty(n_grid_pts)
    for i, grid_pt in enumerate(grid_pt_list):
        lines[i] = int(grid_pt[2].text)
        pixels[i] = int(grid_pt[3].text)
        lats[i] = float(grid_pt[4].text)
        lons[i] = float(grid_pt[5].text)

    unique_line_indices = np.unique(lines)

    if num_bursts is None:
        num_bursts = len(unique_line_indices) - 1
    n_bursts = len(unique_line_indices) - 1
    center_pts = [[]] * n_bursts
    boundary_pts = [[]] * n_bursts

    # zip lines numbers of bursts together and iterate
    for i, (ln0, ln1) in enumerate(
        zip(unique_line_indices[:-1], unique_line_indices[1:])
    ):
        # create masks for lines in current burst
        mask0 = lines == ln0
        mask1 = lines == ln1

        # reverse order of 2nd set of points so plots of boundaries
        # are not connected by a diagonal line
        burst_lons = np.concatenate((lons[mask0], lons[mask1][::-1]))
        burst_lats = np.concatenate((lats[mask0], lats[mask1][::-1]))

        center_pts[i] = calculate_centroid(burst_lons, burst_lats)

        poly = shapely.geometry.Polygon(zip(burst_lons, burst_lats))
        boundary_pts[i] = check_dateline(poly)

    num_border_polygon = len(unique_line_indices) - 1
    if num_bursts > num_border_polygon:
        warnings.warn(
            "Inconsistency between # bursts in subswath, and the # polygons. "
        )
        num_missing_polygons = num_bursts - num_border_polygon

        center_pts += [shapely.Point()] * num_missing_polygons
        boundary_pts += [[shapely.Polygon()]] * num_missing_polygons

    return center_pts, boundary_pts


def get_ipf_version(tree: ET):
    """Extract the IPF version from the ET of manifest.safe.

    Parameters
    ----------
    tree : xml.etree.ElementTree
        ElementTree containing the parsed 'manifest.safe' file.

    Returns
    -------
    ipf_version : version.Version
        IPF version of the burst.
    """
    # get version from software element
    search_term, nsmap = _get_manifest_pattern(
        tree, ["processing", "facility", "software"]
    )
    software_elem = tree.find(search_term, nsmap)
    ipf_version = version.parse(software_elem.attrib["version"])

    return ipf_version


def get_swath_misc_metadata(
    et_manifest: ET, et_product: ET, product_annotation: ProductAnnotation
):
    """
    Extract miscellaneous metadata

    Parameters
    ----------
    et_manifest: ET
        Element Tree of manifest.safe
    et_product:ET
        Element Tree of product annotation file
    product_annotation: ProductAnnotation
        Swath product annotation class

    Returns
    -------
        misc_metadata: SwathMiscMetadata
            swath-wide miscellaneous metadata

    """
    # Load the processing data from the manifest

    search_term, nsmap = _get_manifest_pattern(et_manifest, ["processing"])
    processing_elem = et_manifest.find(search_term, nsmap)
    post_processing_dict = dict(processing_elem.attrib)

    # Extract the facility information from `processing_elem`, and
    # add them to `post_processing_dict`
    post_processing_facility_elem = processing_elem.find("safe:facility", nsmap)
    post_processing_dict.update(dict(post_processing_facility_elem.attrib))

    slant_range_looks = int(
        et_product.find(
            "imageAnnotation/processingInformation/"
            "swathProcParamsList/swathProcParams"
            "/rangeProcessing/numberOfLooks"
        ).text
    )
    azimuth_looks = int(
        et_product.find(
            "imageAnnotation/processingInformation/"
            "swathProcParamsList/swathProcParams/"
            "azimuthProcessing/numberOfLooks"
        ).text
    )

    inc_angle_arr = product_annotation.antenna_pattern_incidence_angle
    inc_angle_aztime_arr = product_annotation.antenna_pattern_azimuth_time

    return SwathMiscMetadata(
        azimuth_looks,
        slant_range_looks,
        inc_angle_aztime_arr,
        inc_angle_arr,
        post_processing_dict,
    )


def get_start_end_track(tree: ET):
    """Extract the start/end relative orbits from manifest.safe file"""
    search_term, nsmap = _get_manifest_pattern(
        tree, ["orbitReference", "relativeOrbitNumber"]
    )
    elem_start, elem_end = tree.findall(search_term, nsmap)
    return int(elem_start.text), int(elem_end.text)


def _get_manifest_pattern(tree: ET, keys: list):
    """Get the search path to extract data from the ET of manifest.safe.

    Parameters
    ----------
    tree : xml.etree.ElementTree
        ElementTree containing the parsed 'manifest.safe' file.
    keys : list
        List of keys to search for in the manifest file.

    Returns
    -------
    str
        Search path to extract from the ET of the manifest.safe XML.

    """
    # https://lxml.de/tutorial.html#namespaces
    # Get the namespace from the root element to avoid full urls
    # This is a dictionary with a short name containing the labels in the
    # XML tree, and the fully qualified URL as the value.
    try:
        nsmap = tree.nsmap
    except AttributeError:
        nsmap = tree.getroot().nsmap
    # path to xmlData in manifest
    xml_meta_path = "metadataSection/metadataObject/metadataWrap/xmlData"
    safe_terms = "/".join([f"safe:{key}" for key in keys])
    return f"{xml_meta_path}/{safe_terms}", nsmap


def get_path_aux_cal(directory_aux_cal: str, str_annotation: str):
    """
    Decide which aux_cal to load
    Criteria to select an AUX_CAL:
    1. Select the auxiliary product(s) with a validity start date/time
       closest to, but not later than, the start of the job order;
    2. If there is more than one product which meets the first criteria
       (e.g. two auxiliary files have the same validity date/time),
       then use the auxiliary product with the latest generation time.

    The criteria above is based on ESA document in the link below:
    https://sentinel.esa.int/documents/247904/1877131/Sentinel-1_IPF_Auxiliary_Product_Specification

    Parameters:
    -----------
    diectory_aux_cal: str
        Directory for the AUX_CAL .zip files
    str_annotation: str
        annotation_path that is used in `burst_from_xml()`

    Return:
    --------
    path_aux_cal: str
        Path to the AUX_CAL file that corresponds to the criteria provided
        None if the matching AUX_CAL is not found in `directory_aux_cal`

    """

    # extract the date string and platform info from str_annotation
    str_safe = os.path.basename(str_annotation.split(".SAFE")[0])

    token_safe = str_safe.split("_")
    str_platform = token_safe[0]
    str_sensing_start = token_safe[5]

    list_aux_cal = glob.glob(f"{directory_aux_cal}/{str_platform}_AUX_CAL_V*.SAFE.zip")

    if len(list_aux_cal) == 0:
        raise ValueError(f"Cannot find AUX_CAL files from {directory_aux_cal} .")

    format_datetime = "%Y%m%dT%H%M%S"

    datetime_sensing_start = datetime.datetime.strptime(
        str_sensing_start, format_datetime
    )

    # sequentially parse the time info of AUX_CAL and search for the matching file
    id_match = None
    dt_validation_prev = None
    dt_generation_prev = 1  # dummy value
    for i, path_aux_cal in enumerate(list_aux_cal):
        token_aux_cal = os.path.basename(path_aux_cal).split("_")
        datetime_validation = datetime.datetime.strptime(
            token_aux_cal[3][1:], format_datetime
        )
        datetime_generation = datetime.datetime.strptime(
            token_aux_cal[4][1:].split(".")[0], format_datetime
        )

        dt_validation = int(
            (datetime_sensing_start - datetime_validation).total_seconds()
        )
        dt_generation = int(
            (datetime_sensing_start - datetime_generation).total_seconds()
        )

        if dt_validation < 0:
            # Validation date is later than the sensing time;
            # Move to the next iteration
            continue

        if dt_validation_prev is None:
            # Initial allocation
            id_match = i
            dt_validation_prev = dt_validation
            dt_generation_prev = dt_generation
            continue

        if dt_validation_prev > dt_validation:
            # Better AUX_CAL found;
            # Replace the candidate to the one in this iteration
            id_match = i
            dt_validation_prev = dt_validation
            dt_generation_prev = dt_generation
            continue

        # Same validity time; Choose the one with latest generation time
        if dt_validation_prev == dt_validation and dt_generation_prev > dt_generation:
            id_match = i
            dt_generation_prev = dt_generation

    if id_match is None:
        print("ERROR finding AUX_CAL to use.")
        return None

    return list_aux_cal[id_match]


def is_eap_correction_necessary(ipf_version: version.Version) -> SimpleNamespace:
    """
    Examines if what level of elevation antenna pattern (EAP) correction is necessary.
    based on the IPF version.
    Based on the comment on PR: https://github.com/opera-adt/s1-reader/pull/48#discussion_r926138372

    Parameter
    ---------
    ipf_version: version.Version
        IPF version of the burst

    Return
    ------
    eap: SimpleNamespace
        eap.magnitude_correction == True if both magnitude and phase need to be corrected
        eap.phase_correction == True if only phase correction is necessary

    """

    # Based on ESA technical document
    eap = SimpleNamespace()

    ipf_243 = version.parse("2.43")
    eap.phase_correction = True if ipf_version < ipf_243 else False

    ipf_236 = version.parse("2.36")
    eap.magnitude_correction = True if ipf_version < ipf_236 else False

    return eap


def get_track_burst_num(track_burst_num_file: str = esa_track_burst_id_file):
    """Read the start / stop burst number info of each track from ESA.

    Parameters:
    -----------
    track_burst_num_file : str
        Path to the track burst number files.

    Returns:
    --------
    track_burst_num : dict
        Dictionary where each key is the track number, and each value is a list
        of two integers for the start and stop burst number
    """

    # read the text file to list
    track_burst_info = np.loadtxt(track_burst_num_file, dtype=int)

    # convert lists into dict
    track_burst_num = dict()
    for track_num, burst_num0, burst_num1 in track_burst_info:
        track_burst_num[track_num] = [burst_num0, burst_num1]

    return track_burst_num


def get_ascending_node_time_orbit(
    orbit_state_vector_list: ET,
    sensing_time: datetime.datetime,
    anx_time_annotation: datetime.datetime = None,
    search_length=None,
):
    """
    Estimate the time of ascending node crossing from orbit

    Parameters
    ----------
    orbit_state_vector_list: ET
        XML elements that points to the list of orbit information.
        Each element should contain the information below:
        TAI, UTC, UT1, Absolute_Orbit, X, Y, Z, VX, VY, VZ, and Quality

    sensing_time: datetime.datetime
        Sensing time of the data

    anx_time_annotation: datetime.datetime
        ascending node crossing (ANX) time retrieved from annotation data.
        - If it is provided, then the orbit ANX time close to this will be returned.
        - If it is `None`, then the orbit ANX time closest to (but no later than)
          the sensing time will be returned.

    search_length: datetime.timedelta
        Search length in time to crop the data in `orbit_state_vector_list`

    Returns
    -------
    _ : datetime.datetime
        Ascending node crossing time calculated from orbit
    """

    # Crop the orbit information before 2 * (orbit period) of sensing time
    if search_length is None:
        search_length = datetime.timedelta(seconds=2 * T_ORBIT)

    # Load the OSVs
    utc_vec_all = [
        datetime.datetime.fromisoformat(osv.find("UTC").text.replace("UTC=", ""))
        for osv in orbit_state_vector_list
    ]
    utc_vec_all = np.array(utc_vec_all)
    pos_z_vec_all = [float(osv.find("Z").text) for osv in orbit_state_vector_list]
    pos_z_vec_all = np.array(pos_z_vec_all)

    # NOTE: tried to apply the same amount of pading in `get_burst_orbit` to
    # get as similar results as possible.
    pad = datetime.timedelta(seconds=PADDING_SHORT)

    # Constrain the search area
    flag_search_area = (utc_vec_all > sensing_time - search_length - pad) & (
        utc_vec_all < sensing_time + pad
    )
    orbit_time_vec = utc_vec_all[flag_search_area]
    orbit_z_vec = pos_z_vec_all[flag_search_area]

    # Detect the event of ascending node crossing from the cropped orbit info.
    # The algorithm was inspired by Scott Staniewicz's PR in the link below:
    # https://github.com/opera-adt/s1-reader/pull/120/
    datetime_ascending_node_crossing_list = []

    # Iterate through the z coordinate in orbit object to
    # detect the ascending node crossing
    iterator_z_time = zip(orbit_z_vec, orbit_z_vec[1:], orbit_time_vec)
    for index_z, (z_prev, z, _) in enumerate(iterator_z_time):
        # Check if ascending node crossing has taken place;
        if not z_prev < 0 <= z:
            continue

        # Extract z coords. and time for interpolation
        index_from = max(index_z - 3, 0)
        index_to = min(index_z + 3, len(orbit_z_vec))
        z_around_crossing = orbit_z_vec[index_from:index_to]
        time_around_crossing = orbit_time_vec[index_from:index_to]

        # Set up spline interpolator and interpolate the time when z is equal to 0.0
        datetime_orbit_ref = time_around_crossing[0]
        relative_time_around_crossing = [
            (t - datetime_orbit_ref).total_seconds() for t in time_around_crossing
        ]

        interpolator_time = InterpolatedUnivariateSpline(
            z_around_crossing, relative_time_around_crossing, k=1
        )
        relative_t_interp = interpolator_time(0.0)

        # Convert the interpolated time into datetime
        # Add the ascending node crossing time if it's before the sensing time
        datetime_ascending_crossing = datetime_orbit_ref + datetime.timedelta(
            seconds=float(relative_t_interp)
        )
        if datetime_ascending_crossing < sensing_time:
            datetime_ascending_node_crossing_list.append(datetime_ascending_crossing)

    if len(datetime_ascending_node_crossing_list) == 0:
        raise ValueError(
            "Cannot detect ascending node crossings "
            "from the orbit information provided."
        )

    # Return the most recent time for ascending node crossing
    if anx_time_annotation is None:
        anx_time_orbit = max(datetime_ascending_node_crossing_list)
    elif isinstance(anx_time_annotation, datetime.datetime):
        anx_time_orbit = min(
            datetime_ascending_node_crossing_list,
            key=lambda anxtime: abs(anxtime - anx_time_annotation),
        )
    else:
        raise ValueError(
            "datatype of `anx_time_annotation` has to be `datetime.datetime`"
        )

    return anx_time_orbit


def get_osv_list_from_orbit(
    orbit_file: str | list,
    swath_start: datetime.datetime,
    swath_stop: datetime.datetime,
):
    """
    Get the list of orbit state vectors as ElementTree (ET) objects from the orbit file `orbit_file`

    - When `orbit_file` is a list, then all of the OSV lists are
      merged and sorted. The merging is done by concatenating
      the OSVs to the "base OSV list".
    - The "base OSV list" is the one that covers
      the swath's start / stop time with padding applied,
      so that the equal time spacing is guaranteed during that time period.

    Parameters
    ----------
    orbit_file: str | list
        Orbit file name, or list of the orbit file names
    swath_start: datetime.datetime
        Sensing start time of the swath
    swath_stop: datetime.datetime
        Sensing stop time of the swath

    Returns
    -------
    orbit_state_vector_list: ET
        Orbit state vector list
    """
    if isinstance(orbit_file, str):
        orbit_tree = ET.parse(orbit_file)
        orbit_state_vector_list = orbit_tree.find("Data_Block/List_of_OSVs")
        return orbit_state_vector_list

    elif isinstance(orbit_file, list) and len(orbit_file) == 1:
        orbit_tree = ET.parse(orbit_file[0])
        orbit_state_vector_list = orbit_tree.find("Data_Block/List_of_OSVs")
        return orbit_state_vector_list

    elif isinstance(orbit_file, list) and len(orbit_file) > 1:
        # Concatenate the orbit files' OSV lists
        # Assuming that the orbit was sorted, and the last orbit in the list if the one that
        # covers the sensing start / stop with padding (i.e. the base orbit file in this function)

        # Load the base orbit file
        base_orbit_tree = ET.parse(orbit_file[-1])
        orbit_state_vector_list = base_orbit_tree.find("Data_Block/List_of_OSVs")

        for i_orbit, src_orbit_filename in enumerate(orbit_file):
            if i_orbit == len(orbit_file) - 1:
                continue
            src_orbit_tree = ET.parse(src_orbit_filename)
            src_osv_vector_list = src_orbit_tree.find("Data_Block/List_of_OSVs")
            orbit_state_vector_list = merge_osv_list(
                orbit_state_vector_list, src_osv_vector_list
            )

        return orbit_state_vector_list

    else:
        raise RuntimeError(
            "Invalid orbit information provided. "
            "It has to be filename or the list of filenames."
        )


def burst_from_xml(
    annotation_path: str,
    orbit_path: str,
    tiff_path: str,
    iw2_annotation_path: str,
    open_method=open,
    flag_apply_eap: bool = True,
):
    """Parse bursts in Sentinel-1 annotation XML.

    Parameters:
    -----------
    annotation_path : str
        Path to Sentinel-1 annotation XML file of specific subswath and
        polarization.
    orbit_path : str
        Path the orbit file.
    tiff_path : str
        Path to tiff file holding Sentinel-1 SLCs.
    iw2_annotation_path : str
        Path to Sentinel-1 annotation XML file of IW2 subswath.
    open_method : function
        Function used to open annotation file.
    flag_apply_eqp: bool
        Flag to turn on/off EAP related functionality

    Returns:
    --------
    bursts : list
        List of Sentinel1BurstSlc objects found in annotation XML.
    """
    _, tail = os.path.split(annotation_path)
    platform_id, subswath, _, pol = [x.upper() for x in tail.split("-")[:4]]
    safe_filename = os.path.basename(annotation_path.split(".SAFE")[0])

    # parse manifest.safe to retrieve IPF version
    # Also load the Product annotation - for EAP calibration and RFI information
    manifest_path = (
        os.path.dirname(annotation_path).replace("annotation", "") + "manifest.safe"
    )
    with (
        open_method(manifest_path, "r") as f_manifest,
        open_method(annotation_path, "r") as f_lads,
    ):
        tree_manifest = ET.parse(f_manifest)
        ipf_version = get_ipf_version(tree_manifest)
        # Parse out the start/end track to determine if we have an
        # equator crossing (for the burst_id calculation).
        start_track, end_track = get_start_end_track(tree_manifest)

        tree_lads = ET.parse(f_lads)
        product_annotation = ProductAnnotation.from_et(tree_lads)

        rfi_annotation_path = annotation_path.replace(
            "annotation/", "annotation/rfi/rfi-"
        )
        # Load RFI information if available
        try:
            with open_method(rfi_annotation_path, "r") as f_rads:
                tree_rads = ET.parse(f_rads)
                burst_rfi_info_swath = SwathRfiInfo.from_et(
                    tree_rads, tree_lads, ipf_version
                )

        except (FileNotFoundError, KeyError):
            if ipf_version >= RFI_INFO_AVAILABLE_FROM:
                warnings.warn(
                    f"RFI annotation expected (IPF version={ipf_version}"
                    f" >= {RFI_INFO_AVAILABLE_FROM}), but not loaded."
                )
            burst_rfi_info_swath = None

        # Load the miscellaneous metadata for CARD4L-NRB
        swath_misc_metadata = get_swath_misc_metadata(
            tree_manifest, tree_lads, product_annotation
        )

    # load the Calibraton annotation
    try:
        calibration_annotation_path = annotation_path.replace(
            "annotation/", "annotation/calibration/calibration-"
        )
        with open_method(calibration_annotation_path, "r") as f_cads:
            tree_cads = ET.parse(f_cads)
            calibration_annotation = CalibrationAnnotation.from_et(
                tree_cads, calibration_annotation_path
            )
    except (FileNotFoundError, KeyError):
        calibration_annotation = None

    # load the Noise annotation
    try:
        noise_annotation_path = annotation_path.replace(
            "annotation/", "annotation/calibration/noise-"
        )
        with open_method(noise_annotation_path, "r") as f_nads:
            tree_nads = ET.parse(f_nads)
            noise_annotation = NoiseAnnotation.from_et(
                tree_nads, ipf_version, noise_annotation_path
            )
    except (FileNotFoundError, KeyError):
        noise_annotation = None

    # load AUX_CAL annotation
    eap_necessity = is_eap_correction_necessary(ipf_version)
    if eap_necessity.phase_correction and flag_apply_eap:
        path_aux_cals = os.path.join(
            f"{os.path.dirname(s1_annotation.__file__)}", "data", "aux_cal"
        )
        path_aux_cal = get_path_aux_cal(path_aux_cals, annotation_path)

        # Raise error flag when AUX_CAL file cannot be found
        if path_aux_cal is None:
            raise FileNotFoundError(
                f"Cannot find corresponding AUX_CAL in {path_aux_cals}. "
                f"Platform: {platform_id}, inst, "
                f"config ID: {product_annotation.inst_config_id}"
            )

        subswath_id = os.path.basename(annotation_path).split("-")[1]
        aux_cal_subswath = AuxCal.load_from_zip_file(path_aux_cal, pol, subswath_id)
    else:
        # No need to load aux_cal (not applying EAP correction)
        aux_cal_subswath = None

    # Nearly all metadata loaded here is common to all bursts in annotation XML
    with open_method(annotation_path, "r") as f:
        tree = ET.parse(f)

        product_info_element = tree.find("generalAnnotation/productInformation")
        azimuth_steer_rate = np.radians(
            float(product_info_element.find("azimuthSteeringRate").text)
        )
        radar_freq = float(product_info_element.find("radarFrequency").text)
        range_sampling_rate = float(product_info_element.find("rangeSamplingRate").text)
        orbit_direction = product_info_element.find("pass").text

        image_info_element = tree.find("imageAnnotation/imageInformation")
        average_azimuth_pixel_spacing = float(
            image_info_element.find("azimuthPixelSpacing").text
        )
        azimuth_time_interval = float(
            image_info_element.find("azimuthTimeInterval").text
        )
        slant_range_time = float(image_info_element.find("slantRangeTime").text)
        ascending_node_time_annotation = as_datetime(
            image_info_element.find("ascendingNodeTime").text
        )
        first_line_utc_time = as_datetime(
            image_info_element.find("productFirstLineUtcTime").text
        )
        last_line_utc_time = as_datetime(
            image_info_element.find("productLastLineUtcTime").text
        )

        downlink_element = tree.find(
            "generalAnnotation/downlinkInformationList/downlinkInformation"
        )
        prf_raw_data = float(downlink_element.find("prf").text)
        rank = int(downlink_element.find("downlinkValues/rank").text)
        range_chirp_ramp_rate = float(
            downlink_element.find("downlinkValues/txPulseRampRate").text
        )

        n_lines = int(tree.find("swathTiming/linesPerBurst").text)
        n_samples = int(tree.find("swathTiming/samplesPerBurst").text)

        az_rate_list_element = tree.find("generalAnnotation/azimuthFmRateList")
        poly_name = "azimuthFmRatePolynomial"
        az_fm_rate_list = [
            parse_polynomial_element(x, poly_name) for x in az_rate_list_element
        ]

        doppler_list_element = tree.find("dopplerCentroid/dcEstimateList")
        poly_name = "dataDcPolynomial"
        doppler_list = [
            parse_polynomial_element(x, poly_name) for x in doppler_list_element
        ]

        rng_processing_element = tree.find(
            "imageAnnotation/processingInformation/"
            "swathProcParamsList/swathProcParams/"
            "rangeProcessing"
        )
        rng_processing_bandwidth = float(
            rng_processing_element.find("processingBandwidth").text
        )
        range_window_type = str(rng_processing_element.find("windowType").text)
        range_window_coeff = float(
            rng_processing_element.find("windowCoefficient").text
        )

        orbit_number = int(tree.find("adsHeader/absoluteOrbitNumber").text)

    wavelength = isce3.core.speed_of_light / radar_freq
    starting_range = slant_range_time * isce3.core.speed_of_light / 2
    range_pxl_spacing = isce3.core.speed_of_light / (2 * range_sampling_rate)

    # calculate the range at mid swath (mid of SM swath, mid of IW2 or mid of EW3)
    with open_method(iw2_annotation_path, "r") as iw2_f:
        iw2_tree = ET.parse(iw2_f)
        iw2_slant_range_time = float(
            iw2_tree.find("imageAnnotation/imageInformation/slantRangeTime").text
        )
        iw2_n_samples = int(iw2_tree.find("swathTiming/samplesPerBurst").text)
        iw2_starting_range = iw2_slant_range_time * isce3.core.speed_of_light / 2
        iw2_mid_range = iw2_starting_range + 0.5 * iw2_n_samples * range_pxl_spacing

    # find orbit state vectors in 'Data_Block/List_of_OSVs'
    if orbit_path:
        orbit_state_vector_list = get_osv_list_from_orbit(
            orbit_path, first_line_utc_time, last_line_utc_time
        )

        # Calculate ascending node crossing time from orbit;
        # compare with the info from annotation
        try:
            ascending_node_time_orbit = get_ascending_node_time_orbit(
                orbit_state_vector_list,
                first_line_utc_time,
                ascending_node_time_annotation,
            )
            ascending_node_time = ascending_node_time_orbit

        except ValueError:
            warnings.warn(
                "Cannot estimate ascending node crossing time from Orbit. "
                "Using the ascending node time in annotation."
            )
            ascending_node_time_orbit = None
            ascending_node_time = ascending_node_time_annotation

        # Calculate the difference in the two ANX times, and give
        # warning message when the difference is noticeable.
        if ascending_node_time_orbit is not None:
            diff_ascending_node_time_seconds = (
                ascending_node_time_orbit - ascending_node_time_annotation
            ).total_seconds()
            if (
                abs(diff_ascending_node_time_seconds)
                > ASCENDING_NODE_TIME_TOLERANCE_IN_SEC
            ):
                warnings.warn(
                    "ascending node time error larger than "
                    f"{ASCENDING_NODE_TIME_TOLERANCE_IN_SEC} seconds was detected: "
                    f"Error = {diff_ascending_node_time_seconds} seconds."
                )

    else:
        warnings.warn(
            "Orbit file was not provided. "
            "Using the ascending node time from annotation."
        )
        ascending_node_time = ascending_node_time_annotation
        orbit_state_vector_list = []

    # load individual burst
    half_burst_in_seconds = 0.5 * (n_lines - 1) * azimuth_time_interval
    burst_list_elements = tree.find("swathTiming/burstList")
    n_bursts = int(burst_list_elements.attrib["count"])
    bursts = [[]] * n_bursts

    center_pts, boundary_pts = get_burst_centers_and_boundaries(
        tree, num_bursts=n_bursts
    )

    for i, burst_list_element in enumerate(burst_list_elements):
        # Zero Doppler azimuth time of the first line of this burst
        sensing_start = as_datetime(burst_list_element.find("azimuthTime").text)
        azimuth_time_mid = sensing_start + datetime.timedelta(
            seconds=half_burst_in_seconds
        )

        # Sensing time of the first input line of this burst [UTC]
        sensing_time = as_datetime(burst_list_element.find("sensingTime").text)
        # Create the burst ID to match the ESA ID scheme
        burst_id = S1BurstId.from_burst_params(
            sensing_time, ascending_node_time, start_track, end_track, subswath
        )

        # choose nearest azimuth FM rate
        az_fm_rate = get_nearest_polynomial(azimuth_time_mid, az_fm_rate_list)

        # choose nearest doppler
        poly1d = get_nearest_polynomial(azimuth_time_mid, doppler_list)
        lut2d = doppler_poly1d_to_lut2d(
            poly1d,
            starting_range,
            range_pxl_spacing,
            (n_lines, n_samples),
            azimuth_time_interval,
        )
        doppler = Doppler(poly1d, lut2d)

        sensing_duration = datetime.timedelta(seconds=n_lines * azimuth_time_interval)
        if len(orbit_state_vector_list) > 0:
            # get orbit from state vector list/element tree

            orbit = get_burst_orbit(
                sensing_start, sensing_start + sensing_duration, orbit_state_vector_list
            )
        else:
            orbit = None

        # determine burst offset and dimensions
        # TODO move to own method
        first_valid_samples = [
            int(val) for val in burst_list_element.find("firstValidSample").text.split()
        ]
        last_valid_samples = [
            int(val) for val in burst_list_element.find("lastValidSample").text.split()
        ]

        first_valid_line = [x >= 0 for x in first_valid_samples].index(True)
        n_valid_lines = [x >= 0 for x in first_valid_samples].count(True)
        last_line = first_valid_line + n_valid_lines - 1

        first_valid_sample = max(
            first_valid_samples[first_valid_line], first_valid_samples[last_line]
        )
        last_sample = min(
            last_valid_samples[first_valid_line], last_valid_samples[last_line]
        )

        # Extract burst-wise information for Calibration, Noise, and EAP correction
        if calibration_annotation is None:
            burst_calibration = None
        else:
            burst_calibration = BurstCalibration.from_calibration_annotation(
                calibration_annotation, sensing_start
            )
        if noise_annotation is None:
            burst_noise = None
        else:
            burst_noise = BurstNoise.from_noise_annotation(
                noise_annotation,
                sensing_start,
                i * n_lines,
                (i + 1) * n_lines - 1,
                ipf_version,
            )
        if aux_cal_subswath is None:
            # Not applying EAP correction; (IPF high enough or user turned that off)
            # No need to fill in `burst_aux_cal`
            burst_aux_cal = None
        else:
            burst_aux_cal = BurstEAP.from_product_annotation_and_aux_cal(
                product_annotation, aux_cal_subswath, sensing_start
            )

        # Extended FM and DC coefficient information
        extended_coeffs = BurstExtendedCoeffs.from_polynomial_lists(
            az_fm_rate_list,
            doppler_list,
            sensing_start,
            sensing_start + sensing_duration,
        )

        # RFI information
        if burst_rfi_info_swath is None:
            burst_rfi_info = None
        else:
            burst_rfi_info = burst_rfi_info_swath.extract_by_aztime(sensing_start)

        # Miscellaneous burst metadata
        burst_misc_metadata = swath_misc_metadata.extract_by_aztime(sensing_start)

        bursts[i] = Sentinel1BurstSlc(
            ipf_version,
            sensing_start,
            radar_freq,
            wavelength,
            azimuth_steer_rate,
            average_azimuth_pixel_spacing,
            azimuth_time_interval,
            slant_range_time,
            starting_range,
            iw2_mid_range,
            range_sampling_rate,
            range_pxl_spacing,
            (n_lines, n_samples),
            az_fm_rate,
            doppler,
            rng_processing_bandwidth,
            pol,
            burst_id,
            platform_id,
            safe_filename,
            center_pts[i],
            boundary_pts[i],
            orbit,
            orbit_direction,
            orbit_number,
            tiff_path,
            i,
            first_valid_sample,
            last_sample,
            first_valid_line,
            last_line,
            range_window_type,
            range_window_coeff,
            rank,
            prf_raw_data,
            range_chirp_ramp_rate,
            burst_calibration,
            burst_noise,
            burst_aux_cal,
            extended_coeffs,
            burst_rfi_info,
            burst_misc_metadata,
        )

    return bursts


def _is_zip_annotation_xml(path: str, id_str: str) -> bool:
    """Check if path is annotation XMl and not calibration or rfi related

    path : str
        Path from SAFE zip to be checked
    id_str : str
        Subswath and polarization to be found. e.g. iw1_slc_vv

    Returns:
    --------
    _ : bool
        Whether or not given path is desired annotation XML
    """
    # break path into tokens by '/'
    tokens = path.split("/")

    # check if 2nd to last path token, directory where file resides, is "annotation"
    # check if last path token, file name, contains ID string
    if tokens[-2] == "annotation" and id_str in tokens[-1]:
        return True
    return False


def load_bursts(
    path: str,
    orbit_path: str,
    swath_num: int,
    pol: str = "vv",
    burst_ids: list[Union[str, S1BurstId]] = None,
    flag_apply_eap: bool = True,
):
    """Find bursts in a Sentinel-1 zip file or a SAFE structured directory.

    Parameters:
    -----------
    path : str
        Path to Sentinel-1 zip file or SAFE directory
    orbit_path : str
        Path the orbit file.
    swath_num : int
        Integer of subswath of desired burst. {1, 2, 3}
    pol : str
        Polarization of desired burst. {hh, vv, hv, vh}
    burst_ids : list[str] or list[S1BurstId]
        List of burst IDs for which their Sentinel1BurstSlc objects will be
        returned. Default of None returns all bursts.
        If not all burst IDs are found, a list containing found bursts will be
        returned (empty list if none are found).
    flag_apply_eap: bool
        Turn on/off EAP related features (AUX_CAL loader)

    Returns:
    --------
    bursts : list
        List of Sentinel1BurstSlc objects found in annotation XML.
    """
    if swath_num < 1 or swath_num > 3:
        raise ValueError("swath_num not <1 or >3")

    if burst_ids is None:
        burst_ids = []

    # ensure burst IDs is a list
    if isinstance(burst_ids, (str, S1BurstId)):
        burst_ids = [burst_ids]

    # lower case polarity to be consistent with file naming convention
    pol = pol.lower()
    pols = ["vv", "vh", "hh", "hv"]
    if pol not in pols:
        raise ValueError(f"polarization not in {pols}")

    id_str = f"iw{swath_num}-slc-{pol}"

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    elif os.path.isdir(path):
        bursts = _burst_from_safe_dir(path, id_str, orbit_path, flag_apply_eap)
    elif os.path.isfile(path):
        bursts = _burst_from_zip(path, id_str, orbit_path, flag_apply_eap)
    else:
        raise ValueError(f"{path} is unsupported")

    if burst_ids:
        bursts = [b for b in bursts if b.burst_id in burst_ids]

        # convert S1BurstId to string for consistent object comparison later
        burst_ids_found = set([str(b.burst_id) for b in bursts])

        # burst IDs as strings for consistent object comparison
        set_burst_ids = set([str(b) for b in burst_ids])
        if not burst_ids_found:
            warnings.warn(f"None of provided burst IDs found in sub-swath {swath_num}")
        elif burst_ids_found != set_burst_ids:
            diff = set_burst_ids.difference(burst_ids_found)
            warn_str = "Not all burst IDs found. \n "
            warn_str += f"Not found: {diff} . \n"
            warn_str += f"Found bursts: {burst_ids_found}"
            warnings.warn(warn_str)

    return bursts


def _burst_from_zip(zip_path: str, id_str: str, orbit_path: str, flag_apply_eap: bool):
    """Find bursts in a Sentinel-1 zip file.

    Parameters:
    -----------
    path : str
        Path to zip file.
    id_str: str
        Identification of desired burst. Format: iw[swath_num]-slc-[pol]
    orbit_path : str
        Path the orbit file.
    flag_apply_eap: bool
        Turn on/off EAP related features (AUX_CAL loader)

    Returns:
    --------
    bursts : list
        List of Sentinel1BurstSlc objects found in annotation XML.
    """
    with zipfile.ZipFile(zip_path, "r") as z_file:
        z_file_list = z_file.namelist()

        # find annotation file - subswath of interest
        f_annotation = [f for f in z_file_list if _is_zip_annotation_xml(f, id_str)]
        if not f_annotation:
            raise ValueError(f"burst {id_str} not in SAFE: {zip_path}")
        f_annotation = f_annotation[0]

        # find annotation file - IW2
        iw2_id_str = f"iw2-{id_str[4:]}"
        iw2_f_annotation = [
            f for f in z_file_list if _is_zip_annotation_xml(f, iw2_id_str)
        ]
        if not iw2_f_annotation:
            raise ValueError(f"burst {iw2_id_str} not in SAFE: {zip_path}")
        iw2_f_annotation = iw2_f_annotation[0]

        # find tiff file
        f_tiff = [
            f for f in z_file_list if "measurement" in f and id_str in f and "tiff" in f
        ]
        f_tiff = f"/vsizip/{zip_path}/{f_tiff[0]}" if f_tiff else ""

        bursts = burst_from_xml(
            f_annotation,
            orbit_path,
            f_tiff,
            iw2_f_annotation,
            z_file.open,
            flag_apply_eap=flag_apply_eap,
        )
        return bursts


def _burst_from_safe_dir(
    safe_dir_path: str, id_str: str, orbit_path: str, flag_apply_eap: bool
):
    """Find bursts in a Sentinel-1 SAFE structured directory.

    Parameters:
    -----------
    path : str
        Path to SAFE directory.
    id_str: str
        Identification of desired burst. Format: iw[swath_num]-slc-[pol]
    orbit_path : str
        Path the orbit file.
    flag_apply_eap: bool
        Turn on/off EAP related features (AUX_CAL loader)

    Returns:
    --------
    bursts : list
        List of Sentinel1BurstSlc objects found in annotation XML.
    """

    # find annotation file - subswath of interest
    annotation_list = os.listdir(f"{safe_dir_path}/annotation")
    f_annotation = [f for f in annotation_list if id_str in f]
    if not f_annotation:
        raise ValueError(f"burst {id_str} not in SAFE: {safe_dir_path}")
    f_annotation = f"{safe_dir_path}/annotation/{f_annotation[0]}"

    # find annotation file - IW2
    iw2_id_str = f"iw2-{id_str[4:]}"
    iw2_f_annotation = [f for f in annotation_list if iw2_id_str in f]
    if not iw2_f_annotation:
        raise ValueError(f"burst {iw2_id_str} not in SAFE: {safe_dir_path}")
    iw2_f_annotation = f"{safe_dir_path}/annotation/{iw2_f_annotation[0]}"

    # find tiff file if measurement directory found
    if os.path.isdir(f"{safe_dir_path}/measurement"):
        f_tiff = glob.glob(f"{safe_dir_path}/measurement/*{id_str}*tiff")
        f_tiff = f_tiff[0] if f_tiff else ""
    else:
        msg = f"measurement directory NOT found in {safe_dir_path}"
        msg += ", continue with metadata only."
        warnings.warn(msg)
        f_tiff = ""

    bursts = burst_from_xml(
        f_annotation,
        orbit_path,
        f_tiff,
        iw2_f_annotation,
        flag_apply_eap=flag_apply_eap,
    )
    return bursts
