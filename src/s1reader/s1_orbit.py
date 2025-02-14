from __future__ import annotations

import datetime
import glob
import os
from pathlib import Path
import warnings

from xml.etree import ElementTree
import requests

# date format used in file names
FMT = "%Y%m%dT%H%M%S"

# Orbital period of Sentinel-1 in seconds:
# 12 days * 86400.0 seconds/day, divided into 175 orbits
T_ORBIT = (12 * 86400.0) / 175.0
PADDING_SHORT = 60

# Temporal margin to apply to the start time of a frame
#  to make sure that the ascending node crossing is
#    included when choosing the orbit file
padding_short = datetime.timedelta(seconds=PADDING_SHORT)
margin_start_time = datetime.timedelta(seconds=T_ORBIT + PADDING_SHORT)

# Scihub guest credential
scihub_user = "gnssguest"
scihub_password = "gnssguest"


def retrieve_orbit_file(safe_file: str, orbit_dir: str, concatenate: bool = False):
    """
    Download or concatenate orbits for S1-A/B SAFE "safe_file"
    If no RESORB orbit file covers the time range [start_time - margin_start_time, end_time], then
    download two RESORB files to cover the time range, and
    concatenate them

    Parameters
    ----------
    safe_file: str
        File path to SAFE file for which download the orbits
    orbit_dir: str
        File path to directory where to store downloaded orbits
    concatenate : bool
        Concatenate two RESORB files if there is no single RESORB file that
        covers the time frame with the margin added

    Returns
    -------
    orbit_file : str
        Path to the orbit file.
    """

    # Create output directory & check internet connection
    os.makedirs(orbit_dir, exist_ok=True)
    _check_internet_connection()

    # Parse info from SAFE file name
    mission_id, _, sensing_start_time, sensing_end_time, _ = _parse_safe_filename(
        safe_file
    )

    # Apply margin to the start time
    search_start_time = sensing_start_time - margin_start_time
    search_end_time = sensing_end_time + padding_short

    # Find precise orbit first
    orbit_dict = _get_orbit_dict(
        mission_id, search_start_time, search_end_time, "AUX_POEORB"
    )

    # If orbit dict is empty, find restituted orbits
    if orbit_dict is None:
        orbit_dict = _get_orbit_dict(
            mission_id, search_start_time, search_end_time, "AUX_RESORB"
        )

    # Download orbit file
    if orbit_dict is not None:
        orbit_file = os.path.join(orbit_dir, f"{orbit_dict['orbit_name']}.EOF")
        if not os.path.exists(orbit_file):
            _download_orbit_file(orbit_dict["orbit_url"], orbit_file)

        return orbit_file

    # POEORB is not found, or there is no RESORB file that
    # covers the sensing period + margin at the starting time.
    # Try to find two subsequent RESORB files that covers the
    # sensing period + margins at the starting time.
    #
    # NOTE about timing design of RESORB files based on the investigation in 09/2023
    # Duration (H:M:S):                                     3:17:30
    # Overlap between the subsequent RESORB  (H:M:S):       1:38:46
    # Time shifting between the subsequent RESORB  (H:M:S): 1:38:44
    # T_orb  (H:M:S):                                       1:38:44.57

    print("Attempting to download RESORB files.")
    resorb_dict_earlier = _get_orbit_dict(
        mission_id,
        search_start_time,
        search_start_time + 2 * padding_short,
        "AUX_RESORB",
    )

    resorb_dict_later = _get_orbit_dict(
        mission_id,
        sensing_start_time - padding_short,
        sensing_end_time + padding_short,
        "AUX_RESORB",
    )

    resorb_dict_list = [resorb_dict_earlier, resorb_dict_later]

    if not all(resorb_dict_list):
        warnings.warn("Cannot find RESORB pair that covers the timeframe in the query")
        return

    # Either concatenate the orbit, or return the RESORB pair
    resorb_file_list = []
    for resorb_dict in resorb_dict_list:
        resorb_file = os.path.join(orbit_dir, f"{resorb_dict['orbit_name']}.EOF")
        if not os.path.exists(resorb_file):
            _download_orbit_file(orbit_dir, resorb_dict["orbit_url"])
        resorb_file_list.append(resorb_file)

    # concatenate the RESORB xml file.
    # NOTE Careful about the order how the RESORBs are concatenated to avoid
    # the non-uniform spacing of OSVs during the sensing times
    # 11111111111111111111111                                    2222222222222222222222
    #                2222222222222222222222      11111111111111111111111
    # 1111111111111112222222222222222222222      11111111111111111111111222222222222222
    #  |              |---sensing time---|        |               |---sensing time---|
    # ANX crossing                               ANX crossing
    # CASE 1: adding earlier RESORB to latter    CASE 2: Adding latter RESORB to earlier
    #                                            (non-uniform temporal spacing takes place
    #                                          between `1` and `2` during the sensing time)
    #
    # CASE 1 is favorable down stream processing with ISCE3 and therefore in the following
    # we concatenate the two orbits using CASE 1 approach which is
    # adding earlier RESORB to latter (i.e. CASE 1 above)
    if concatenate:
        concat_resorb_file = combine_xml_orbit_elements(
            resorb_file_list[1], resorb_file_list[0]
        )
        return concat_resorb_file

    return resorb_file_list


def _check_internet_connection():
    """
    Check connection availability
    """
    url = "http://google.com"
    try:
        requests.get(url, timeout=10)
    except (requests.ConnectionError, requests.Timeout) as exception:
        raise ConnectionError(f"Unable to reach {url}: {exception}")


def _parse_safe_filename(safe_filename):
    """
    Extract info from S1-A/B SAFE filename
    SAFE filename structure:
    S1A_IW_SLC__1SDV_20150224T114043_20150224T114111_004764_005E86_AD02.SAFE

    Parameters
    -----------
    safe_filename: string
       Path to S1-A/B SAFE file

    Returns
    -------
    List of [mission_id, mode_id, start_datetime,
                end_datetime, abs_orbit_num]
       mission_id: sensor identifier (S1A or S1B)
       mode_id: mode/beam (e.g. IW)
       start_datetime: acquisition start datetime
       stop_datetime: acquisition stop datetime
       abs_orbit_num: absolute orbit number

    Examples
    ---------
    parse_safe_filename('S1A_IW_SLC__1SDV_20150224T114043_20150224T114111_004764_005E86_AD02.SAFE')
    returns
    ['S1A', 'IW', datetime.datetime(2015, 2, 24, 11, 40, 43),\
    datetime.datetime(2015, 2, 24, 11, 41, 11), 4764]
    """

    safe_name = os.path.basename(safe_filename)
    mission_id = safe_name[:3]
    sensor_mode = safe_name[4:6]
    start_datetime = datetime.datetime.strptime(safe_name[17:32], FMT)
    end_datetime = datetime.datetime.strptime(safe_name[33:48], FMT)
    abs_orb_num = int(safe_name[49:55])

    return [mission_id, sensor_mode, start_datetime, end_datetime, abs_orb_num]


def _get_file_name_tokens(zip_path: str) -> [str, list[datetime.datetime]]:
    """Extract swath platform ID and start/stop times from SAFE zip file path.

    Parameters
    ----------
    zip_path: list[str]
        List containing orbit path strings.
        Orbit files required to adhere to naming convention found here:
        https://sentinels.copernicus.eu/documents/247904/351187/Copernicus_Sentinels_POD_Service_File_Format_Specification

    Returns
    -------
    mission_id: ('S1A', 'S1B')
    t_swath_start_stop: list[datetime.datetime]
        Swath start/stop times
    """
    mission_id, _, start_time, end_time, _ = _parse_safe_filename(zip_path)
    return mission_id, [start_time, end_time]


def _get_orbit_dict(mission_id, start_time, end_time, orbit_type):
    """
    Query Copernicus GNSS API to find latest orbit file
    Parameters
    ----------
    mission_id: str
        Sentinel satellite identifier ('S1A' or 'S1B')
    start_time: datetime object
        Starting time for search query
    end_time: datetime object
        Ending time  for search query
    orbit_type: str
        Type of orbit to download (AUX_POEORB: precise, AUX_RESORB: restituted)

    Returns
    -------
    orbit_dict: dict
        Python dictionary with [orbit_name, orbit_type, download_url]
    """
    # Required for orbit download
    scihub_url = "https://scihub.copernicus.eu/gnss/odata/v1/Products"
    # Namespaces of the XML file returned by the S1 query. Will they change it?
    m_url = "{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}"
    d_url = "{http://schemas.microsoft.com/ado/2007/08/dataservices}"

    # Check if correct orbit_type
    if orbit_type not in ["AUX_POEORB", "AUX_RESORB"]:
        err_msg = f"{orbit_type} not a valid orbit type"
        raise ValueError(err_msg)

    new_start_time = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    new_end_time = end_time.strftime("%Y-%m-%dT%H:%M:%S")
    query_string = (
        f"startswith(Name,'{mission_id}') and substringof('{orbit_type}',Name) "
        f"and ContentDate/Start lt datetime'{new_start_time}' and ContentDate/End gt datetime'{new_end_time}'"
    )
    query_params = {
        "$top": 1,
        "$orderby": "ContentDate/Start asc",
        "$filter": query_string,
    }
    query_response = requests.get(
        url=scihub_url, params=query_params, auth=(scihub_user, scihub_password)
    )
    # Parse XML tree from query response
    xml_tree = ElementTree.fromstring(query_response.content)
    # Extract w3.org URL
    w3_url = xml_tree.tag.split("feed")[0]

    # Extract orbit's name, id, url
    orbit_id = xml_tree.findtext(f".//{w3_url}entry/{m_url}properties/{d_url}Id")
    orbit_url = f"{scihub_url}('{orbit_id}')/$value"
    orbit_name = xml_tree.findtext(f"./{w3_url}entry/{w3_url}title")

    if orbit_id is not None:
        orbit_dict = {
            "orbit_name": orbit_name,
            "orbit_type": orbit_type,
            "orbit_url": orbit_url,
        }
    else:
        orbit_dict = None
    return orbit_dict


def _download_orbit_file(orbit_url: str, output_file: str) -> None:
    """
    Download S1-A/B orbits
    Parameters
    ----------
    orbit_url: str
        Remote url of orbit file to download
    output_file: str
        Path to save output file
    """
    print("downloading URL:", orbit_url)
    response = requests.get(url=orbit_url, auth=(scihub_user, scihub_password))
    response.raise_for_status()

    # Save orbits
    with open(output_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                f.flush()


def get_orbit_file_from_dir(
    zip_path: str, orbit_dir: str, auto_download: bool = False, concat_resorb=False
) -> str | list | None:
    """Get orbit state vector list for a given swath.

    Parameters
    ----------
    zip_path : string
        Path to Sentinel1 SAFE zip file. Base names required to adhere to the
        format described here:
        https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
    orbit_dir : string
        Path to directory containing orbit files. Orbit files required to adhere
        to naming convention found here:
        https://s1qc.asf.alaska.edu/aux_poeorb/
    auto_download : bool
        Automatically download the orbit file if not exist in the orbit_dir.
    concat_resorb : bool
        Concatenate two RESORB files if there is no single RESORB file that
        covers the time frame with the margin added

    Returns
    -------
    orbit_file : str
        Path to the orbit file.
    """

    # check the existence of input file path and directory
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} does not exist")

    if not os.path.isdir(orbit_dir):
        if not auto_download:
            raise NotADirectoryError(f"{orbit_dir} not found")
        else:
            print(f"{orbit_dir} not found, creating directory.")
            os.makedirs(orbit_dir, exist_ok=True)

    # search for orbit file
    orbit_file_list = glob.glob(os.path.join(orbit_dir, "S1*.EOF"))

    orbit_file = get_orbit_file_from_list(zip_path, orbit_file_list, concat_resorb)

    if orbit_file:
        return orbit_file

    if not auto_download:
        msg = (
            f"No orbit file was found for {os.path.basename(zip_path)} "
            f"from the directory provided: {orbit_dir}"
        )
        warnings.warn(msg)
        return

    # Attempt auto download
    orbit_file = retrieve_orbit_file(zip_path, orbit_dir, concat_resorb)
    return orbit_file


def get_orbit_file_from_list(
    zip_path: str, orbit_file_list: list, concat_resorb: bool = False
) -> str | list | None:
    """Get orbit file for a given S-1 swath from a list of files

    Parameters
    ----------
    zip_path : string
        Path to Sentinel1 SAFE zip file. Base names required to adhere to the
        format described here:
        https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
    orbit_file_list : list
        List of the orbit files that exists in the system.
    concat_resorb : bool
        Concatenate two RESORB files if there is no single RESORB file that
        covers the time frame with the margin added

    Returns
    -------
    orbit_file : str
        Path to the orbit file, or an empty string if no orbit file was found.
    """
    # check the existence of input file path and directory
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} does not exist")

    # extract platform id, start and end times from swath file name
    mission_id, t_swath_start_stop = _get_file_name_tokens(zip_path)

    # Apply temporal margin to the start time of the frame
    # 1st element: start time, 2nd element: end time
    # t_swath_start_stop[0] = t_swath_start_stop[0] - margin_start_time
    t_search_window = [t_swath_start_stop[0] - margin_start_time, t_swath_start_stop[1]]

    # Sort the orbit files for reproducibility in the search result
    orbit_file_list_sorted = sorted(orbit_file_list, key=lambda x: os.path.basename(x))

    # initiate output
    orbit_file_final = ""

    # search for orbit file
    for orbit_file in orbit_file_list_sorted:
        # check if file validity
        if not os.path.isfile(orbit_file):
            continue
        if mission_id not in os.path.basename(orbit_file):
            continue

        # get file name and extract state vector start/end time strings
        t_orbit_start, t_orbit_end = os.path.basename(orbit_file).split("_")[-2:]

        # strip 'V' at start of start time string
        t_orbit_start = datetime.datetime.strptime(t_orbit_start[1:], FMT)

        # string '.EOF' from end of end time string
        t_orbit_stop = datetime.datetime.strptime(t_orbit_end[:-4], FMT)

        # check if:
        # 1. swath start and stop time > orbit file start time
        # 2. swath start and stop time < orbit file stop time
        if all([t_orbit_start < t < t_orbit_stop for t in t_search_window]):
            orbit_file_final = orbit_file
            break

    if not orbit_file_final:
        msg = (
            "No single orbit file was found in the file list provided. "
            "Attempting to find set of RESORB files that covers the time period."
        )
        warnings.warn(msg)
        orbit_file_final = get_resorb_pair_from_list(
            zip_path, orbit_file_list_sorted, concat_resorb
        )

    if not orbit_file_final:
        msg = "No single orbit file was found in the file list provided."
        warnings.warn(msg)

    return orbit_file_final


def _covers_timeframe(orbit_file: str, t_start_stop_frame: list) -> bool:
    """
    Check if `orbit_file` covers `t_start_stop_frame`

    Parameters
    ----------
    orbit_file: str
        Orbit file
    t_start_stop_frame: list(datetime.datetime)
        start / stop time of the time frame as a list.

    Returns
    -------
    _: Bool
        `True` if the orbit file covers the time range; `False`, otherwise
    """
    # get file name and extract state vector start/end time strings
    t_orbit_start, t_orbit_stop = os.path.basename(orbit_file).split("_")[-2:]

    # strip 'V' at start of start time string
    t_orbit_start = datetime.datetime.strptime(t_orbit_start[1:], FMT)

    # string '.EOF' from end of end time string
    t_orbit_stop = datetime.datetime.strptime(t_orbit_stop[:-4], FMT)

    # check if:
    # 1. swath start and stop time > orbit file start time, and
    # 2. swath start and stop time < orbit file stop time
    return all(t_orbit_start < t < t_orbit_stop for t in t_start_stop_frame)


def get_resorb_pair_from_list(
    zip_path: str, orbit_file_list: list, concatenate_resorb: bool = False
) -> list | str | None:
    """
    Find if there are two RESORB files that cover [start - margin_start_time, end]
    If found, try to concatenate them when `concatenate_resorb` is `True`

    Code based on `get_orbit_file_from_list()`

    Parameters
    ----------
    zip_path : string
        Path to the Sentinel-1 SAFE zip file. Base names required to adhere to the
        format described here:
        https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
    orbit_file_list : list
        List of the orbit files that exist in the system.
    concatenate_resorb : bool
        Concatenate two RESORB files if there is no single RESORB file that
        covers the time frame with the margin added

    Returns
    -------
    orbit_file : str
        Path to the orbit file, or `None` if no orbit file was found.
    """
    # check the existence of input file path and directory
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} does not exist")

    # extract platform id, start and end times from swath file name
    mission_id, t_swath_start_stop = _get_file_name_tokens(zip_path)

    # choose the RESORB file only
    resorb_file_list = [
        orbit_file
        for orbit_file in orbit_file_list
        if "_RESORB_" in os.path.basename(orbit_file)
    ]
    pad_1min = datetime.timedelta(seconds=PADDING_SHORT)
    resorb_filename_earlier = None
    resorb_filename_later = None
    for resorb_file in resorb_file_list:
        if mission_id not in os.path.basename(resorb_file):
            continue

        # NOTE: the size of the search window was set to be small to avoid
        # the potential edge case like below:
        # 1111111111 1111111111| |
        #         |2222222222 2222222222
        #         |          3333333333 3333333333
        #         |           | |
        #         |-----------|-|
        #                 |    |
        #               T_orb  Sensing
        # In that case, query for the time window `Sensing` time window can find
        # some orbit file, but the query for `T_orb` has a risk of not finding any orbits,
        # depending on the sensing start time's relative position wrt. orbit 1, and
        # the length of the padding.

        # 1. Try to find the orbit file that covers the sensing start-stop
        # with small padding (like 60 sec.)
        t_swath_start_stop_safe = [
            t_swath_start_stop[0] - pad_1min,
            t_swath_start_stop[1] + pad_1min,
        ]
        if (
            _covers_timeframe(resorb_file, t_swath_start_stop_safe)
            and resorb_filename_later is None
        ):
            print("Found RESOEB file covering the S1 SAFE frame.")
            resorb_filename_later = resorb_file
            continue

        # 2. Try to find the orbit file that covers
        # sensing time - T_orb with small padding
        t_swath_start_stop_anx = [
            t_swath_start_stop[0] - margin_start_time,
            t_swath_start_stop[0] - margin_start_time + 2 * pad_1min,
        ]
        if (
            _covers_timeframe(resorb_file, t_swath_start_stop_anx)
            and resorb_filename_earlier is None
        ):
            print("Found RESOEB file covering ANX before sensing start")
            resorb_filename_earlier = resorb_file
            continue

        # break out of the for loop when the RESORB files are found
        if resorb_filename_earlier is not None and resorb_filename_later is not None:
            break

    # if 1. and 2. are successful return the result as a list of orbit file, or
    # as a concatenated RESORB file
    if resorb_filename_earlier is not None and resorb_filename_later is not None:
        if concatenate_resorb:
            # BE CAREFUL ABOUT THE ORDER HOW THEY ARE CONCATENATED.
            # See NOTE in retrieve_orbit_file() for detail.
            concat_resorb_filename = combine_xml_orbit_elements(
                resorb_filename_later, resorb_filename_earlier
            )
            print("RESORB Concatenation successful.")
            return concat_resorb_filename
        else:
            return [resorb_filename_earlier, resorb_filename_later]

    print("Cannot find a single RESORB file that meets the time frame criteria.")
    return


def combine_xml_orbit_elements(file1: str, file2: str) -> str:
    """Combine the orbit elements from two XML files.

    `file1` is the "base" of the output: All of the orbit state vectors from `file1` are used,
    and only the OSVs from `file2` which do fall within the time range of `file1` are included.

    Create a new .EOF file with the combined results.
    Output is named with the start_datetime and stop_datetime changed..

    Parameters
    ----------
    file1 : str
        The path to the first .EOF file.
    file2 : str
        The path to the second .EOF file.

    Returns
    -------
    _ : str
        Name of the newly created EOF file.
    """

    def get_dt(root: ElementTree.ElementTree, tag_name: str) -> datetime.datetime:
        time_str = root.find(f".//{tag_name}").text.split("=")[-1]
        return datetime.datetime.fromisoformat(time_str)

    # Parse the XML files
    tree1 = ElementTree.parse(file1)
    tree2 = ElementTree.parse(file2)

    root1 = tree1.getroot()
    root2 = tree2.getroot()

    # Extract the Validity_Start and Validity_Stop timestamps from both files
    start_time1 = get_dt(root1, "Validity_Start")
    stop_time1 = get_dt(root1, "Validity_Stop")
    start_time2 = get_dt(root2, "Validity_Start")
    stop_time2 = get_dt(root2, "Validity_Stop")

    # Determine the new Validity_Start and Validity_Stop values
    new_start_dt = min(start_time1, start_time2)
    new_stop_dt = max(stop_time1, stop_time2)

    # Update the Validity_Start and Validity_Stop timestamps in the first XML
    root1.find(".//Validity_Start").text = "UTC=" + new_start_dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    root1.find(".//Validity_Stop").text = "UTC=" + new_stop_dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    # Combine the <OSV> elements
    list_of_osvs1 = root1.find(".//List_of_OSVs")
    list_of_osvs2 = root2.find(".//List_of_OSVs")

    merge_osv_list(list_of_osvs1, list_of_osvs2)

    outfile = _generate_filename(file1, new_start_dt, new_stop_dt)
    tree1.write(outfile, encoding="UTF-8", xml_declaration=True)
    return outfile


def merge_osv_list(list_of_osvs1, list_of_osvs2):
    """
    Merge the two orbit state vector list (OSV list) in orbit files into one.
    Apply sorting to make sure the OSVs are in chronological order
    `list_of_osvs1` will be the "base OSV list" while the OSVs in
    `list_of_osvs2` not in the time range of the base OSV list will be
    appended.

    Parameters
    ---------
        list_of_osv1: ElementTree
            Based OSV list
        list_of_osve2: ElementTree
            OSV list to be augmented to the base OSV list

    Returns
    -------
        list_of_osvs1: ElementTree
            Merged OSV list
    """
    # Extract the UTC from the OSV of the first XML
    osv1_utc_list = [_get_utc_time_from_osv(osv1) for osv1 in list_of_osvs1]

    min_utc_osv1 = min(osv1_utc_list)
    max_utc_osv1 = max(osv1_utc_list)

    for osv in list_of_osvs2.findall("OSV"):
        utc_osv2 = _get_utc_time_from_osv(osv)

        if min_utc_osv1 < utc_osv2 < max_utc_osv1:
            continue
        list_of_osvs1.append(osv)

    # sort the OSVs in the conatenated OSV
    list_of_osvs1 = _sort_list_of_osv(list_of_osvs1)

    # Adjust the count attribute in <List_of_OSVs>
    new_count = len(list_of_osvs1.findall("OSV"))
    list_of_osvs1.set("count", str(new_count))

    return list_of_osvs1


def _generate_filename(
    file_base: str, new_start: datetime.datetime, new_stop: datetime.datetime
) -> str:
    """Generate a new filename based on the two provided filenames.

    Parameters
    ----------
    file_base : str
        The name of one of the concatenated files
    new_start : datetime
        The new first datetime of the updated orbital elements
    new_stop : datetime
        The new final datetime of the updated orbital elements

    Returns
    -------
    str
        Generated filename.
    """
    product_name = Path(file_base).name
    index_prefix = product_name.index("_V")
    # Example orbit file name for reference:
    # S1A_OPER_AUX_PREORB_OPOD_20200325T131800_V20200325T121452_20200325T184952
    fmt = "%Y%m%dT%H%M%S"
    new_start_stop_str = new_start.strftime(fmt) + "_" + new_stop.strftime(fmt)
    new_product_name = product_name[: index_prefix + 2] + new_start_stop_str
    return str(file_base).replace(product_name, new_product_name) + ".EOF"


def _sort_list_of_osv(list_of_osvs):
    """
    Sort the orbit state vectors (OSV) with respect to the UTC time

    Parameters
    ----------
    list_of_osvs: ET.ElementTree
        Orbit state vectors (OSVs) as XML ElementTree (ET) objects

    Returns
    -------
    list_of_osvs: ET.ElementTree
        Sorted orbit state vectors (OSVs) with respect to UTC time
    """
    utc_osv_list = [_get_utc_time_from_osv(osv) for osv in list_of_osvs]

    sorted_index_list = [
        index for index, _ in sorted(enumerate(utc_osv_list), key=lambda x: x[1])
    ]

    list_of_osvs_copy = list_of_osvs.__copy__()

    for i_osv, _ in enumerate(list_of_osvs_copy):
        index_to_replace = sorted_index_list[i_osv]
        list_of_osvs[i_osv] = list_of_osvs_copy[index_to_replace].__copy__()

    return list_of_osvs


def _get_utc_time_from_osv(osv):
    """
    Extract the UTC time from orbit state vector element in orbit file

    Parameters
    ----------
    osv: ElementTree
        orbit state vector parsed as .xml

    Returns
    -------
    datetime_utc: datetime.datetime
        Orbit state vector's UTC time
    """
    utc_osv_string = osv.find("UTC").text.replace("UTC=", "")
    datetime_utc = datetime.datetime.fromisoformat(utc_osv_string)
    return datetime_utc
