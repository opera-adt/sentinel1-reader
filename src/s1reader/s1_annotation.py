"""
A module to load annotation files for Sentinel-1 IW SLC SAFE data
To be used for the class "Sentinel1BurstSlc"
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import os
import warnings
import zipfile

from types import SimpleNamespace

import lxml.etree as ET
import numpy as np

from isce3.core import speed_of_light
from packaging import version
from s1reader.s1_orbit import T_ORBIT
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d


# Minimum IPF version from which the S1 product's Noise Annotation
# Data Set (NADS) includes azimuth noise vector annotation
min_ipf_version_az_noise_vector = version.parse("2.90")

# Minimum IPF version from which the RFI information gets available
# source: "Sentinel-1: Using the RFI annotations", reference no: MPC-0540,
# URL: (https://sentinel.esa.int/documents/247904/1653442/
#       DI-MPC-OTH-0540-1-0-RFI-Tech-Note.pdf)
RFI_INFO_AVAILABLE_FROM = version.Version("3.40")

# Dictionary of the fields in RFI information, and their data type castor
dict_datatype_rfi = {
    "swath": str,
    "azimuthTime": lambda T: datetime.datetime.strptime(T, "%Y-%m-%dT%H:%M:%S.%f"),
    "inBandOutBandPowerRatio": float,
    "percentageAffectedLines": float,
    "avgPercentageAffectedSamples": float,
    "maxPercentageAffectedSamples": float,
    "numSubBlocks": int,
    "subBlockSize": int,
    "maxPercentageAffectedBW": float,
    "percentageBlocksPersistentRfi": float,
    "maxPercentageBWAffectedPersistentRfi": float,
}


def element_to_dict(elem_in: ET, dict_tree: dict = None):
    """
    Recursively parse the element tree,
    return the results as SimpleNameSpace

    Parameters
    ----------
    elem_in: ElementTree
        Input element tree object
    dict_tree: dict
        Dictionary to be populated

    Returns
    -------
    dict_tree: dict
        A populated dictionary by `elem_in`
    """
    if dict_tree is None:
        dict_tree = {}
    key_elem = elem_in.tag
    child_elem = list(elem_in.iterchildren())

    if len(child_elem) == 0:
        # Reached the tree end
        text_elem = elem_in.text

        if key_elem in dict_datatype_rfi:
            elem_datatype = dict_datatype_rfi[key_elem]
        else:
            warnings.warn(
                f"Data type for element {key_elem} is not defined. "
                f'Casting the value "{text_elem}" as string.'
            )
            elem_datatype = str
        dict_tree[key_elem] = elem_datatype(text_elem)

    else:
        dict_tree[key_elem] = {}
        for et_child in child_elem:
            element_to_dict(et_child, dict_tree[key_elem])

    return dict_tree


@dataclass
class AnnotationBase:
    """
    A virtual base class of the inheriting annotation class i.e. Product, Calibration, and Noise.
    Not intended for standalone use.
    """

    xml_et: ET

    @classmethod
    def _parse_scalar(cls, path_field: str, str_type: str):
        """A class method that parse the scalar value in AnnotationBase.xml_et

        Parameters
        ----------
        path_field : str
            Field in the xml_et to parse
        str_type : str
            Specify how the texts in the field will be parsed.
            accepted values:
                {'datetime', 'scalar_int', 'scalar_float', 'vector_int', 'vector_float', 'str'}

        Returns
        -------
        val_out: {datetime.datetime, int, float, np.array, str}
            Parsed data in the annotation
            Datatype of vel_out follows str_type.
            val_out becomes np.array when str_type is vector*

        """

        elem_field = cls.xml_et.find(path_field)
        if str_type == "datetime":
            val_out = datetime.datetime.strptime(
                elem_field.text, "%Y-%m-%dT%H:%M:%S.%f"
            )

        elif str_type == "scalar_int":
            val_out = int(elem_field.text)

        elif str_type == "scalar_float":
            val_out = float(elem_field.text)

        elif str_type == "vector_int":
            val_out = np.array([int(strin) for strin in elem_field.text.split()])

        elif str_type == "vector_float":
            val_out = np.array([float(strin) for strin in elem_field.text.split()])

        elif str_type == "str":
            val_out = elem_field.text

        else:
            raise ValueError(f'Unsupported type the element: "{str_type}"')

        return val_out

    @classmethod
    def _parse_vectorlist(cls, name_vector_list: str, name_vector: str, str_type: str):
        """A class method that parse the list of the values from xml_et in the class

        Parameters
        ----------
        name_vector_list : str
            List Field in the xml_et to parse
        name_vector : str
            Name of the field in each elements of the VectorList
            (e.g. 'noiseLut' in 'noiseVectorList')
        str_type : str
            Specify how the texts in the field will be parsed
            accepted values:
                {'datetime', 'scalar_int', 'scalar_float', 'vector_int', 'vector_float', 'str'}

        Returns
        -------
        val_out: list
            Parsed data in the annotation

        """

        element_to_parse = cls.xml_et.find(name_vector_list)
        num_element = len(element_to_parse)

        list_out = [None] * num_element

        if str_type == "datetime":
            for i, elem in enumerate(element_to_parse):
                str_elem = elem.find(name_vector).text
                list_out[i] = datetime.datetime.strptime(
                    str_elem, "%Y-%m-%dT%H:%M:%S.%f"
                )
            list_out = np.array(list_out)

        elif str_type == "scalar_int":
            for i, elem in enumerate(element_to_parse):
                str_elem = elem.find(name_vector).text
                list_out[i] = int(str_elem)

        elif str_type == "scalar_float":
            for i, elem in enumerate(element_to_parse):
                str_elem = elem.find(name_vector).text
                list_out[i] = float(str_elem)

        elif str_type == "vector_int":
            for i, elem in enumerate(element_to_parse):
                str_elem = elem.find(name_vector).text
                list_out[i] = np.array([int(strin) for strin in str_elem.split()])

        elif str_type == "vector_float":
            for i, elem in enumerate(element_to_parse):
                str_elem = elem.find(name_vector).text
                list_out[i] = np.array([float(strin) for strin in str_elem.split()])

        elif str_type == "str":
            list_out = element_to_parse[0].find(name_vector).text

        else:
            raise ValueError(f"Cannot recognize the type of the element: {str_type}")

        return list_out


@dataclass
class CalibrationAnnotation(AnnotationBase):
    """Reader for Calibration Annotation Data Set (CADS)"""

    basename_annotation: str
    list_azimuth_time: np.ndarray
    list_line: list
    list_pixel: None
    list_sigma_nought: list
    list_beta_nought: list
    list_gamma: list
    list_dn: list

    @classmethod
    def from_et(cls, et_in: ET, path_annotation: str):
        """
        Extracts the list of calibration informaton from etree from
        the Calibration Annotation Data Set (CADS).
        Parameters:
        -----------
        et_in: ET
            ElementTree From CADS .xml file

        Returns:
        --------
        cls: CalibrationAnnotation
            Instance of CalibrationAnnotation initialized by the input parameter
        """

        cls.xml_et = et_in
        cls.basename_annotation = os.path.basename(path_annotation)

        cls.list_azimuth_time = cls._parse_vectorlist(
            "calibrationVectorList", "azimuthTime", "datetime"
        )
        cls.list_line = cls._parse_vectorlist(
            "calibrationVectorList", "line", "scalar_int"
        )
        cls.list_pixel = cls._parse_vectorlist(
            "calibrationVectorList", "pixel", "vector_int"
        )
        cls.list_sigma_nought = cls._parse_vectorlist(
            "calibrationVectorList", "sigmaNought", "vector_float"
        )
        cls.list_beta_nought = cls._parse_vectorlist(
            "calibrationVectorList", "betaNought", "vector_float"
        )
        cls.list_gamma = cls._parse_vectorlist(
            "calibrationVectorList", "gamma", "vector_float"
        )
        cls.list_dn = cls._parse_vectorlist(
            "calibrationVectorList", "dn", "vector_float"
        )

        return cls


@dataclass
class NoiseAnnotation(AnnotationBase):
    """
    Reader for Noise Annotation Data Set (NADS) for IW SLC
    Based on ESA documentation: "Thermal Denoising of Products Generated by the S-1 IPF"
    """

    basename_annotation: str
    rg_list_azimuth_time: np.ndarray
    rg_list_line: list
    rg_list_pixel: list
    rg_list_noise_range_lut: list
    az_first_azimuth_line: int
    az_first_range_sample: int
    az_last_azimuth_line: int
    az_last_range_sample: int
    az_line: np.ndarray
    az_noise_azimuth_lut: np.ndarray

    @classmethod
    def from_et(cls, et_in: ET, ipf_version: version.Version, path_annotation: str):
        """
        Extracts list of noise information from etree

        Parameter
        ----------
        et_in : xml.etree.ElementTree
            Parsed NADS annotation .xml

        Return
        -------
        cls: NoiseAnnotation
            Parsed NADS from et_in
        """

        cls.xml_et = et_in
        cls.basename_annotation = os.path.basename(path_annotation)

        if ipf_version < min_ipf_version_az_noise_vector:  # legacy SAFE data
            cls.rg_list_azimuth_time = cls._parse_vectorlist(
                "noiseVectorList", "azimuthTime", "datetime"
            )
            cls.rg_list_line = cls._parse_vectorlist(
                "noiseVectorList", "line", "scalar_int"
            )
            cls.rg_list_pixel = cls._parse_vectorlist(
                "noiseVectorList", "pixel", "vector_int"
            )
            cls.rg_list_noise_range_lut = cls._parse_vectorlist(
                "noiseVectorList", "noiseLut", "vector_float"
            )

            cls.az_first_azimuth_line = None
            cls.az_first_range_sample = None
            cls.az_last_azimuth_line = None
            cls.az_last_range_sample = None
            cls.az_line = None
            cls.az_noise_azimuth_lut = None

        else:
            cls.rg_list_azimuth_time = cls._parse_vectorlist(
                "noiseRangeVectorList", "azimuthTime", "datetime"
            )
            cls.rg_list_line = cls._parse_vectorlist(
                "noiseRangeVectorList", "line", "scalar_int"
            )
            cls.rg_list_pixel = cls._parse_vectorlist(
                "noiseRangeVectorList", "pixel", "vector_int"
            )
            cls.rg_list_noise_range_lut = cls._parse_vectorlist(
                "noiseRangeVectorList", "noiseRangeLut", "vector_float"
            )
            cls.az_first_azimuth_line = cls._parse_vectorlist(
                "noiseAzimuthVectorList", "firstAzimuthLine", "scalar_int"
            )[0]
            cls.az_first_range_sample = cls._parse_vectorlist(
                "noiseAzimuthVectorList", "firstRangeSample", "scalar_int"
            )[0]
            cls.az_last_azimuth_line = cls._parse_vectorlist(
                "noiseAzimuthVectorList", "lastAzimuthLine", "scalar_int"
            )[0]
            cls.az_last_range_sample = cls._parse_vectorlist(
                "noiseAzimuthVectorList", "lastRangeSample", "scalar_int"
            )[0]
            cls.az_line = cls._parse_vectorlist(
                "noiseAzimuthVectorList", "line", "vector_int"
            )[0]
            cls.az_noise_azimuth_lut = cls._parse_vectorlist(
                "noiseAzimuthVectorList", "noiseAzimuthLut", "vector_float"
            )[0]

        return cls


@dataclass
class ProductAnnotation(AnnotationBase):
    """
    Reader for L1 Product annotation for IW SLC
    For Elevation Antenna Pattern (EAP) correction
    """

    image_information_slant_range_time: float

    # Attributes to be used when determining what AUX_CAL to load
    instrument_cfg_id: int

    # elevation_angle:
    antenna_pattern_azimuth_time: list
    antenna_pattern_slant_range_time: list
    antenna_pattern_elevation_angle: list
    antenna_pattern_elevation_pattern: list
    antenna_pattern_incidence_angle: list

    ascending_node_time: datetime.datetime
    number_of_samples: int
    range_sampling_rate: float

    slant_range_time: float

    @classmethod
    def from_et(cls, et_in: ET):
        """
        Extracts list of product information from etree from
        L1 annotation data set (LADS) Parameter

        ----------
        et_in : xml.etree.ElementTree
            Parsed LADS annotation .xml

        Return
        -------
        cls: ProductAnnotation
            Parsed LADS from et_in
        """

        cls.xml_et = et_in

        cls.antenna_pattern_azimuth_time = cls._parse_vectorlist(
            "antennaPattern/antennaPatternList", "azimuthTime", "datetime"
        )
        cls.antenna_pattern_slant_range_time = cls._parse_vectorlist(
            "antennaPattern/antennaPatternList", "slantRangeTime", "vector_float"
        )
        cls.antenna_pattern_elevation_angle = cls._parse_vectorlist(
            "antennaPattern/antennaPatternList", "elevationAngle", "vector_float"
        )
        cls.antenna_pattern_elevation_pattern = cls._parse_vectorlist(
            "antennaPattern/antennaPatternList", "elevationPattern", "vector_float"
        )

        cls.antenna_pattern_incidence_angle = cls._parse_vectorlist(
            "antennaPattern/antennaPatternList", "incidenceAngle", "vector_float"
        )

        cls.image_information_slant_range_time = cls._parse_scalar(
            "imageAnnotation/imageInformation/slantRangeTime", "scalar_float"
        )
        cls.ascending_node_time = cls._parse_scalar(
            "imageAnnotation/imageInformation/ascendingNodeTime", "datetime"
        )
        cls.number_of_samples = cls._parse_scalar(
            "imageAnnotation/imageInformation/numberOfSamples", "scalar_int"
        )
        cls.range_sampling_rate = cls._parse_scalar(
            "generalAnnotation/productInformation/rangeSamplingRate", "scalar_float"
        )
        cls.slant_range_time = cls._parse_scalar(
            "imageAnnotation/imageInformation/slantRangeTime", "scalar_float"
        )

        cls.inst_config_id = cls._parse_scalar(
            "generalAnnotation/downlinkInformationList/downlinkInformation/"
            "downlinkValues/instrumentConfigId",
            "scalar_int",
        )

        return cls


@dataclass
class AuxCal(AnnotationBase):
    """AUX_CAL information for EAP correction"""

    beam_nominal_near_range: float
    beam_nominal_far_range: float

    elevation_angle_increment: float
    elevation_antenna_pattern: np.ndarray

    azimuth_angle_increment: float
    azimuth_antenna_pattern: np.ndarray
    azimuth_antenna_element_pattern_increment: float
    azimuth_antenna_element_pattern: float
    absolute_calibration_constant: float
    noise_calibration_factor: float

    @classmethod
    def load_from_zip_file(cls, path_aux_cal_zip: str, pol: str, str_swath: str):
        """
        A class method that extracts list of information AUX_CAL from the input ET.

        Parameters
        ---------
        path_aux_cal_zip : str
            Path to the AUX_CAL .zip file
        pol: str {'vv','vh','hh','hv'}
            Polarization of interest
        str_swath: {'iw1','iw2','iw3'}
            IW subswath of interest

        Returns
        -------
        cls: AuxCal class populated by et_in in the parameter

        """

        if not path_aux_cal_zip.endswith(".zip"):
            raise ValueError("Only AUX_CAL files in .zip format are accepted.")

        if os.path.exists(path_aux_cal_zip):
            str_safe_aux_cal = os.path.basename(path_aux_cal_zip).replace(".zip", "")
            # detect the platform from path_aux_cal_zip
            str_platform = str_safe_aux_cal.split("_")[0]
        else:
            raise ValueError(f"Cannot find AUX_CAL .zip file: {path_aux_cal_zip}")

        with zipfile.ZipFile(path_aux_cal_zip, "r") as zipfile_aux_cal:
            filepath_xml = f"{str_safe_aux_cal}/data/{str_platform.lower()}-aux-cal.xml"
            # check if the input file has the aux_cal .xml file to load
            list_files_in_zip = [zf.filename for zf in zipfile_aux_cal.filelist]

            if filepath_xml not in list_files_in_zip:
                raise ValueError(
                    f"Cannot find {filepath_xml} in "
                    f"zip file {path_aux_cal_zip}.\n"
                    "Make sure if the legit AUX_CAL .zip file is provided."
                )

            with zipfile_aux_cal.open(filepath_xml, "r") as f_aux_cal:
                et_in = ET.parse(f_aux_cal)

        calibration_params_list = et_in.find("calibrationParamsList")
        for calibration_params in calibration_params_list:
            swath_xml = calibration_params.find("swath").text
            polarisation_xml = calibration_params.find("polarisation").text
            if polarisation_xml == pol.upper() and swath_xml == str_swath.upper():
                cls.beam_nominal_near_range = float(
                    calibration_params.find(
                        "elevationAntennaPattern/beamNominalNearRange"
                    ).text
                )
                cls.beam_nominal_far_range = float(
                    calibration_params.find(
                        "elevationAntennaPattern/beamNominalFarRange"
                    ).text
                )
                cls.elevation_angle_increment = float(
                    calibration_params.find(
                        "elevationAntennaPattern/elevationAngleIncrement"
                    ).text
                )

                n_val = int(
                    calibration_params.find("elevationAntennaPattern/values").attrib[
                        "count"
                    ]
                )
                arr_eap_val = np.array(
                    [
                        float(token_val)
                        for token_val in calibration_params.find(
                            "elevationAntennaPattern/values"
                        ).text.split()
                    ]
                )

                if n_val == len(arr_eap_val):
                    # Provided in real numbers: In case of AUX_CAL for old IPFs.
                    cls.elevation_antenna_pattern = arr_eap_val
                elif n_val * 2 == len(arr_eap_val):
                    # Provided in complex numbers: In case of recent IPFs e.g. 3.10
                    cls.elevation_antenna_pattern = (
                        arr_eap_val[0::2] + arr_eap_val[1::2] * 1.0j
                    )
                else:
                    raise ValueError(
                        "The number of values does not match. "
                        f"n_val={n_val}, "
                        f"#len(elevationAntennaPattern/values)={len(arr_eap_val)}"
                    )

                cls.azimuth_angle_increment = float(
                    calibration_params.find(
                        "azimuthAntennaPattern/azimuthAngleIncrement"
                    ).text
                )
                cls.azimuth_antenna_pattern = np.array(
                    [
                        float(token_val)
                        for token_val in calibration_params.find(
                            "azimuthAntennaPattern/values"
                        ).text.split()
                    ]
                )

                cls.azimuth_antenna_element_pattern_increment = float(
                    calibration_params.find(
                        "azimuthAntennaElementPattern/azimuthAngleIncrement"
                    ).text
                )
                cls.azimuth_antenna_element_pattern = np.array(
                    [
                        float(token_val)
                        for token_val in calibration_params.find(
                            "azimuthAntennaElementPattern/values"
                        ).text.split()
                    ]
                )

                cls.absolute_calibration_constant = float(
                    calibration_params.find("absoluteCalibrationConstant").text
                )
                cls.noise_calibration_factor = float(
                    calibration_params.find("noiseCalibrationFactor").text
                )

        return cls


@dataclass
class SwathRfiInfo:
    """
    Burst RFI information in a swath
    Reference documentation: "Sentinel-1: Using the RFI annotations" by
    G.Hajduch et al.

    url = "https://sentinel.esa.int/documents/247904/1653442/
           DI-MPC-OTH-0540-1-0-RFI-Tech-Note.pdf/
           4b4fa95d-039f-5c78-fb90-06d307b3c13a?t=1644988601315"
    """

    # RFI info in the product annotation
    rfi_mitigation_performed: str
    rfi_mitigation_domain: str

    # RFI info in the RFI annotation
    rfi_burst_report_list: list
    azimuth_time_list: list

    @classmethod
    def from_et(cls, et_rfi: ET, et_product: ET, ipf_version: version.Version):
        """Load RFI information from etree

        Parameters
        ----------
        et_rfi: ET
            XML ElementTree from RFI annotation
        et_product: ET
            XML ElementTree from product annotation
        ipf_version: version.Version
            IPF version of the input sentinel-1 data

        Returns
        -------
        cls: SwathRfiInfo
            dataclass populated by this function
        """

        if ipf_version < RFI_INFO_AVAILABLE_FROM:
            # RFI related processing is not in place
            # return an empty dataclass
            return None

        # Attempt to locate the RFI information from the input annotations
        header_lads = et_product.find("imageAnnotation/processingInformation")
        if header_lads is None:
            raise ValueError(
                "Cannot locate the element in the product "
                "anotation where RFI mitigation info is located."
            )

        header_rfi = et_rfi.find("rfiBurstReportList")
        if header_rfi is None:
            raise ValueError("Cannot locate `rfiBurstReportList` in the RFI annotation")

        # Start to load RFI information
        cls.rfi_mitigation_performed = header_lads.find("rfiMitigationPerformed").text
        cls.rfi_mitigation_domain = header_lads.find("rfiMitigationDomain").text

        num_burst_rfi_report = len(header_rfi)
        cls.rfi_burst_report_list = [None] * num_burst_rfi_report
        cls.azimuth_time_list = [None] * num_burst_rfi_report

        for i_burst, elem_burst in enumerate(header_rfi):
            cls.rfi_burst_report_list[i_burst] = element_to_dict(elem_burst)[
                "rfiBurstReport"
            ]
            cls.azimuth_time_list[i_burst] = cls.rfi_burst_report_list[i_burst][
                "azimuthTime"
            ]

        return cls

    @classmethod
    def extract_by_aztime(cls, aztime_start: datetime.datetime):
        """
        Extract the burst RFI report that is within the azimuth time of a burst

        Parameters
        ----------
        aztime_start: datetime.datetime
            Starting azimuth time of a burst

        Returns
        -------
        rfi_info: SimpleNamespace
            A SimpleNamespace that contains the burst RFI report as a dictionary,
            along with the RFI related information from the product annotation
        """

        # find the corresponding burst
        index_burst = closest_block_to_azimuth_time(
            np.array(cls.azimuth_time_list), aztime_start
        )

        burst_report_out = cls.rfi_burst_report_list[index_burst]

        rfi_info = SimpleNamespace()
        rfi_info.rfi_mitigation_performed = cls.rfi_mitigation_performed
        rfi_info.rfi_mitigation_domain = cls.rfi_mitigation_domain
        rfi_info.rfi_burst_report = burst_report_out

        return rfi_info


@dataclass
class SwathMiscMetadata:
    """
    Miscellaneous metadata
    """

    azimuth_looks: int
    slant_range_looks: int
    aztime_vec: np.ndarray
    inc_angle_list: list

    # Processing data from manifest
    slc_post_processing: dict

    def extract_by_aztime(self, aztime_start: datetime.datetime):
        """
        Extract the miscellaneous metadata for a burst that
        corresponds to `aztime_start`

        Parameters
        ----------
        aztime_start: datetime.datetime
            Starting azimuth time of a burst

        Returns
        -------
        burst_misc_metadata: SimpleNamespace
            A SimpleNamespace that contains the misc. metadata
        """
        index_burst = closest_block_to_azimuth_time(self.aztime_vec, aztime_start)
        inc_angle_burst = self.inc_angle_list[index_burst]

        burst_misc_metadata = SimpleNamespace()

        # Metadata names to be populated into OPERA products as
        # the source data's processing information
        keys_misc_metadata = ["stop", "country", "organisation", "site"]
        for key_metadata in keys_misc_metadata:
            if key_metadata not in self.slc_post_processing:
                self.slc_post_processing[key_metadata] = (
                    "Not available in sentinel-1 manifest.safe"
                )

        burst_misc_metadata.processing_info_dict = self.slc_post_processing
        burst_misc_metadata.azimuth_looks = self.azimuth_looks
        burst_misc_metadata.slant_range_looks = self.slant_range_looks
        burst_misc_metadata.inc_angle_near_range = inc_angle_burst[0]
        burst_misc_metadata.inc_angle_far_range = inc_angle_burst[-1]

        return burst_misc_metadata


def closest_block_to_azimuth_time(
    vector_azimuth_time: np.ndarray, azimuth_time_burst: datetime.datetime
) -> int:
    """
    Find the id of the closest data block in annotation.
    To be used when populating BurstNoise, BurstCalibration, and BurstEAP.

    Parameters
    ----------
    vector_azimuth_time : np.ndarray
        numpy array azimuth time whose data type is datetime.datetime
    azimuth_time_burst: datetime.datetime
        Azimuth time of the burst

    Returns
    -------
    _: int
        Index of vector_azimuth_time that is the closest to azimuth_burst_time

    """

    return np.argmin(np.abs(vector_azimuth_time - azimuth_time_burst))


@dataclass
class BurstNoise:
    """Noise correction information for Sentinel-1 burst"""

    basename_nads: str
    range_azimuth_time: datetime.datetime
    range_line: float
    range_pixel: np.ndarray
    range_lut: np.ndarray

    azimuth_first_azimuth_line: int
    azimuth_first_range_sample: int
    azimuth_last_azimuth_line: int
    azimuth_last_range_sample: int
    azimuth_line: np.ndarray
    azimuth_lut: np.ndarray

    line_from: int
    line_to: int

    @classmethod
    def from_noise_annotation(
        cls,
        noise_annotation: NoiseAnnotation,
        azimuth_time: datetime.datetime,
        line_from: int,
        line_to: int,
        ipf_version: version.Version,
    ):
        """
        Extracts the noise correction information for
        individual burst from NoiseAnnotation

        Parameters
        ----------
        noise_annotation: NoiseAnnotation
            Subswath-wide noise annotation information
        azimuth_time : datetime.datetime
            Azimuth time of the burst
        line_from: int
            First line of the burst in the subswath
        line_to: int
            Last line of the burst in the subswath
        ipf_version: float
            IPF version of the SAFE data

        Returns
        -------
        cls: BurstNoise
            Instance of BurstNoise initialized by the input parameters

        """

        basename_nads = noise_annotation.basename_annotation
        id_closest = closest_block_to_azimuth_time(
            noise_annotation.rg_list_azimuth_time, azimuth_time
        )

        range_azimuth_time = noise_annotation.rg_list_azimuth_time[id_closest]
        range_line = noise_annotation.rg_list_line[id_closest]
        range_pixel = noise_annotation.rg_list_pixel[id_closest]
        range_lut = noise_annotation.rg_list_noise_range_lut[id_closest]

        azimuth_first_azimuth_line = noise_annotation.az_first_azimuth_line
        azimuth_first_range_sample = noise_annotation.az_first_range_sample
        azimuth_last_azimuth_line = noise_annotation.az_last_azimuth_line
        azimuth_last_range_sample = noise_annotation.az_last_range_sample

        if ipf_version >= min_ipf_version_az_noise_vector:
            # Azimuth noise LUT exists - crop to the extent of the burst
            id_top = np.argmin(np.abs(noise_annotation.az_line - line_from))
            id_bottom = np.argmin(np.abs(noise_annotation.az_line - line_to))

            # put some margin when possible
            if id_top > 0:
                id_top -= 1
            if id_bottom < len(noise_annotation.az_line) - 1:
                id_bottom += 1
            azimuth_line = noise_annotation.az_line[id_top : id_bottom + 1]
            azimuth_lut = noise_annotation.az_noise_azimuth_lut[id_top : id_bottom + 1]

        else:
            azimuth_line = None
            azimuth_lut = None

        return cls(
            basename_nads,
            range_azimuth_time,
            range_line,
            range_pixel,
            range_lut,
            azimuth_first_azimuth_line,
            azimuth_first_range_sample,
            azimuth_last_azimuth_line,
            azimuth_last_range_sample,
            azimuth_line,
            azimuth_lut,
            line_from,
            line_to,
        )

    def compute_thermal_noise_lut(self, shape_lut):
        """
        Calculate thermal noise LUT whose shape is `shape_lut`

        Parameter:
        ----------
        shape_lut: tuple or list
            Shape of the output LUT

        Returns
        -------
        arr_lut_total: np.ndarray
            2d array containing thermal noise correction look up table values
        """

        nrows, ncols = shape_lut

        # Interpolate the range noise vector
        rg_lut_interp_obj = InterpolatedUnivariateSpline(
            self.range_pixel, self.range_lut, k=1
        )
        if self.azimuth_last_range_sample is not None:
            vec_rg = np.arange(self.azimuth_last_range_sample + 1)
        else:
            vec_rg = np.arange(ncols)
        rg_lut_interpolated = rg_lut_interp_obj(vec_rg)

        # Interpolate the azimuth noise vector
        if (self.azimuth_line is None) or (self.azimuth_lut is None):
            az_lut_interpolated = np.ones(nrows)
        else:  # IPF >= 2.90
            az_lut_interp_obj = InterpolatedUnivariateSpline(
                self.azimuth_line, self.azimuth_lut, k=1
            )
            vec_az = np.arange(self.line_from, self.line_to + 1)
            az_lut_interpolated = az_lut_interp_obj(vec_az)

        arr_lut_total = np.matmul(
            az_lut_interpolated[..., np.newaxis], rg_lut_interpolated[np.newaxis, ...]
        )

        return arr_lut_total


@dataclass
class BurstCalibration:
    """Calibration information for Sentinel-1 IW SLC burst"""

    basename_cads: str
    azimuth_time: datetime.datetime = None
    line: float = None
    pixel: np.ndarray = None
    sigma_naught: np.ndarray = None
    beta_naught: np.ndarray = None
    gamma: np.ndarray = None
    dn: np.ndarray = None

    @classmethod
    def from_calibration_annotation(
        cls,
        calibration_annotation: CalibrationAnnotation,
        azimuth_time: datetime.datetime,
    ):
        """
        A class method that extracts the calibration info for the burst

        Parameters
        ----------
        calibration_annotation: CalibrationAnnotation
            A subswath-wide calibraion information from CADS file
        azimuth_time: datetime.datetime
            Azimuth time of the burst

        Returns
        -------
        cls: BurstCalibration
            Radiometric correction information for the burst
        """

        basename_cads = calibration_annotation.basename_annotation
        id_closest = closest_block_to_azimuth_time(
            calibration_annotation.list_azimuth_time, azimuth_time
        )

        azimuth_time = calibration_annotation.list_azimuth_time[id_closest]
        line = calibration_annotation.list_line[id_closest]
        pixel = calibration_annotation.list_pixel[id_closest]
        sigma_naught = calibration_annotation.list_sigma_nought[id_closest]
        gamma = calibration_annotation.list_gamma[id_closest]
        dn = calibration_annotation.list_dn[id_closest]

        # Check if all values in the list of beta_naught LUTs are the same
        matrix_beta_naught = np.array(calibration_annotation.list_beta_nought)
        if matrix_beta_naught.min() == matrix_beta_naught.max():
            beta_naught = np.min(matrix_beta_naught)
        else:
            # TODO Switch to LUT-based method when there is significant changes in the array
            beta_naught = np.mean(matrix_beta_naught)

        return cls(
            basename_cads,
            azimuth_time,
            line,
            pixel,
            sigma_naught,
            beta_naught,
            gamma,
            dn,
        )


@dataclass
class BurstEAP:
    """EAP correction information for Sentinel-1 IW SLC burst"""

    # from LADS
    freq_sampling: float  # range sampling rate
    eta_start: datetime.datetime
    tau_0: float  # imageInformation/slantRangeTime
    tau_sub: np.ndarray  # antennaPattern/slantRangeTime
    theta_sub: np.ndarray  # antennaPattern/elevationAngle
    azimuth_time: datetime.datetime
    ascending_node_time: datetime.datetime

    # from AUX_CAL
    gain_eap: np.ndarray  # elevationAntennaPattern
    delta_theta: float  # elavationAngleIncrement

    @classmethod
    def from_product_annotation_and_aux_cal(
        cls,
        product_annotation: ProductAnnotation,
        aux_cal: AuxCal,
        azimuth_time: datetime.datetime,
    ):
        """
        A class method that extracts the EAP correction info for the IW SLC burst

        Parameters
        ----------
        product_annotation: ProductAnnotation
            A swath-wide product annotation class

        aux_cal: AuxCal
            AUX_CAL information that corresponds to the sensing time

        azimuth_time: datetime.datetime
            Azimuth time of the burst

        Returns
        -------
        cls: BurstEAP
            A burst-wide information for EAP correction

        """
        id_closest = closest_block_to_azimuth_time(
            product_annotation.antenna_pattern_azimuth_time, azimuth_time
        )
        freq_sampling = product_annotation.range_sampling_rate
        eta_start = azimuth_time
        tau_0 = product_annotation.slant_range_time
        tau_sub = product_annotation.antenna_pattern_slant_range_time[id_closest]
        theta_sub = product_annotation.antenna_pattern_elevation_angle[id_closest]
        gain_eap = aux_cal.elevation_antenna_pattern
        delta_theta = aux_cal.elevation_angle_increment

        ascending_node_time = product_annotation.ascending_node_time

        return cls(
            freq_sampling,
            eta_start,
            tau_0,
            tau_sub,
            theta_sub,
            azimuth_time,
            ascending_node_time,
            gain_eap,
            delta_theta,
        )

    def compute_eap_compensation_lut(self, num_sample):
        """
        Returns LUT for EAP compensation whose size is `num_sample`.
        Based on ESA docuemnt :
        "Impact of the Elevation Antenna Pattern Phase Compensation
         on the Interferometric Phase Preservation"

        Document URL:
        https://sentinel.esa.int/documents/247904/1653440/Sentinel-1-IPF_EAP_Phase_correction

        Parameter:
        num_sample: int
            Size of the output LUT

        Return:
        -------
            gain_eap_interpolatd: EAP phase for the burst to be compensated

        """

        n_elt = len(self.gain_eap)

        theta_am = (np.arange(n_elt) - (n_elt - 1) / 2) * self.delta_theta

        delta_anx = self.eta_start - self.ascending_node_time
        theta_offnadir = self._anx2roll(
            delta_anx.seconds + delta_anx.microseconds * 1.0e-6
        )

        theta_eap = theta_am + theta_offnadir

        tau = self.tau_0 + np.arange(num_sample) / self.freq_sampling

        theta = np.interp(tau, self.tau_sub, self.theta_sub)

        interpolator_gain = interp1d(theta_eap, self.gain_eap)
        gain_eap_interpolated = interpolator_gain(theta)

        return gain_eap_interpolated

    def _anx2roll(self, delta_anx):
        """
        Returns the Platform nominal roll as function of elapsed time from
        ascending node crossing time (ANX). (Implemented from S1A documentation.)

        Code copied from ISCE2.

        The units in this function is based on the reference documentation in the URL below:
        https://sentinel.esa.int/documents/247904/1653440/Sentinel-1-IPF_EAP_Phase_correction

        Parameters
        ----------
        delta_anx: float
            elapsed time from ascending node crossing time

        Returns
        -------
        nominal_roll: float
            Estimated nominal roll (degrees)
        """

        # Estimate altitude based on time elapsed since ANX
        altitude = self._anx2height(delta_anx)

        # Reference altitude (km)
        href = 711.700

        # Reference boresight at reference altitude (degrees)
        boresight_ref = 29.450

        # Partial derivative of roll vs altitude (degrees/m)
        alpha_roll = 0.0566

        # Estimate nominal roll i.e. theta off nadir (degrees)
        nominal_roll = boresight_ref - alpha_roll * (altitude / 1000.0 - href)

        return nominal_roll

    @classmethod
    def _anx2height(cls, delta_anx):
        """
        Returns the platform nominal height as function of elapse time from
        ascending node crossing time (ANX).
        Implementation from S1A documention.

        Code copied from ISCE2.


        Parameters:
        -----------
        delta_anx: float
            elapsed time from ANX time


        Returns:
        --------
        h_t: float
            nominal height of the platform

        """

        # Average height (m)
        h_0 = 707714.8  # ;m

        # Perturbation amplitudes (m)
        h = np.array([8351.5, 8947.0, 23.32, 11.74])

        # Perturbation phases (radians)
        phi = np.array([3.1495, -1.5655, -3.1297, 4.7222])

        # Angular velocity (rad/sec)
        worb = 2 * np.pi / T_ORBIT

        # Evaluation of series
        h_t = h_0
        for i, h_i in enumerate(h):
            h_t += h_i * np.sin((i + 1) * worb * delta_anx + phi[i])

        return h_t


@dataclass
class BurstExtendedCoeffs:
    """
    Segments of FM rate / Doppler centroid polynomial coefficients.
    For (linear) interpolation of FM rate / Doppler Centroid along azimuth.
    To be used for calculating azimuth FM rate mismatch mitigation
    """

    # FM rate
    fm_rate_aztime_vec: np.ndarray
    fm_rate_coeff_arr: np.ndarray
    fm_rate_tau0_vec: np.ndarray

    # Doppler centroid
    dc_aztime_vec: np.ndarray
    dc_coeff_arr: np.ndarray
    dc_tau0_vec: np.ndarray

    @classmethod
    def from_polynomial_lists(
        cls,
        az_fm_rate_list: list,
        doppler_centroid_list: list,
        sensing_start: datetime.datetime,
        sensing_end: datetime.datetime,
    ):
        """
        Extract coefficients from the list of the polynomial lists that fall within
        the provided sensing start / end times of a burst.

        Parameters:
        -----------
        az_fm_rate_list: list[isce3.core.Poly1d]
            List of azimuth FM rate polynomials
        doppler_centroid_list: list[isce3.core.Poly1d]
            List of doppler centroid polynomials
        sensing_start: datetime.datetime
            Azimuth start time of the burst
        sensing_end: datetime.datetime
            Azimuth end time of the burst
        """

        # Extract polynomial info for azimuth FM rate
        (fm_rate_aztime_burst_vec, fm_rate_coeff_burst_arr, fm_rate_tau0_burst_vec) = (
            cls.extract_polynomial_sequence(
                az_fm_rate_list, sensing_start, sensing_end, handle_out_of_range=True
            )
        )

        (dc_aztime_burst_vec, dc_coeff_burst_arr, dc_tau0_burst_vec) = (
            cls.extract_polynomial_sequence(
                doppler_centroid_list,
                sensing_start,
                sensing_end,
                handle_out_of_range=True,
            )
        )

        return cls(
            fm_rate_aztime_burst_vec,
            fm_rate_coeff_burst_arr,
            fm_rate_tau0_burst_vec,
            dc_aztime_burst_vec,
            dc_coeff_burst_arr,
            dc_tau0_burst_vec,
        )

    @classmethod
    def extract_polynomial_sequence(
        cls,
        polynomial_list: list,
        datetime_start: datetime.datetime,
        datetime_end: datetime.datetime,
        handle_out_of_range=True,
    ):
        """
        Scan `vec_azimuth_time` end find indices of the vector
        that covers the period defined with
        `datetime_start` and `datetime_end`

        Parameters:
        -----------
        polynomial_list: list
            list of (azimuth_time, isce3.core.Poly1d)
        datetime_start: datetime.datetime
            Start time of the period
        datetime_end: datetime.datetime
            end time of the period

        Returns:
        --------
        tuple
            Tuple of (vec_aztime_sequence,
                      arr_coeff_sequence,
                      vec_tau0_sequence)
            as a sequence of polynomial info that covers the period
            defined in the parameters.
            vec_aztime_sequence: azimuth time of each sample in the sequence
            arr_coeff_sequence: N by 3 npy array whose row is coefficients of
                                each sample in the sequence
            vec_tau0_sequence: Range start time of each sample in the sequence
        """

        # NOTE: dt is defined as: [azimuth time] - [start/end time]
        # find index of poly time closest to start time that is less than start time
        dt_wrt_start = np.array(
            [(poly[0] - datetime_start).total_seconds() for poly in polynomial_list]
        )
        dt_wrt_start = np.ma.masked_array(dt_wrt_start, mask=dt_wrt_start > 0)
        index_start = np.argmax(dt_wrt_start)

        # find index of poly time closest to end time that is greater than end time
        dt_wrt_end = np.array(
            [(poly[0] - datetime_end).total_seconds() for poly in polynomial_list]
        )
        dt_wrt_end = np.ma.masked_array(dt_wrt_end, mask=dt_wrt_end < 0)
        index_end = np.argmin(dt_wrt_end)

        # Handle the case that the burst's sensing period exceeds `polynomial_list`
        if index_end == 0 and index_start > 0:
            index_end = len(polynomial_list) - 1

        # Done extracting the IDs. Extract the polynomial sequence
        vec_aztime_sequence = []
        arr_coeff_sequence = []
        vec_tau0_sequence = []

        # Scale factor to convert range (in meters) to seconds (tau)
        range_to_tau = 2.0 / speed_of_light

        # Take care of the case when the az. time of the polynomial list does not cover
        # the sensing start/stop
        if (index_end == index_start) and handle_out_of_range:
            #      0--1--2--3--4--5 <- az. time of polynomial list (index shown on the left)
            # |--|                   <- sensing start / stop
            if index_start == 0:
                index_end += 1

            # 0--1--2--3--4--5      <- az. time of polynomial list (index shown on the left)
            #                  |--| <- sensing start / stop
            else:
                index_start -= 1

        for poly in polynomial_list[index_start : index_end + 1]:
            vec_aztime_sequence.append(poly[0])
            arr_coeff_sequence.append(poly[1].coeffs)
            vec_tau0_sequence.append(poly[1].mean * range_to_tau)

        return (
            np.array(vec_aztime_sequence),
            np.array(arr_coeff_sequence),
            np.array(vec_tau0_sequence),
        )
