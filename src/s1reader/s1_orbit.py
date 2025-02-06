from __future__ import annotations

import datetime
import glob
import os
import warnings
import logging
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from functools import cache
from typing import Literal

logger = logging.getLogger(__name__)

# Date format used in file names
FMT = "%Y%m%dT%H%M%S"

# Orbital period of Sentinel-1 in seconds:
# 12 days * 86400.0 seconds/day, divided into 175 orbits
T_ORBIT = (12 * 86400.0) / 175.0
PADDING_SHORT = 60

# Temporal margin to apply to the start time of a frame to make sure that the
# ascending node crossing is included when choosing the orbit file
padding_short = datetime.timedelta(seconds=PADDING_SHORT)
margin_start_time = datetime.timedelta(seconds=T_ORBIT + PADDING_SHORT)

# ASF S3 Bucket for orbit files
ASF_BUCKET_NAME = "s1-orbits"


@cache
def list_public_bucket(bucket_name: str, prefix: str = "") -> list[str]:
    """List all objects in a public S3 bucket.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket.
    prefix : str, optional
        Prefix to filter objects, by default "".

    Returns
    -------
    list[str]
        List of object keys in the bucket.
    """
    endpoint = f"https://{bucket_name}.s3.amazonaws.com"
    marker: str | None = None
    keys: list[str] = []

    while True:
        params = {"prefix": prefix}
        if marker:
            params["marker"] = marker

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Error fetching bucket contents: {e}")
            raise

        root = ET.fromstring(response.content)
        for contents in root.findall(
            "{http://s3.amazonaws.com/doc/2006-03-01/}Contents"
        ):
            key = contents.find("{http://s3.amazonaws.com/doc/2006-03-01/}Key")
            if key is not None:
                keys.append(key.text or "")
                logger.debug(f"Found key: {key.text}")

        is_truncated = root.find("{http://s3.amazonaws.com/doc/2006-03-01/}IsTruncated")
        if (
            is_truncated is not None
            and is_truncated.text
            and is_truncated.text.lower() == "true"
        ):
            next_marker = root.find(
                "{http://s3.amazonaws.com/doc/2006-03-01/}NextMarker"
            )
            if next_marker is not None:
                marker = next_marker.text
            else:
                found_keys = root.findall(
                    "{http://s3.amazonaws.com/doc/2006-03-01/}Contents/{http://s3.amazonaws.com/doc/2006-03-01/}Key"
                )
                if found_keys:
                    marker = found_keys[-1].text
                else:
                    break
        else:
            break

    return keys


def get_orbit_files(orbit_type: Literal["precise", "restituted"]) -> list[str]:
    """Get a list of precise or restituted orbit files from the ASF S3 service.

    Parameters
    ----------
    orbit_type : Literal["precise", "restituted"]
        Type of orbit files to retrieve.

    Returns
    -------
    list[str]
        List of orbit file keys.

    Raises
    ------
    ValueError
        If an invalid orbit_type is provided.
    """
    prefix = (
        "AUX_POEORB"
        if orbit_type == "precise"
        else "AUX_RESORB"
        if orbit_type == "restituted"
        else None
    )
    if prefix is None:
        raise ValueError("orbit_type must be either 'precise' or 'restituted'")

    all_keys = list_public_bucket(ASF_BUCKET_NAME)
    orbit_files = [key for key in all_keys if key.startswith(prefix)]

    logger.info(f"Found {len(orbit_files)} {orbit_type} orbit files")
    return orbit_files


def download_orbit_file_from_s3(key: str, orbit_dir: str) -> str:
    """
    Download an orbit file from the ASF S3 bucket.

    Parameters
    ----------
    key : str
        The object key in the S3 bucket.
    orbit_dir : str
        Local directory where the file will be saved.

    Returns
    -------
    str
        Path to the downloaded file.
    """
    local_file = os.path.join(orbit_dir, os.path.basename(key))
    url = f"https://{ASF_BUCKET_NAME}.s3.amazonaws.com/{key}"
    logger.info(f"Downloading orbit file from {url}")
    response = requests.get(url)
    response.raise_for_status()

    with open(local_file, "wb") as f:
        f.write(response.content)
    return local_file


def retrieve_orbit_file(
    safe_file: str,
    orbit_dir: str,
    concatenate: bool = False,
    orbit_type_preference: Literal["precise", "restituted"] = "precise",
) -> str | list | None:
    """
    Retrieve the orbit file for a given SAFE file using the ASF S3 service.

    This high-level function replaces the old Scihub-based API.

    Parameters
    ----------
    safe_file : str
        File path to the SAFE file.
    orbit_dir : str
        Directory to store downloaded orbit files.
    concatenate : bool, optional
        If True, concatenate a pair of restituted orbit files when no single file covers the time window.
    orbit_type_preference : Literal["precise", "restituted"], optional
        Preferred orbit file type (default is "precise").

    Returns
    -------
    orbit_file : str | list | None
        A local orbit file path, a list of files, or None if no orbit file is found.
    """
    os.makedirs(orbit_dir, exist_ok=True)

    mission_id, _, sensing_start_time, sensing_end_time, _ = _parse_safe_filename(
        safe_file
    )
    search_start_time = sensing_start_time - margin_start_time
    search_end_time = sensing_end_time + padding_short
    t_search_window = [search_start_time, search_end_time]

    def candidate_covers_timeframe(key: str, window: list[datetime.datetime]) -> bool:
        """
        Check if the orbit file key covers the given time window.
        Expected key format: ..._V{start}_{end}.EOF
        """
        base = os.path.basename(key)
        try:
            parts = base.split("_")
            start_str = parts[-2]
            end_str = parts[-1]
            if start_str.startswith("V"):
                start_str = start_str[1:]
            if end_str.endswith(".EOF"):
                end_str = end_str[:-4]
            t_start = datetime.datetime.strptime(start_str, FMT)
            t_end = datetime.datetime.strptime(end_str, FMT)
            return t_start < window[0] and window[1] < t_end
        except Exception as e:
            logger.error(f"Error parsing orbit file key {key}: {e}")
            return False

    # Try the preferred orbit type first.
    candidates = get_orbit_files(
        "precise" if orbit_type_preference == "precise" else "restituted"
    )
    filtered = [
        key
        for key in candidates
        if mission_id in key and candidate_covers_timeframe(key, t_search_window)
    ]
    if filtered:
        chosen_key = sorted(filtered)[0]
        local_file = os.path.join(orbit_dir, os.path.basename(chosen_key))
        if not os.path.exists(local_file):
            local_file = download_orbit_file_from_s3(chosen_key, orbit_dir)
        return local_file

    # If not found and the preferred type is "precise", try the restituted orbit files.
    if orbit_type_preference == "precise":
        candidates = get_orbit_files("restituted")
        filtered = [
            key
            for key in candidates
            if mission_id in key and candidate_covers_timeframe(key, t_search_window)
        ]
        if filtered:
            chosen_key = sorted(filtered)[0]
            local_file = os.path.join(orbit_dir, os.path.basename(chosen_key))
            if not os.path.exists(local_file):
                local_file = download_orbit_file_from_s3(chosen_key, orbit_dir)
            return local_file

    logger.info("Attempting to retrieve a pair of restituted orbit files.")
    candidates = get_orbit_files("restituted")
    window_earlier = [search_start_time, search_start_time + 2 * padding_short]
    window_later = [
        sensing_start_time - padding_short,
        sensing_end_time + padding_short,
    ]
    candidate_earlier = None
    candidate_later = None

    for key in sorted(candidates):
        if mission_id not in key:
            continue
        if candidate_earlier is None and candidate_covers_timeframe(
            key, window_earlier
        ):
            logger.info(
                "Found restituted orbit file covering ANX before sensing start."
            )
            candidate_earlier = key
        elif candidate_later is None and candidate_covers_timeframe(key, window_later):
            logger.info("Found restituted orbit file covering the S1 SAFE frame.")
            candidate_later = key
        if candidate_earlier and candidate_later:
            break

    if not (candidate_earlier and candidate_later):
        warnings.warn(
            "Cannot find a pair of restituted orbit files covering the time window"
        )
        return None

    local_file_earlier = download_orbit_file_from_s3(candidate_earlier, orbit_dir)
    local_file_later = download_orbit_file_from_s3(candidate_later, orbit_dir)

    if concatenate:
        concat_file = combine_xml_orbit_elements(local_file_later, local_file_earlier)
        logger.info("RESORB concatenation successful.")
        return concat_file
    else:
        return [local_file_earlier, local_file_later]


def _parse_safe_filename(safe_filename):
    """
    Extract info from S1-A/B SAFE filename.

    SAFE filename structure:
      S1A_IW_SLC__1SDV_20150224T114043_20150224T114111_004764_005E86_AD02.SAFE

    Returns
    -------
    [mission_id, sensor_mode, start_datetime, end_datetime, abs_orbit_num]
    """
    safe_name = os.path.basename(safe_filename)
    mission_id = safe_name[:3]
    sensor_mode = safe_name[4:6]
    start_datetime = datetime.datetime.strptime(safe_name[17:32], FMT)
    end_datetime = datetime.datetime.strptime(safe_name[33:48], FMT)
    abs_orb_num = int(safe_name[49:55])
    return [mission_id, sensor_mode, start_datetime, end_datetime, abs_orb_num]


def _get_file_name_tokens(zip_path: str) -> tuple[str, list[datetime.datetime]]:
    """
    Extract swath platform ID and start/stop times from a SAFE file path.

    Returns
    -------
    mission_id: str
    t_swath_start_stop: list[datetime.datetime]
        Swath start/stop times.
    """
    mission_id, _, start_time, end_time, _ = _parse_safe_filename(zip_path)
    return mission_id, [start_time, end_time]


def get_orbit_file_from_dir(
    zip_path: str,
    orbit_dir: str,
    auto_download: bool = False,
    concat_resorb: bool = False,
) -> str | list | None:
    """
    Get the orbit state vector list for a given swath from a directory.

    If the orbit file is not found locally and auto_download is True,
    the file is retrieved from ASF S3.

    Parameters
    ----------
    zip_path : str
        Path to Sentinel-1 SAFE zip file.
    orbit_dir : str
        Directory containing orbit files.
    auto_download : bool
        Automatically download the orbit file if not found.
    concat_resorb : bool
        Concatenate two RESORB files if needed.

    Returns
    -------
    str or list or None
        Path to the orbit file, a list of orbit files, or None.
    """

    if not os.path.isdir(orbit_dir):
        if not auto_download:
            raise NotADirectoryError(f"{orbit_dir} not found")
        else:
            logger.info(f"{orbit_dir} not found, creating directory.")
            os.makedirs(orbit_dir, exist_ok=True)

    orbit_file_list = glob.glob(os.path.join(orbit_dir, "S1*.EOF"))
    orbit_file = get_orbit_file_from_list(zip_path, orbit_file_list, concat_resorb)

    if orbit_file:
        return orbit_file

    if not auto_download:
        warnings.warn(
            f"No orbit file was found for {os.path.basename(zip_path)} "
            f"from the directory {orbit_dir}"
        )
        return None

    orbit_file = retrieve_orbit_file(zip_path, orbit_dir, concat_resorb)
    return orbit_file


def get_orbit_file_from_list(
    zip_path: str, orbit_file_list: list, concat_resorb: bool = False
) -> str | list | None:
    """
    Get the orbit file for a given S-1 swath from a list of files.
    """

    mission_id, t_swath_start_stop = _get_file_name_tokens(zip_path)
    t_search_window = [t_swath_start_stop[0] - margin_start_time, t_swath_start_stop[1]]
    orbit_file_final: str | None = None

    for orbit_file in sorted(orbit_file_list, key=lambda x: os.path.basename(x)):
        if not os.path.isfile(orbit_file):
            continue
        if mission_id not in os.path.basename(orbit_file):
            continue

        t_orbit_start, t_orbit_end = os.path.basename(orbit_file).split("_")[-2:]
        t_orbit_start = datetime.datetime.strptime(t_orbit_start[1:], FMT)
        t_orbit_stop = datetime.datetime.strptime(t_orbit_end[:-4], FMT)

        if all([t_orbit_start < t < t_orbit_stop for t in t_search_window]):
            orbit_file_final = orbit_file
            break

    if not orbit_file_final:
        warnings.warn(
            "No single orbit file was found in the file list provided. "
            "Attempting to find a set of RESORB files that covers the time period."
        )
        orbit_file_final = get_resorb_pair_from_list(
            zip_path, sorted(orbit_file_list), concat_resorb
        )

    if not orbit_file_final:
        warnings.warn("No single orbit file was found in the file list provided.")

    return orbit_file_final


def _covers_timeframe(orbit_file: str, t_start_stop_frame: list) -> bool:
    """
    Check if the orbit file covers the specified time frame.

    Parameters
    ----------
    orbit_file : str
        Orbit file.
    t_start_stop_frame : list[datetime.datetime]
        Start/stop time frame.

    Returns
    -------
    bool
        True if the orbit file covers the time frame, False otherwise.
    """
    start_str, stop_str = os.path.basename(orbit_file).split("_")[-2:]
    t_orbit_start = datetime.datetime.strptime(start_str[1:], FMT)
    t_orbit_stop = datetime.datetime.strptime(stop_str[:-4], FMT)
    return all(t_orbit_start < t < t_orbit_stop for t in t_start_stop_frame)


def get_resorb_pair_from_list(
    zip_path: str, orbit_file_list: list, concatenate_resorb: bool = False
) -> list | str | None:
    """
    Find two RESORB files that cover the required time frame.

    Used if POEORB is not found, or there is no RESORB file that
    covers the sensing period + margin at the starting time.
    Try to find two subsequent RESORB files that covers the
    sensing period + margins at the starting time.

    NOTE about timing design of RESORB files based on the investigation in 09/2023
    Duration (H:M:S):                                     3:17:30
    Overlap between the subsequent RESORB  (H:M:S):       1:38:46
    Time shifting between the subsequent RESORB  (H:M:S): 1:38:44
    T_orb  (H:M:S):                                       1:38:44.57
    """

    mission_id, t_swath_start_stop = _get_file_name_tokens(zip_path)
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
            logger.info("Found RESORB file covering the S1 SAFE frame.")
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
            logger.info("Found RESORB file covering ANX before sensing start.")
            resorb_filename_earlier = resorb_file
            continue

        if resorb_filename_earlier is not None and resorb_filename_later is not None:
            break

    # if 1. and 2. are successful return the result as a list of orbit file, or
    # as a concatenated RESORB file
    # Either concatenate the orbit, or return the RESORB pair
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
    if resorb_filename_earlier is not None and resorb_filename_later is not None:
        if concatenate_resorb:
            # BE CAREFUL ABOUT THE ORDER HOW THEY ARE CONCATENATED.
            # See NOTE in retrieve_orbit_file() for detail.
            concat_resorb_filename = combine_xml_orbit_elements(
                resorb_filename_later, resorb_filename_earlier
            )
            logger.info("RESORB concatenation successful.")
            return concat_resorb_filename
        else:
            return [resorb_filename_earlier, resorb_filename_later]

    logger.info("Cannot find a pair of RESORB files that meet the time frame criteria.")
    return None


def combine_xml_orbit_elements(file1: str, file2: str) -> str:
    """
    Combine the orbit elements from two XML orbit files.

    `file1` is the "base" of the output: All of the orbit state vectors from `file1` are used,
    and only the OSVs from `file2` which do fall within the time range of `file1` are included.

    Create a new .EOF file with the combined results.
    Output is named with the start_datetime and stop_datetime changed..

    Returns
    -------
    str
        Name of the newly created concatenated orbit file.
    """

    def get_dt(root: ET.Element, tag_name: str) -> datetime.datetime:
        time_str = root.find(f".//{tag_name}").text.split("=")[-1]
        return datetime.datetime.fromisoformat(time_str)

    tree1 = ET.parse(file1)
    tree2 = ET.parse(file2)
    root1 = tree1.getroot()
    root2 = tree2.getroot()

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

    list_of_osvs1 = root1.find(".//List_of_OSVs")
    list_of_osvs2 = root2.find(".//List_of_OSVs")
    merge_osv_list(list_of_osvs1, list_of_osvs2)

    outfile = _generate_filename(file1, new_start_dt, new_stop_dt)
    tree1.write(outfile, encoding="UTF-8", xml_declaration=True)
    return outfile


def merge_osv_list(list_of_osvs1, list_of_osvs2):
    """
    Merge two orbit state vector lists and sort them in chronological order.

    Apply sorting to make sure the OSVs are in chronological order
    `list_of_osvs1` will be the "base OSV list" while the OSVs in
    `list_of_osvs2` not in the time range of the base OSV list will be
    appended.
    """
    osv1_utc_list = [_get_utc_time_from_osv(osv1) for osv1 in list_of_osvs1]
    min_utc_osv1 = min(osv1_utc_list)
    max_utc_osv1 = max(osv1_utc_list)

    for osv in list_of_osvs2.findall("OSV"):
        utc_osv2 = _get_utc_time_from_osv(osv)
        if min_utc_osv1 < utc_osv2 < max_utc_osv1:
            continue
        list_of_osvs1.append(osv)

    list_of_osvs1 = _sort_list_of_osv(list_of_osvs1)
    # Adjust the count attribute in <List_of_OSVs>
    new_count = len(list_of_osvs1.findall("OSV"))
    list_of_osvs1.set("count", str(new_count))
    return list_of_osvs1


def _generate_filename(
    file_base: str, new_start: datetime.datetime, new_stop: datetime.datetime
) -> str:
    """
    Generate a new filename for the concatenated orbit file.

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
    fmt = "%Y%m%dT%H%M%S"
    new_start_stop_str = new_start.strftime(fmt) + "_" + new_stop.strftime(fmt)
    new_product_name = product_name[: index_prefix + 2] + new_start_stop_str
    return str(file_base).replace(product_name, new_product_name) + ".EOF"


def _sort_list_of_osv(list_of_osvs):
    """
    Sort the orbit state vector elements by UTC timestamp.

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
    Extract the UTC time from an orbit state vector element.

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
    return datetime.datetime.fromisoformat(utc_osv_string)
