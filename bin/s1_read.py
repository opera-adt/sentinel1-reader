import os
import sys

import s1reader

if __name__ == "__main__":
    """testing script that prints burst info and write SLC to file"""
    # TODO replace with argparse
    path = sys.argv[1]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")

    i_subswath = int(sys.argv[2])
    if i_subswath < 1 or i_subswath > 3:
        raise ValueError("i_subswath not <1 or >3")

    pol = sys.argv[3]
    pols = ["vv", "vh", "hh", "hv"]
    if pol not in pols:
        raise ValueError("polarization not in {pols}")

    orbit_dir = sys.argv[4]
    if not os.path.isdir(orbit_dir):
        raise NotADirectoryError(f"{orbit_dir} not found")
    orbit_path = s1reader.get_orbit_file_from_dir(path, orbit_dir)

    bursts = s1reader.load_bursts(path, orbit_path, i_subswath, pol)

    # print out IDs and lat/lon centers of all bursts
    for i, burst in enumerate(bursts):
        print(burst.burst_id, burst.center)

    # write to ENVI (default)
    burst.slc_to_file("burst.slc")
    # write to geotiff
    burst.slc_to_file("burst.tif", "GTiff")
    # write to VRT
    burst.slc_to_file("burst.vrt", "VRT")
