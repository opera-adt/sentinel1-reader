import argparse
from pathlib import Path

try:
    import fiona
    import geopandas as gpd
    import pandas as pd
except ImportError:
    print("ERROR: fiona, geopandas, and pandas are required for this script.")
    raise

from osgeo import osr
from shapely.geometry import Polygon
from shapely import wkt

from s1reader.s1_orbit import get_orbit_file_from_dir
from s1reader.s1_reader import load_bursts


def command_line_parser():
    """
    Command line parser
    """
    parser = argparse.ArgumentParser(
        description="Create a burst map for a single slc",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-s", "--slc", help="Sentinel-1 product to load.")
    parser.add_argument("-d", "--orbit-dir", help="Directory containing orbit files")
    parser.add_argument(
        "-x",
        "--x-spacing",
        type=float,
        default=5,
        help="Spacing of the geogrid in x direction",
    )
    parser.add_argument(
        "-y",
        "--y-spacing",
        type=float,
        default=10,
        help="Spacing of the geogrid in y direction",
    )
    parser.add_argument("-e", "--epsg", type=int, help="EPSG for output coordinates")
    parser.add_argument(
        "-o",
        "--output",
        default="burst_map.gpkg",
        dest="output",
        help="Base filename for all output burst map products",
    )
    return parser.parse_args()


def burst_map(slc, orbit_dir, x_spacing, y_spacing, epsg, output_filename):
    """Create a CSV of SLC metadata and plot bursts.

    Parameters
    ----------
    slc: str
      Path to SLC file
    orbit_dir: str
      Path to directory containing orbit files
    x_spacing: float
      Spacing of the geogrid in the x direction
    y_spacing: float
      Spacing to the geogrid in the y direction
    epsg: int
      EPSG code for the output coodrdinates
    output_filename: str
      Filename used for the output CSV, shp, html, and kml files

    Returns
    -------
    output_filename.csv, output_filename.shp, output_filename.html, output_filename.kml
    """
    # Initialize dictionary that will contain all the info for geocoding
    burst_map = {
        "burst_id": [],
        "length": [],
        "width": [],
        "spacing_x": [],
        "spacing_y": [],
        "min_x": [],
        "max_x": [],
        "min_y": [],
        "max_y": [],
        "first_valid_line": [],
        "last_valid_line": [],
        "first_valid_sample": [],
        "last_valid_sample": [],
        "border": [],
    }
    i_subswath = [1, 2, 3]
    pol = "vv"
    orbit_path = get_orbit_file_from_dir(slc, orbit_dir) if orbit_dir else None

    for subswath in i_subswath:
        ref_bursts = load_bursts(slc, orbit_path, subswath, pol)
        for burst in ref_bursts:
            burst_map["burst_id"].append(burst.burst_id)
            burst_map["length"].append(burst.shape[0])
            burst_map["width"].append(burst.shape[1])
            burst_map["spacing_x"].append(x_spacing)
            burst_map["spacing_y"].append(y_spacing)
            burst_map["first_valid_line"].append(burst.first_valid_line)
            burst_map["last_valid_line"].append(burst.last_valid_line)
            burst_map["first_valid_sample"].append(burst.first_valid_sample)
            burst_map["last_valid_sample"].append(burst.last_valid_sample)

            # TODO: this will ignore the other border for bursts on the antimeridian.
            # Should probably turn into a MultiPolygon
            poly = burst.border[0]
            # Give some margin to the polygon
            margin = 0.001
            poly = poly.buffer(margin)
            burst_map["border"].append(Polygon(poly.exterior.coords).wkt)

            # Transform coordinates from lat/long to EPSG
            llh = osr.SpatialReference()
            llh.ImportFromEPSG(4326)
            tgt = osr.SpatialReference()

            tgt_x, tgt_y = [], []
            x, y = poly.exterior.coords.xy
            tgt.ImportFromEPSG(int(epsg))
            trans = osr.CoordinateTransformation(llh, tgt)
            for lx, ly in zip(x, y):
                dummy_y, dummy_x, dummy_z = trans.TransformPoint(ly, lx, 0)
                tgt_x.append(dummy_x)
                tgt_y.append(dummy_y)

            # TODO: Get the min/max from the burst database
            if epsg == 4326:
                x_min = x_spacing * (min(tgt_x) / x_spacing)
                y_min = y_spacing * (min(tgt_y) / y_spacing)
                x_max = x_spacing * (max(tgt_x) / x_spacing)
                y_max = y_spacing * (max(tgt_y) / y_spacing)
            else:
                x_min = x_spacing * round(min(tgt_x) / x_spacing)
                y_min = y_spacing * round(min(tgt_y) / y_spacing)
                x_max = x_spacing * round(max(tgt_x) / x_spacing)
                y_max = y_spacing * round(max(tgt_y) / y_spacing)

            # Allocate coordinates inside the dictionary
            burst_map["min_x"].append(x_min)
            burst_map["min_y"].append(y_min)
            burst_map["max_x"].append(x_max)
            burst_map["max_y"].append(y_max)

    out_path = Path(output_filename)

    # Save generated burst map as csv
    data = pd.DataFrame.from_dict(burst_map)
    data.to_csv(out_path.with_suffix(".csv"))

    # Create GeoDataFrame to plot bursts on a map
    df = data
    df["border"] = df["border"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df.rename(columns={"border": "geometry"}), crs="epsg:4326")

    gdf.to_file(out_path.with_suffix(".gpkg"), driver="GPKG")
    # Save the GeoDataFrame as a shapefile (some people may prefer the format)
    gdf.to_file(out_path.with_suffix(".shp"))

    # Save the GeoDataFrame as a kml
    kml_path = out_path.with_suffix(".kml")
    if kml_path.exists():
        kml_path.unlink()

    fiona.supported_drivers["KML"] = "rw"
    gdf.to_file(kml_path, driver="KML")

    # Plot bursts on an interactive map
    m = gdf.explore(
        column="burst_id",  # make choropleth based on "Burst ID" column
        tooltip="burst_id",  # show "Burst ID" value in tooltip (on hover)
        popup=True,  # show all values in popup (on click)
        tiles="CartoDB positron",  # use "CartoDB positron" tiles
        cmap="Set1",  # use "Set1" matplotlib colormap
        style_kwds=dict(color="black"),  # use black outline
    )

    m.save(out_path.with_suffix(".html"))


if __name__ == "__main__":
    cmd = command_line_parser()
    burst_map(
        cmd.slc, cmd.orbit_dir, cmd.x_spacing, cmd.y_spacing, cmd.epsg, cmd.output
    )
