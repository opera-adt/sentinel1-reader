"""Wrapper of s1etad module to read S1-ETAD products."""

import datetime
import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline

try:
    import s1etad
except ImportError:
    raise ImportError("Can NOT import s1etad (https://gitlab.com/s1-etad/s1-etad)!")


def get_eta_correction_from_slc_burst(
    slc_burst,
    eta_dir,
    corr_type="sum",
    include_tropo=True,
    resample=True,
    plot=False,
    verbose=True,
    unit="second",
):
    """Get the ETAD correction for one S1 burst.

    Parameters
    ----------
    slc_burst: Sentinel1BurstSlc
        Sentinel-1 burst object from sentinel1_reader
    eta_dir: str
        Sentinel-1 ETAD product directory
    corr_type: (list of) str
        Sentinel-1 ETAD correction type:
        sar, atm, sum, bistatic, doppler, fmrate, geodetic, ionospheric, tropospheric
        where sar = bistatic + doppler + fmrate
              atm = ionospheric + tropospheric
              sum = sar + atm + geodetic
    resample: bool
        resample the low resolution ETA to the full resample SLC size
    unit: str
        output ETA correction unit:
        pixel, second (meter is not recommended, as time info is more consistent and universal in SAR)

    Returns:
    ----------
    slc_rg_corr: np.ndarray in float32 in size of (lines, samples)
        S1 ETAD correction in range direction in the unit of meter or second
    slc_az_corr: np.ndarray in float32 in size of (lines, samples)
        S1 ETAD correction in azimuth direction in the unit of meter or second
    """
    vprint = print if verbose else lambda *args, **kwargs: None
    if unit not in ["second", "meter", "pixel"]:
        raise ValueError(f"Un-recognized input unit={unit}!")
    meter = unit == "meter"
    # When True, read s1etad product in meters
    # When False, read s1etad product in seconds [recommended]

    # locate / read ETA burst
    eta_burst, eta = get_eta_burst_from_slc_burst(slc_burst, eta_dir, verbose=verbose)

    # read ETA correction data
    corr_type = corr_type.lower()
    if isinstance(corr_type, list):
        corr_types = list(corr_type)
    elif corr_type == "sar":
        corr_types = ["bistatic", "doppler", "fmrate"]
    elif corr_type == "atm":
        corr_types = ["ionospheric", "tropospheric"]
    else:
        corr_types = [corr_type]
    vprint(f"read correction data with type: {corr_type}")

    eta_rg_corr = np.zeros((eta_burst.lines, eta_burst.samples), dtype=np.float32)
    eta_az_corr = np.zeros((eta_burst.lines, eta_burst.samples), dtype=np.float32)
    for corr_type in corr_types:
        correction = eta_burst.get_correction(corr_type, meter=meter)

        if "x" in correction.keys():
            scale = slc_burst.range_sampling_rate if unit == "pixel" else 1.0
            eta_rg_corr += correction["x"] * scale

        if "y" in correction.keys():
            scale = 1.0 / slc_burst.azimuth_time_interval if unit == "pixel" else 1.0
            eta_az_corr += correction["y"] * scale

    if not include_tropo and any(x in corr_types for x in ["tropospheric", "sum"]):
        print("excluding tropospheric component from ETAD products in X direction")
        correction = eta_burst.get_correction("tropospheric")
        scale = slc_burst.range_sampling_rate if unit == "pixel" else 1.0
        eta_rg_corr -= correction["x"] * scale

    if resample or plot:
        # calculate ETA grid
        eta_az_start = (
            eta.min_azimuth_time - slc_burst.sensing_mid
        ).total_seconds() + eta_burst.sampling_start["y"]
        eta_rg_start = eta.min_range_time + eta_burst.sampling_start["x"]
        eta_az_ax = eta_az_start + np.arange(eta_burst.lines) * eta_burst.sampling["y"]
        eta_rg_ax = (
            eta_rg_start + np.arange(eta_burst.samples) * eta_burst.sampling["x"]
        )

        # calculate SLC grid
        slc_az_ax = (
            np.arange(slc_burst.length) * slc_burst.azimuth_time_interval
            + (slc_burst.sensing_start - slc_burst.sensing_mid).total_seconds()
        )
        slc_rg_ax = (
            np.arange(slc_burst.width) / slc_burst.range_sampling_rate
            + slc_burst.slant_range_time
        )

    if resample:
        # resample ETA correction data to the SLC grid
        vprint("resampling the ETA correction data from ETA grid to SLC grid ...")
        rg_interp = RectBivariateSpline(
            eta_az_ax, eta_rg_ax, eta_rg_corr, kx=1, ky=1
        )  # bi-linear
        az_interp = RectBivariateSpline(
            eta_az_ax, eta_rg_ax, eta_az_corr, kx=1, ky=1
        )  # bi-linear
        slc_rg_corr = rg_interp(slc_az_ax, slc_rg_ax)
        slc_az_corr = az_interp(slc_az_ax, slc_rg_ax)

    if plot:
        vprint("plot ETA correction data and grid")

        # figure 1 - ETA corrections
        fig, axs = plt.subplots(
            nrows=2, ncols=1, figsize=[12, 6], sharex=True, sharey=True
        )
        for ax, corr, title in zip(axs, [eta_rg_corr, eta_az_corr], ["x", "y"]):
            im = ax.imshow(corr, aspect="auto", interpolation="nearest")
            fig.colorbar(im, ax=ax, shrink=0.8, location="right").set_label("pixel")
            ax.set_title(f"correction [{title}]")
            ax.set_ylabel("Azimuth [pixel]")
        axs[1].set_xlabel("Range [pixel]")
        fig.tight_layout()

        # figure 2 - ETA & SLC grids [for comparison/checking]
        eta_box = np.asarray(
            [
                (eta_rg_ax[0], eta_az_ax[0]),
                (eta_rg_ax[0], eta_az_ax[-1]),
                (eta_rg_ax[-1], eta_az_ax[-1]),
                (eta_rg_ax[-1], eta_az_ax[0]),
                (eta_rg_ax[0], eta_az_ax[0]),
            ]
        )
        slc_box = np.asarray(
            [
                (slc_rg_ax[0], slc_az_ax[0]),
                (slc_rg_ax[0], slc_az_ax[-1]),
                (slc_rg_ax[-1], slc_az_ax[-1]),
                (slc_rg_ax[-1], slc_az_ax[0]),
                (slc_rg_ax[0], slc_az_ax[0]),
            ]
        )

        fig, ax = plt.subplots(figsize=[12, 3])
        ax.plot(eta_box[:, 0] * 1e3, eta_box[:, 1], "C0", label="ETA")
        ax.plot(slc_box[:, 0] * 1e3, slc_box[:, 1], "C1", label="SLC")
        ax.set_xlabel("Range [ms]")
        ax.set_ylabel("Azimuth [s]")
        ax.set_title("grid")
        ax.legend()
        ax.grid()

        plt.show()

    if not resample:
        return eta_rg_corr, eta_az_corr

    return slc_rg_corr, slc_az_corr


def get_eta_burst_from_slc_burst(slc_burst, eta_dir, verbose=True):
    """Read ETA burst corresponding to the input SLC burst."""

    # locate ETAD file
    eta_file = get_eta_file_from_slc_burst(slc_burst, eta_dir, verbose=verbose)

    # read ETA file using s1-etad
    eta = s1etad.Sentinel1Etad(eta_file)

    # locate the ETA burst
    t0_query = slc_burst.sensing_start - datetime.timedelta(seconds=0.25)
    t1_query = slc_burst.sensing_stop + datetime.timedelta(seconds=0.25)
    if verbose:
        print(
            f"search ETA burst in {slc_burst.swath_name} with the following time range:"
        )
        print(f"start time: {t0_query}")
        print(f"end   time: {t1_query}")

    selection = eta.query_burst(
        swath=slc_burst.swath_name.upper(),
        first_time=t0_query,
        last_time=t1_query,
    )

    if len(selection) == 0:
        raise ValueError("No ETA burst found!")
    elif len(selection) > 1:
        raise ValueError(
            "More than 1 ETA burst found, please adjust your search/query criteria!"
        )

    eta_burst = eta[slc_burst.swath_name.upper()][selection.bIndex.values[0]]

    return eta_burst, eta


def get_eta_file_from_slc_burst(slc_burst, eta_dir, verbose=True):
    """Get/locate ETAD file path based on SLC burst."""

    # safe filename --> ETA filename pattern
    fparts = os.path.basename(slc_burst.safe_filename).split("_")
    eta_fbase = (
        f"{fparts[0]}_IW_ETA__*_{fparts[5]}_{fparts[6]}_{fparts[7]}_{fparts[8]}_*.SAFE"
    )

    # search the ETA filename pattern
    eta_file = glob.glob(os.path.join(eta_dir, eta_fbase))[0]
    if verbose:
        print(f"search ETA file with pattern: {eta_fbase}")
        print(f"locate ETA file: {eta_file}")

    return eta_file
