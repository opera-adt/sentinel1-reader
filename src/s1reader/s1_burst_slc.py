import os
from dataclasses import dataclass
import datetime
import tempfile
from typing import Optional
import warnings

from packaging import version
from types import SimpleNamespace

import isce3
import numpy as np
from osgeo import gdal
from scipy.interpolate import InterpolatedUnivariateSpline
from s1reader import s1_annotation
from .s1_burst_id import S1BurstId


# Other functionalities
def polyfit(
    xin,
    yin,
    zin,
    azimuth_order,
    range_order,
    sig=None,
    snr=None,
    cond=1.0e-12,
    max_order=True,
):
    """
    Fit 2-D polynomial
    Parameters:
    xin: np.ndarray
       Array locations along x direction
    yin: np.ndarray
       Array locations along y direction
    zin: np.ndarray
       Array locations along z direction
    azimuth_order: int
       Azimuth polynomial order
    range_order: int
       Slant range polynomial order
    sig: -
       ---------------------------
    snr: float
       Signal to noise ratio
    cond: float
       ---------------------------
    max_order: bool
       ---------------------------

    Returns:
    poly: isce3.core.Poly2D
       class represents a polynomial function of range
       'x' and azimuth 'y'
    """
    x = np.array(xin)
    xmin = np.min(x)
    xnorm = np.max(x) - xmin
    if xnorm == 0:
        xnorm = 1.0
    x = (x - xmin) / xnorm

    y = np.array(yin)
    ymin = np.min(y)
    ynorm = np.max(y) - ymin
    if ynorm == 0:
        ynorm = 1.0
    y = (y - ymin) / ynorm

    z = np.array(zin)
    big_order = max(azimuth_order, range_order)

    arr_list = []
    for ii in range(azimuth_order + 1):
        yfact = np.power(y, ii)
        for jj in range(range_order + 1):
            xfact = np.power(x, jj) * yfact
            if max_order:
                if (ii + jj) <= big_order:
                    arr_list.append(xfact.reshape((x.size, 1)))
            else:
                arr_list.append(xfact.reshape((x.size, 1)))

    A = np.hstack(arr_list)
    if sig is not None and snr is not None:
        raise Exception("Only one of sig / snr can be provided")
    if sig is not None:
        snr = 1.0 + 1.0 / sig
    if snr is not None:
        A = A / snr[:, None]
        z = z / snr

    val, res, _, _ = np.linalg.lstsq(A, z, rcond=cond)
    if len(res) > 0:
        print("Chi squared: %f" % (np.sqrt(res / (1.0 * len(z)))))
    else:
        print("No chi squared value....")
        print("Try reducing rank of polynomial.")

    coeffs = []
    count = 0
    for ii in range(azimuth_order + 1):
        row = []
        for jj in range(range_order + 1):
            if max_order:
                if (ii + jj) <= big_order:
                    row.append(val[count])
                    count = count + 1
                else:
                    row.append(0.0)
            else:
                row.append(val[count])
                count = count + 1
        coeffs.append(row)
    poly = isce3.core.Poly2d(coeffs, xmin, ymin, xnorm, ynorm)
    return poly


def _evaluate_polynomial_array(arr_polynomial, grid_tau, vec_tau0):
    """
    Evaluate the polynomials on the correction grid.
    To be used for calculating azimuth FM mismatch rate.

    Parameters:
    -----------
    arr_polynomial: np.ndarray
        coefficients interpolated to azimuth times in each line
        in correction grid
    grid_tau: np.ndarray
        2d numpy array filled with range time in the correction grid
    vec_tau0: np.ndarray
        range start time for each line in the correction grid

    Return:
    -------
    eval_out: np.ndarray
        Evaluated values on the correction grid
    """

    ncol = grid_tau.shape[1]
    term_tau = grid_tau - vec_tau0 * np.ones(ncol)[np.newaxis, ...]

    eval_out = (
        arr_polynomial[:, 0][..., np.newaxis] * np.ones(ncol)[np.newaxis, ...]
        + (arr_polynomial[:, 1][..., np.newaxis] * np.ones(ncol)[np.newaxis, ...])
        * term_tau
        + (arr_polynomial[:, 2][..., np.newaxis] * np.ones(ncol)[np.newaxis, ...])
        * term_tau**2
    )

    return eval_out


def _llh_to_ecef(lat, lon, hgt, ellipsoid, in_degree=True):
    """
    Calculate cartesian coordinates in ECEF from
    latitude, longitude, and altitude

    Parameters:
    -----------
    lat: np.ndarray
        latitude as numpy array
    lon: np.ndarray
        longitude as numpy array
    hgt: np.ndarray
        height as numpy array
    ellipsoid: isce3.core.Ellipsoid
        Definition of Ellipsoid
    in_degree: bool
        True if the units of lat and lon are degrees.
        False if the units are radian.

    Return:
    x_ecef : ndarray
        ECEF X coordinate (in meters).
    y_ecef : ndarray
        ECEF X coordinate (in meters).
    z_ecef : ndarray
        ECEF X coordinate (in meters).
        x, y, and z as a tuple of np.ndarray

    """

    if in_degree:
        rad_lat = np.radians(lat)
        rad_lon = np.radians(lon)
    else:
        rad_lat = lat
        rad_lon = lon

    v_ellipsoid = ellipsoid.a / np.sqrt(
        1 - ellipsoid.e2 * np.sin(rad_lat) * np.sin(rad_lat)
    )

    x_ecef = (v_ellipsoid + hgt) * np.cos(rad_lat) * np.cos(rad_lon)
    y_ecef = (v_ellipsoid + hgt) * np.cos(rad_lat) * np.sin(rad_lon)
    z_ecef = (v_ellipsoid * (1 - ellipsoid.e2) + hgt) * np.sin(rad_lat)

    return (x_ecef, y_ecef, z_ecef)


@dataclass
class AzimuthCarrierComponents:
    kt: np.ndarray
    eta: float
    eta_ref: float

    @property
    def antenna_steering_doppler(self):
        return self.kt * (self.eta - self.eta_ref)

    @property
    def carrier(self):
        return np.pi * self.kt * ((self.eta - self.eta_ref) ** 2)


@dataclass(frozen=True)
class Doppler:
    poly1d: isce3.core.Poly1d
    lut2d: isce3.core.LUT2d


@dataclass(frozen=True)
class Sentinel1BurstSlc:
    """Raw values extracted from SAFE XML."""

    # ipf_version:float
    ipf_version: version.Version
    sensing_start: datetime.datetime
    radar_center_frequency: float
    wavelength: float
    azimuth_steer_rate: float
    average_azimuth_pixel_spacing: float
    azimuth_time_interval: float
    slant_range_time: float
    starting_range: float
    iw2_mid_range: float
    range_sampling_rate: float
    range_pixel_spacing: float
    shape: tuple()
    azimuth_fm_rate: isce3.core.Poly1d
    doppler: Doppler
    range_bandwidth: float
    polarization: str  # {VV, VH, HH, HV}
    burst_id: S1BurstId
    platform_id: str  # S1{A,B}
    safe_filename: str  # SAFE file name
    center: tuple  # {center lon, center lat} in degrees
    border: list  # list of lon, lat coordinate tuples (in degrees) representing burst border
    orbit: isce3.core.Orbit
    orbit_direction: str
    abs_orbit_number: int  # Absolute orbit number
    # VRT params
    tiff_path: str  # path to measurement tiff in SAFE/zip
    i_burst: int
    first_valid_sample: int
    last_valid_sample: int
    first_valid_line: int
    last_valid_line: int
    # window parameters
    range_window_type: str
    range_window_coefficient: float
    rank: int  # The number of PRI between transmitted pulse and return echo.
    prf_raw_data: float  # Pulse repetition frequency (PRF) of the raw data [Hz]
    range_chirp_rate: float  # Range chirp rate [Hz]

    # Correction information
    burst_calibration: s1_annotation.BurstCalibration  # Radiometric correction
    burst_noise: s1_annotation.BurstNoise  # Thermal noise correction
    burst_eap: s1_annotation.BurstEAP  # EAP correction

    # Time series of FM rate / Doppler centroid polynomial coefficients
    # for azimuth FM rate mismatch mitigation
    extended_coeffs: s1_annotation.BurstExtendedCoeffs

    burst_rfi_info: SimpleNamespace

    burst_misc_metadata: SimpleNamespace

    def __str__(self):
        return f"Sentinel1BurstSlc: {self.burst_id} at {self.sensing_start}"

    def __repr__(self):
        return f"{self.__class__.__name__}(burst_id={self.burst_id})"

    def as_isce3_radargrid(
        self, az_step: Optional[float] = None, rg_step: Optional[float] = None
    ):
        """Init and return isce3.product.RadarGridParameters.

        The `az_step` and `rg_step` parameters are used to construct a
        decimated grid. If not specified, the grid will be at the full radar
        resolution.
        Note that increasing the range/azimuth step size does not change the sensing
        start of the grid, as the grid is decimated rather than multilooked.

        Parameters
        ----------
        az_step : float, optional
            Azimuth step size in seconds. If not provided, the azimuth step
            size is set to the azimuth time interval.
        rg_step : float, optional
            Range step size in meters. If not provided, the range step size
            is set to the range pixel spacing.

        Returns
        -------
        _ : RadarGridParameters
            RadarGridParameters constructed from class members.
        """

        length, width = self.shape
        if az_step is None:
            az_step = self.azimuth_time_interval
        else:
            if az_step < 0:
                raise ValueError("az_step cannot be negative")
            length_in_seconds = length * self.azimuth_time_interval
            if az_step > length_in_seconds:
                raise ValueError("az_step cannot be larger than radar grid")
            length = int(length_in_seconds / az_step)

        if rg_step is None:
            rg_step = self.range_pixel_spacing
        else:
            if rg_step < 0:
                raise ValueError("rg_step cannot be negative")
            width_in_meters = width * self.range_pixel_spacing
            if rg_step > width_in_meters:
                raise ValueError("rg_step cannot be larger than radar grid")
            width = int(width_in_meters / rg_step)

        prf = 1 / az_step

        time_delta = datetime.timedelta(days=2)
        ref_epoch = isce3.core.DateTime(self.sensing_start - time_delta)
        # sensing start with respect to reference epoch
        sensing_start = time_delta.total_seconds()

        # init radar grid
        return isce3.product.RadarGridParameters(
            sensing_start,
            self.wavelength,
            prf,
            self.starting_range,
            rg_step,
            isce3.core.LookSide.Right,
            length,
            width,
            ref_epoch,
        )

    def slc_to_file(self, out_path: str, fmt: str = "ENVI"):
        """Write burst to GTiff file.

        Parameters:
        -----------
        out_path : string
            Path of output GTiff file.
        """
        if not self.tiff_path:
            warn_str = "Unable write SLC to file. Burst does not contain image data; only metadata."
            warnings.warn(warn_str)
            return

        # get output directory of out_path
        dst_dir, _ = os.path.split(out_path)

        # create VRT; make temporary if output not VRT
        if fmt != "VRT":
            temp_vrt = tempfile.NamedTemporaryFile(dir=dst_dir)
            vrt_fname = temp_vrt.name
        else:
            vrt_fname = out_path
        self.slc_to_vrt_file(vrt_fname)

        if fmt == "VRT":
            return

        # open temporary VRT and translate to GTiff
        src_ds = gdal.Open(vrt_fname)
        gdal.Translate(out_path, src_ds, format=fmt)

        # clean up
        src_ds = None

    def slc_to_vrt_file(self, out_path):
        """Write burst to VRT file.

        Parameters:
        -----------
        out_path : string
            Path of output VRT file.
        """
        if not self.tiff_path:
            warn_str = "Unable write SLC to file. Burst does not contain image data; only metadata."
            warnings.warn(warn_str)
            return

        line_offset = self.i_burst * self.shape[0]

        inwidth = self.last_valid_sample - self.first_valid_sample + 1
        inlength = self.last_valid_line - self.first_valid_line + 1
        outlength, outwidth = self.shape
        yoffset = line_offset + self.first_valid_line
        localyoffset = self.first_valid_line
        xoffset = self.first_valid_sample
        gdal_obj = gdal.Open(self.tiff_path, gdal.GA_ReadOnly)
        fullwidth = gdal_obj.RasterXSize
        fulllength = gdal_obj.RasterYSize

        # TODO maybe cleaner to write with ElementTree
        tmpl = f'''<VRTDataset rasterXSize="{outwidth}" rasterYSize="{outlength}">
    <VRTRasterBand dataType="CFloat32" band="1">
        <NoDataValue>0.0</NoDataValue>
        <SimpleSource>
            <SourceFilename relativeToVRT="1">{self.tiff_path}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SourceProperties RasterXSize="{fullwidth}" RasterYSize="{fulllength}" DataType="CInt16"/>
            <SrcRect xOff="{xoffset}" yOff="{yoffset}" xSize="{inwidth}" ySize="{inlength}"/>
            <DstRect xOff="{xoffset}" yOff="{localyoffset}" xSize="{inwidth}" ySize="{inlength}"/>
        </SimpleSource>
    </VRTRasterBand>
</VRTDataset>'''

        with open(out_path, "w") as fid:
            fid.write(tmpl)

    def get_az_carrier_poly(
        self,
        offset=0.0,
        xstep=500,
        ystep=50,
        az_order=5,
        rg_order=3,
        index_as_coord=False,
    ):
        """
        Estimate burst azimuth carrier polymonials
        Parameters
        ----------
        offset: float
            Offset between reference and secondary bursts
        xstep: int
            Spacing along x direction
        ystep: int
            Spacing along y direction
        az_order: int
            Azimuth polynomial order
        rg_order: int
            Slant range polynomial order
        index_as_coord: bool
            If true, polyfit with az/range indices. Else, polyfit with az/range.

        Returns
        -------
        poly: isce3.core.Poly2D
           class represents a polynomial function of range
           'x' and azimuth 'y'
        """

        rdr_grid = self.as_isce3_radargrid()

        lines, samples = self.shape
        x = np.arange(0, samples, xstep, dtype=int)
        y = np.arange(0, lines, ystep, dtype=int)
        x_mesh, y_mesh = np.meshgrid(x, y)

        # Estimate azimuth carrier
        az_carr_comp = self.az_carrier_components(
            offset=offset, position=(y_mesh, x_mesh)
        )

        # Fit azimuth carrier polynomial with x/y or range/azimuth
        if index_as_coord:
            az_carrier_poly = polyfit(
                x_mesh.flatten() + 1,
                y_mesh.flatten() + 1,
                az_carr_comp.carrier.flatten(),
                az_order,
                rg_order,
            )
        else:
            # Convert x/y to range/azimuth
            rg = self.starting_range + (x + 1) * self.range_pixel_spacing
            az = rdr_grid.sensing_start + (y + 1) * self.azimuth_time_interval
            rg_mesh, az_mesh = np.meshgrid(rg, az)

            # Estimate azimuth carrier polynomials
            az_carrier_poly = polyfit(
                rg_mesh.flatten(),
                az_mesh.flatten(),
                az_carr_comp.carrier.flatten(),
                az_order,
                rg_order,
            )

        return az_carrier_poly

    def as_dict(self):
        """
        Return SLC class attributes as dict

        Returns
        -------
        self_as_dict: dict
           Dict representation as a dict
        """
        self_as_dict = {}
        for key, val in self.__dict__.items():
            if key == "sensing_start":
                val = str(val)
            elif key == "center":
                val = val.coords[0]
            elif isinstance(val, np.float64):
                val = float(val)
            elif key == "azimuth_fm_rate":
                temp = {}
                temp["order"] = val.order
                temp["mean"] = val.mean
                temp["std"] = val.std
                temp["coeffs"] = val.coeffs
                val = temp
            elif key == "burst_id":
                val = str(val)
            elif key == "border":
                val = self.border[0].wkt
            elif key == "doppler":
                temp = {}

                temp["poly1d"] = {}
                temp["poly1d"]["order"] = val.poly1d.order
                temp["poly1d"]["mean"] = val.poly1d.mean
                temp["poly1d"]["std"] = val.poly1d.std
                temp["poly1d"]["coeffs"] = val.poly1d.coeffs

                temp["lut2d"] = {}
                temp["lut2d"]["x_start"] = val.lut2d.x_start
                temp["lut2d"]["x_spacing"] = val.lut2d.x_spacing
                temp["lut2d"]["y_start"] = val.lut2d.y_start
                temp["lut2d"]["y_spacing"] = val.lut2d.y_spacing
                temp["lut2d"]["length"] = val.lut2d.length
                temp["lut2d"]["width"] = val.lut2d.width
                temp["lut2d"]["data"] = val.lut2d.data.flatten().tolist()

                val = temp
            elif key == "orbit":
                temp = {}
                temp["ref_epoch"] = str(val.reference_epoch)
                temp["time"] = {}
                temp["time"]["first"] = val.time.first
                temp["time"]["spacing"] = val.time.spacing
                temp["time"]["last"] = val.time.last
                temp["time"]["size"] = val.time.size
                temp["position_x"] = val.position[:, 0].tolist()
                temp["position_y"] = val.position[:, 1].tolist()
                temp["position_z"] = val.position[:, 2].tolist()
                temp["velocity_x"] = val.velocity[:, 0].tolist()
                temp["velocity_y"] = val.velocity[:, 1].tolist()
                temp["velocity_z"] = val.velocity[:, 2].tolist()
                val = temp
            self_as_dict[key] = val
        return self_as_dict

    def _steps_to_vecs(self, range_step: float, az_step: float):
        """Convert range_step (meters) and az_step (sec) into aranges to generate LUT2ds."""
        radargrid = self.as_isce3_radargrid(az_step=az_step, rg_step=range_step)
        n_az, n_range = radargrid.shape

        range_vec = self.starting_range + np.arange(0, n_range) * range_step
        az_vec = radargrid.sensing_start + np.arange(0, n_az) * az_step
        return range_vec, az_vec

    def bistatic_delay(self, range_step=1, az_step=1):
        """Computes the bistatic delay correction in azimuth direction
        due to the movement of the platform between pulse transmission and echo reception
        as described in equation (21) in Gisinger et al. (2021, TGRS).

        References
        -------
        Gisinger, C., Schubert, A., Breit, H., Garthwaite, M., Balss, U., Willberg, M., et al.
          (2021). In-Depth Verification of Sentinel-1 and TerraSAR-X Geolocation Accuracy Using
          the Australian Corner Reflector Array. IEEE Trans. Geosci. Remote Sens., 59(2), 1154-
          1181. doi:10.1109/TGRS.2019.2961248
        ETAD-DLR-DD-0008, Algorithm Technical Baseline Document. Available: https://sentinels.
          copernicus.eu/documents/247904/4629150/Sentinel-1-ETAD-Algorithm-Technical-Baseline-
          Document.pdf

        Parameters
        -------
        range_step : int
            Spacing along x/range direction [meters]
        az_step : int
            Spacing along y/azimuth direction [seconds]

        Returns
        -------
           LUT2D object of bistatic delay correction in seconds as a function
           of the azimuth time and slant range, or range and azimuth indices.
           This correction needs to be added to the SLC tagged azimuth time to
           get the corrected azimuth times.
        """

        pri = 1.0 / self.prf_raw_data
        tau0 = self.rank * pri
        tau_mid = self.iw2_mid_range * 2.0 / isce3.core.speed_of_light

        slant_vec, az_vec = self._steps_to_vecs(range_step, az_step)

        tau = slant_vec * 2.0 / isce3.core.speed_of_light

        # the first term (tau_mid/2) is the bulk bistatic delay which was
        # removed from the orginial azimuth time by the ESA IPF. Based on
        # Gisinger et al. (2021) and ETAD ATBD, ESA IPF has used the mid of
        # the second subswath to compute the bulk bistatic delay. However
        # currently we have not been able to verify this from ESA documents.
        # This implementation follows the Gisinger et al. (2021) for now, we
        # can revise when we hear back from ESA folks.
        bistatic_correction_vec = tau_mid / 2 + tau / 2 - tau0
        ny = az_vec.size
        bistatic_correction = np.tile(bistatic_correction_vec.reshape(1, -1), (ny, 1))

        return isce3.core.LUT2d(slant_vec, az_vec, bistatic_correction)

    def geometrical_and_steering_doppler(self, range_step=500, az_step=50):
        """
        Compute total Doppler which is the sum of two components:
        (1) the geometrical Doppler induced by the relative movement
        of the sensor and target
        (2) the TOPS specicifc Doppler caused by the electric steering
        of the beam along the azimuth direction resulting in Doppler varying
        with azimuth time.
        Parameters
        ----------
        range_step: int
            Spacing along x/range direction [meters]
        az_step: int
            Spacing along y/azimuth direction [seconds]

        Returns
        -------
           LUT2D object of total doppler in Hz as a function of the azimuth
           time and slant range, or range and azimuth indices.
           This correction needs to be added to the SLC tagged azimuth time to
           get the corrected azimuth times.
        """
        range_vec, az_vec = self._steps_to_vecs(range_step, az_step)

        # convert from meters to pixels
        x_vec = (range_vec - self.starting_range) / self.range_pixel_spacing

        # convert from seconds to pixels
        rdrgrid = self.as_isce3_radargrid()
        y_vec = (az_vec - rdrgrid.sensing_start) / self.azimuth_time_interval

        # compute az carrier components with pixels
        x_mesh, y_mesh = np.meshgrid(x_vec, y_vec)
        az_carr_comp = self.az_carrier_components(offset=0.0, position=(y_mesh, x_mesh))

        geometrical_doppler = self.doppler.poly1d.eval(range_vec)

        total_doppler = az_carr_comp.antenna_steering_doppler + geometrical_doppler

        return isce3.core.LUT2d(range_vec, az_vec, total_doppler)

    def doppler_induced_range_shift(self, range_step=500, az_step=50):
        """
        Computes the range delay caused by the Doppler shift as described
        by Gisinger et al 2021

        Parameters
        ----------
        range_step: int
            Spacing along x/range direction [meters]
        az_step: int
            Spacing along y/azimuth direction [seconds]

        Returns
        -------
        isce3.core.LUT2d:
           LUT2D object of range delay correction [seconds] as a function
           of the azimuth time and slant range, or x and y indices.

        """
        range_vec, az_vec = self._steps_to_vecs(range_step, az_step)

        doppler_shift = self.geometrical_and_steering_doppler(
            range_step=range_step, az_step=az_step
        )
        tau_corr = doppler_shift.data / self.range_chirp_rate

        return isce3.core.LUT2d(range_vec, az_vec, tau_corr)

    def az_carrier_components(self, offset, position):
        """
        Estimate azimuth carrier and store in numpy arrary. Also return
        contributing components.

        Parameters
        ----------
        offset: float
           Offset between reference and secondary burst
        position: tuple
           Tuple of locations along y and x directions in pixels

        Returns
        -------
        eta: float
            zero-Doppler azimuth time centered in the middle of the burst
        eta_ref: float
            refernce time
        kt: np.ndarray
            Doppler centroid rate in the focused TOPS SLC data [Hz/s]
        carr: np.ndarray
           Azimuth carrier

        Reference
        ---------
        https://sentinels.copernicus.eu/documents/247904/0/Sentinel-1-TOPS-SLC_Deramping/b041f20f-e820-46b7-a3ed-af36b8eb7fa0
        """
        # Get self.sensing mid relative to orbit reference epoch
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        orbit_ref_epoch = datetime.datetime.strptime(
            self.orbit.reference_epoch.__str__()[:-3], fmt
        )

        t_mid = self.sensing_mid - orbit_ref_epoch
        _, v = self.orbit.interpolate(t_mid.total_seconds())
        vs = np.linalg.norm(v)
        ks = 2 * vs * self.azimuth_steer_rate / self.wavelength

        y, x = position

        n_lines, _ = self.shape
        eta = (y - (n_lines // 2) + offset) * self.azimuth_time_interval
        rng = self.starting_range + x * self.range_pixel_spacing

        f_etac = np.array(self.doppler.poly1d.eval(rng.flatten().tolist())).reshape(
            rng.shape
        )
        ka = np.array(self.azimuth_fm_rate.eval(rng.flatten().tolist())).reshape(
            rng.shape
        )

        eta_ref = (
            self.doppler.poly1d.eval(self.starting_range)
            / self.azimuth_fm_rate.eval(self.starting_range)
        ) - (f_etac / ka)
        kt = ks / (1.0 - ks / ka)

        return AzimuthCarrierComponents(kt, eta, eta_ref)

    def az_fm_rate_mismatch_mitigation(
        self,
        path_dem: str,
        path_scratch: str = None,
        range_step=None,
        az_step=None,
        threshold_rdr2geo=1e-8,
        numiter_rdr2geo=25,
    ):
        """
        - Calculate Lon / Lat / Hgt in radar grid, to be used for the
          actual computation of az fm mismatch rate
        - Define the radar grid for the correction.
        - call `az_fm_rate_mismatch_from_llh` to do the actual computation


        Parameters:
        -----------
        path_dem: str
            Path to the DEM to calculate the actual azimuth FM rate
        path_scratch: str
            Path to the scratch directory to store intermediate data
            If None, the scratch files will be saved on temporary directory generated by system
        range_step: float
            Range step of the correction grid in meters
        az_step: float
            Azimuth step of the correction grid in seconds
        threshold_rdr2geo: int
            Threshold of the iteration for rdr2geo
        numiter_rdr2geo: int
            Maximum number of iteration for rdr2geo

        Return:
        -------
        _: isce3.core.LUT2d
            azimuth FM rate mismatch rate in radar grid in seconds.

        Examples
        ----------
        >>> correction_grid = burst.az_fm_rate_mismatch_mitigation("my_dem.tif")
        """

        # Create temporary directory for scratch when
        # `path_scratch` is not provided.
        if path_scratch is None:
            temp_dir_obj = tempfile.TemporaryDirectory()
            path_scratch = temp_dir_obj.name
        else:
            temp_dir_obj = None

        os.makedirs(path_scratch, exist_ok=True)

        # define the radar grid to calculate az fm mismatch rate
        correction_radargrid = self.as_isce3_radargrid(
            az_step=az_step, rg_step=range_step
        )

        # Run topo on scratch directory
        if not os.path.isfile(path_dem):
            raise FileNotFoundError(f"not found - DEM {path_dem}")
        dem_raster = isce3.io.Raster(path_dem)
        epsg = dem_raster.get_epsg()
        proj = isce3.core.make_projection(epsg)
        ellipsoid = proj.ellipsoid
        grid_doppler = isce3.core.LUT2d()

        rdr2geo_obj = isce3.geometry.Rdr2Geo(
            correction_radargrid,
            self.orbit,
            ellipsoid,
            grid_doppler,
            threshold=threshold_rdr2geo,
            numiter=numiter_rdr2geo,
        )

        str_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        list_filename_llh = [
            f"{path_scratch}/{llh}_{str_datetime}.rdr" for llh in ["lat", "lon", "hgt"]
        ]
        lat_raster, lon_raster, hgt_raster = [
            isce3.io.Raster(
                filename_llh,
                correction_radargrid.width,
                correction_radargrid.length,
                1,
                gdal.GDT_Float64,
                "ENVI",
            )
            for filename_llh in list_filename_llh
        ]

        rdr2geo_obj.topo(dem_raster, lon_raster, lat_raster, hgt_raster)

        # make sure that the ISCE3 rasters are written out to file system
        lat_raster.close_dataset()
        lon_raster.close_dataset()
        hgt_raster.close_dataset()

        # Load the lon / lat / hgt value from the raster
        arr_lat = gdal.Open(list_filename_llh[0], gdal.GA_ReadOnly).ReadAsArray()
        arr_lon = gdal.Open(list_filename_llh[1], gdal.GA_ReadOnly).ReadAsArray()
        arr_hgt = gdal.Open(list_filename_llh[2], gdal.GA_ReadOnly).ReadAsArray()

        lut_az_fm_mismatch = self.az_fm_rate_mismatch_from_llh(
            arr_lat, arr_lon, arr_hgt, ellipsoid, correction_radargrid
        )

        # Clean up the temporary directory in case it exists.
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

        return lut_az_fm_mismatch

    def az_fm_rate_mismatch_from_llh(
        self,
        lat_map: np.ndarray,
        lon_map: np.ndarray,
        hgt_map: np.ndarray,
        ellipsoid: isce3.core.Ellipsoid,
        correction_radargrid: isce3.product.RadarGridParameters,
    ):
        """
        Take in lat / lon / hgt in radar grid, along with ellipsoid and the radar grid.
        Calculate azimuth FM rate mismatch mitigation based on algorithm
        described in [1]

        Parameters:
        -----------
        lat_map: np.ndarray
            Latitude in radar grid, unit: degrees
        lon_map: np.ndarray,
            Longitude in radar grid, unit: degrees
        hgt_map: np.ndarray,
            Height in radar grid, unit: meters
        ellipsoid: isce3.core.Ellipsoid
            Reference ellipsoid to be used
        correction_radargrid: isce3.product.RadarGridParameters
            Radar grid as the definition of the correction grid

        Return:
        -------
        _: isce3.core.LUT2d
            azimuth FM rate mismatch rate in radar grid in seconds.

        References
        ----------
        [1] C. Gisinger, "S-1 ETAD project Algorithm Technical
            Baseline Document, The integration of GIS, remote sensing,"
            DLR, ETAD-DLR-DD-0008, 2020.
        """

        # Define the correction grid from the radargrid
        # Also define the staggered grid in azimuth to calculate acceeration
        width_grid = correction_radargrid.width
        length_grid = correction_radargrid.length
        intv_t = 1.0 / correction_radargrid.prf
        intv_tau = (
            correction_radargrid.range_pixel_spacing * 2.0 / isce3.core.speed_of_light
        )
        delta_sec_from_ref_epoch = (
            correction_radargrid.ref_epoch - self.orbit.reference_epoch
        ).total_seconds() + correction_radargrid.sensing_start
        vec_t = np.arange(length_grid) * intv_t + delta_sec_from_ref_epoch
        vec_t_staggered = (
            np.arange(length_grid + 1) * intv_t + delta_sec_from_ref_epoch - intv_t / 2
        )

        vec_tau = np.arange(width_grid) * intv_tau

        grid_tau, grid_t = np.meshgrid(vec_tau, vec_t)

        vec_position_intp = np.zeros((length_grid, 3))
        vec_vel_intp = np.zeros((length_grid, 3))
        vec_vel_intp_staggered = [None] * (length_grid + 1)

        for i_azimuth, t_azimuth in enumerate(vec_t):
            vec_position_intp[i_azimuth, :], vec_vel_intp[i_azimuth, :] = (
                self.orbit.interpolate(t_azimuth)
            )

        # Calculate velocity on staggered grid
        # to kinematically calculate acceleration
        for i_azimuth, t_azimuth in enumerate(vec_t_staggered):
            _, vec_vel_intp_staggered[i_azimuth] = self.orbit.interpolate(t_azimuth)

        vec_acceleration_intp = np.diff(vec_vel_intp_staggered, axis=0) / intv_t

        # convert azimuth time to seconds from the reference epoch of burst_in.orbit
        fm_rate_aztime_sec_vec = [
            (
                isce3.core.DateTime(datetime_vec) - self.orbit.reference_epoch
            ).total_seconds()
            for datetime_vec in self.extended_coeffs.fm_rate_aztime_vec
        ]

        dc_aztime_sec_vec = [
            (
                isce3.core.DateTime(datetime_vec) - self.orbit.reference_epoch
            ).total_seconds()
            for datetime_vec in self.extended_coeffs.dc_aztime_vec
        ]

        # calculate splined interpolation of the coeffs. and tau_0s
        if len(fm_rate_aztime_sec_vec) <= 1:
            # Interpolator object cannot be created with only one set of polynomials.
            # Such case happens when there is no polygon that falls between a burst's start / stop
            # which was found from very few Sentinel-1 IW SLCs processed by IPF 002.36
            # Return an empty LUT2d in that case
            return isce3.core.LUT2d()
        interpolator_tau0_ka = InterpolatedUnivariateSpline(
            fm_rate_aztime_sec_vec, self.extended_coeffs.fm_rate_tau0_vec, ext=3, k=1
        )
        tau0_ka_interp = interpolator_tau0_ka(vec_t)[..., np.newaxis]

        if len(dc_aztime_sec_vec) <= 1:
            # Interpolator object cannot be created with only one set of polynomials.
            # Such case happens when there is no polygon that falls between a burst's start / stop
            # which was found from very few Sentinel-1 IW SLCs processed by IPF 002.36
            # Return an empty LUT2d in that case
            return isce3.core.LUT2d()
        interpolator_tau0_fdc_interp = InterpolatedUnivariateSpline(
            dc_aztime_sec_vec, self.extended_coeffs.dc_tau0_vec, ext=3, k=1
        )
        tau0_fdc_interp = interpolator_tau0_fdc_interp(vec_t)[..., np.newaxis]

        # add range time origin to vec_tau
        grid_tau += tau0_ka_interp * np.ones(vec_tau.shape)[np.newaxis, ...]

        # Interpolate the DC and fm rate coeffs along azimuth time
        def interp_coeffs(az_time, coeffs, az_time_interp):
            """Convenience function to interpolate DC and FM rate coefficients"""
            interpolated_coeffs = []
            for i in range(3):
                coeff_interpolator = InterpolatedUnivariateSpline(
                    az_time, coeffs[:, i], k=1
                )
                interpolated_coeffs.append(coeff_interpolator(az_time_interp))
            return np.array(interpolated_coeffs).transpose()

        dc_coeffs = interp_coeffs(
            dc_aztime_sec_vec, self.extended_coeffs.dc_coeff_arr, vec_t
        )

        fm_rate_coeffs = interp_coeffs(
            fm_rate_aztime_sec_vec, self.extended_coeffs.fm_rate_coeff_arr, vec_t
        )

        # Do the bunch of computation
        kappa_steer_vec = (
            2
            * (np.linalg.norm(vec_vel_intp, axis=1))
            / isce3.core.speed_of_light
            * self.radar_center_frequency
            * self.azimuth_steer_rate
        )

        kappa_steer_grid = (
            kappa_steer_vec[..., np.newaxis]
            * np.ones(grid_tau.shape[1])[np.newaxis, ...]
        )

        t_burst = (grid_t[0, 0] + grid_t[-1, 0]) / 2.0
        index_mid_burst_t = int(grid_t.shape[0] / 2 + 0.5)
        tau_burst = (
            grid_tau[index_mid_burst_t, 0] + grid_tau[index_mid_burst_t, -1]
        ) / 2.0
        tau0_fdc_burst = tau0_fdc_interp[index_mid_burst_t]
        tau0_fm_rate_burst = tau0_ka_interp[index_mid_burst_t]

        a0_burst, a1_burst, a2_burst = dc_coeffs[index_mid_burst_t, :]
        b0_burst, b1_burst, b2_burst = fm_rate_coeffs[index_mid_burst_t, :]

        kappa_annotation_burst = (
            b0_burst
            + b1_burst * (tau_burst - tau0_fm_rate_burst)
            + b2_burst * (tau_burst - tau0_fm_rate_burst) ** 2
        )

        kappa_annotation_grid = _evaluate_polynomial_array(
            fm_rate_coeffs, grid_tau, tau0_ka_interp
        )

        grid_kappa_t = (kappa_annotation_grid * kappa_steer_grid) / (
            kappa_annotation_grid - kappa_steer_grid
        )
        freq_dcg_burst = (
            a0_burst
            + a1_burst * (tau_burst - tau0_fdc_burst)
            + a2_burst * (tau_burst - tau0_fdc_burst) ** 2
        )

        grid_freq_dcg = _evaluate_polynomial_array(dc_coeffs, grid_tau, tau0_fdc_interp)

        grid_freq_dc = grid_freq_dcg + grid_kappa_t * (
            (grid_t - t_burst)
            + (
                freq_dcg_burst / kappa_annotation_burst
                - grid_freq_dcg / kappa_annotation_grid
            )
        )

        x_ecef, y_ecef, z_ecef = _llh_to_ecef(lat_map, lon_map, hgt_map, ellipsoid)

        # Populate the position, velocity, and
        # acceleration of the satellite to the correction grid
        tau_ones = np.ones(grid_tau.shape[1])[np.newaxis, ...]
        x_s, y_s, z_s = [
            vec_position_intp[:, i][..., np.newaxis] * tau_ones for i in range(3)
        ]
        vx_s, vy_s, vz_s = [
            vec_vel_intp[:, i][..., np.newaxis] * tau_ones for i in range(3)
        ]
        ax_s, ay_s, az_s = [
            vec_acceleration_intp[:, i][..., np.newaxis] * tau_ones for i in range(3)
        ]

        mag_xs_xg = np.sqrt(
            (x_s - x_ecef) ** 2 + (y_s - y_ecef) ** 2 + (z_s - z_ecef) ** 2
        )

        dotp_dxsg_acc = (
            (x_s - x_ecef) * ax_s + (y_s - y_ecef) * ay_s + (z_s - z_ecef) * az_s
        )

        kappa_annotation_true = (
            -(2 * self.radar_center_frequency)
            / (isce3.core.speed_of_light * mag_xs_xg)
            * (dotp_dxsg_acc + (vx_s**2 + vy_s**2 + vz_s**2))
        )

        delta_t_freq_mm = grid_freq_dc * (
            -1 / kappa_annotation_grid + 1 / kappa_annotation_true
        )

        # Prepare to export to LUT2d
        vec_range = grid_tau[index_mid_burst_t, :] * isce3.core.speed_of_light / 2.0

        return isce3.core.LUT2d(vec_range, vec_t, delta_t_freq_mm)

    @property
    def sensing_mid(self):
        """Returns sensing mid as datetime.datetime object.

        Returns:
        --------
        _ : datetime.datetime
            Sensing mid as datetime.datetime object.
        """
        d_seconds = 0.5 * self.length * self.azimuth_time_interval
        return self.sensing_start + datetime.timedelta(seconds=d_seconds)

    @property
    def sensing_stop(self):
        """Returns sensing end as datetime.datetime object.

        Returns:
        --------
        _ : datetime.datetime
            Sensing end as datetime.datetime object.
        """
        d_seconds = (self.length - 1) * self.azimuth_time_interval
        return self.sensing_start + datetime.timedelta(seconds=d_seconds)

    @property
    def burst_duration(self):
        """Returns burst sensing duration as float in seconds.

        Returns:
        --------
        _ : float
            Burst sensing duration as float in seconds.
        """
        return self.azimuth_time_interval * self.length

    @property
    def length(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    @property
    def swath_name(self):
        """Swath name in iw1, iw2, iw3."""
        return self.burst_id.subswath.lower()

    @property
    def thermal_noise_lut(self):
        """
        Returns the LUT for thermal noise correction for the burst
        """
        if self.burst_noise is None:
            raise ValueError("burst_noise is not defined for this burst.")

        return self.burst_noise.compute_thermal_noise_lut(self.shape)

    @property
    def eap_compensation_lut(self):
        """Returns LUT for EAP compensation.

        Returns:
        -------
            _: Interpolated EAP gain for the burst's lines

        """
        if self.burst_eap is None:
            raise ValueError(
                "burst_eap is not defined for this burst."
                f" IPF version = {self.ipf_version}"
            )

        return self.burst_eap.compute_eap_compensation_lut(self.width)

    @property
    def relative_orbit_number(self):
        """Returns the relative orbit number of the burst."""
        orbit_number_offset = 73 if self.platform_id == "S1A" else 202
        return (self.abs_orbit_number - orbit_number_offset) % 175 + 1
