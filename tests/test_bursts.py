from datetime import timedelta

import isce3
import numpy as np


def test_burst(bursts):
    last_valid_lines = [1487, 1489, 1489, 1490, 1487, 1488, 1488, 1489, 1488]
    first_valid_lines = [28, 27, 27, 27, 28, 28, 28, 27, 28]
    doppler_poly1d_coeffs = [
        [-22.63739, 41814.14, -44259730.0],
        [-8.528021, -33560.36, 28706860.0],
        [-4.17043, -44729.37, 32448680.0],
        [-8.181067, -30353.52, 17331050.0],
        [-16.11892, 26711.75, -34255860.0],
        [-10.66584, 2126.559, -15965500.0],
        [-11.98844, 13693.94, -20996410.0],
        [3.296118, -44014.56, 29321840.0],
        [-5.65537, 8260.13, -21226210.0],
    ]

    az_fm_rate_poly1d_coeffs = [
        [-2056.065941779171, 353463.4449453847, -54169735.41467991],
        [-2056.139679961154, 353454.5995180805, -54167947.63163446],
        [-2056.227137638971, 353445.5676283827, -54163311.51609433],
        [-2056.310368464548, 353436.8144712183, -54159705.64889784],
        [-2056.399182330127, 353430.0278188085, -54159190.52496677],
        [-2056.481997647131, 353420.3529224118, -54153675.66869747],
        [-2056.555248889294, 353410.1388230529, -54149584.97851501],
        [-2056.633884304171, 353399.9756644769, -54144676.15045655],
        [-2056.701472691132, 353389.9614836443, -54143009.57327797],
    ]

    for i, burst in enumerate(bursts):
        expected_burst_id = f"t071_{151199 + i}_iw3"
        assert burst.burst_id == expected_burst_id
        assert burst.i_burst == i
        assert burst.abs_orbit_number == 32518

        assert burst.radar_center_frequency == 5405000454.33435
        assert burst.wavelength == 0.05546576
        assert burst.azimuth_steer_rate == 0.024389943375862838

        # the average azimuth pixel spacing varies across bursts
        assert burst.average_azimuth_pixel_spacing > 13.9
        assert burst.average_azimuth_pixel_spacing < 14.0

        assert burst.starting_range == 901673.89084624
        assert burst.iw2_mid_range == 875604.926001518
        assert burst.range_sampling_rate == 64345238.12571428
        assert burst.range_pixel_spacing == 2.329562114715323
        assert burst.shape == (1515, 24492)
        assert burst.range_bandwidth == 42789918.40322842

        assert burst.polarization == "VV"
        assert burst.platform_id == "S1A"

        assert burst.range_window_type == "Hamming"
        assert burst.range_window_coefficient == 0.75
        assert burst.rank == 10
        assert burst.prf_raw_data == 1685.817302492702
        assert burst.range_chirp_rate == 801450949070.5804

        assert burst.first_valid_sample == 451
        assert burst.last_valid_sample == 24119
        assert burst.first_valid_line == first_valid_lines[i]
        assert burst.last_valid_line == last_valid_lines[i]

        assert burst.doppler.poly1d.order == 2
        assert burst.doppler.poly1d.mean == 800884.7203639568
        assert burst.doppler.poly1d.std == 149896229.0
        assert burst.doppler.poly1d.coeffs == doppler_poly1d_coeffs[i]

        # compare doppler poly1d and lut2d
        r0 = burst.starting_range + 0.5 * burst.width * burst.range_pixel_spacing
        t0 = isce3.core.DateTime(burst.sensing_mid) - burst.orbit.reference_epoch
        assert np.isclose(
            burst.doppler.lut2d.eval(t0.total_seconds(), r0),
            burst.doppler.poly1d.eval(r0),
        )

        assert burst.azimuth_fm_rate.order == 2
        assert burst.azimuth_fm_rate.mean == 901673.89084624
        assert burst.azimuth_fm_rate.std == 149896229.0
        assert burst.azimuth_fm_rate.coeffs == az_fm_rate_poly1d_coeffs[i]


def test_as_isce3_radargrid(bursts):
    for burst in bursts:
        grid = burst.as_isce3_radargrid()
        assert grid.width == burst.width
        assert grid.length == burst.length
        assert grid.starting_range == burst.starting_range
        dt = isce3.core.DateTime((burst.sensing_start - timedelta(days=2)))
        assert dt == grid.ref_epoch
        assert grid.prf == 1 / burst.azimuth_time_interval
        assert grid.range_pixel_spacing == burst.range_pixel_spacing
        assert str(grid.lookside) == "LookSide.Right"
        assert grid.wavelength == burst.wavelength


def test_as_isce3_radargrid_step_change(bursts):
    # Change the az_step, rg_step in .as_isce3_radargrid()
    burst = bursts[0]
    rg_step = burst.range_pixel_spacing
    az_step = burst.azimuth_time_interval
    grid = burst.as_isce3_radargrid(az_step=az_step, rg_step=rg_step)
    assert grid.width == burst.width
    assert grid.length == burst.length
    assert grid.prf == 1 / az_step

    rg_step *= 2
    grid = burst.as_isce3_radargrid(rg_step=rg_step)
    assert grid.width == burst.width // 2
    assert grid.length == burst.length

    az_step *= 2
    grid = burst.as_isce3_radargrid(az_step=az_step)
    assert grid.width == burst.width
    assert grid.length == burst.length // 2
