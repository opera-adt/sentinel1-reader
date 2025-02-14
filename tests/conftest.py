import pathlib
import types

import pytest

from s1reader import s1_reader


@pytest.fixture(scope="session")
def test_paths():
    test_paths = types.SimpleNamespace()

    test_path = pathlib.Path(__file__).parent.resolve()
    test_paths.safe = f"{test_path}/data/S1A_IW_SLC__1SDV_20200511T135117_20200511T135144_032518_03C421_7768.zip"
    test_paths.orbit_dir = f"{test_path}/data/orbits"
    test_paths.orbit_file = (
        "S1A_OPER_AUX_POEORB_OPOD_20210318T120818_V20200510T225942_20200512T005942.EOF"
    )

    return test_paths


@pytest.fixture(scope="session")
def bursts(test_paths):
    i_subswath = 3
    pol = "vv"

    orbit_path = f"{test_paths.orbit_dir}/{test_paths.orbit_file}"
    bursts = s1_reader.load_bursts(test_paths.safe, orbit_path, i_subswath, pol)

    return bursts
