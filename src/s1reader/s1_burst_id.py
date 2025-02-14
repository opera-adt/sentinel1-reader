import datetime
from dataclasses import dataclass
from typing import ClassVar
from s1reader.s1_orbit import T_ORBIT
import numpy as np


@dataclass(frozen=True)
class S1BurstId:
    # Constants in Table 9-7 of Sentinel-1 SLC Detailed Algorithm Definition
    T_beam: ClassVar[float] = 2.758273  # interval of one burst [s]
    T_pre: ClassVar[float] = 2.299849  # Preamble time interval [s]
    T_orb: ClassVar[float] = T_ORBIT  # Nominal orbit period [s]
    track_number: int
    esa_burst_id: int
    subswath: str

    @classmethod
    def from_burst_params(
        cls,
        sensing_time: datetime.datetime,
        ascending_node_dt: datetime.datetime,
        start_track: int,
        end_track: int,
        subswath: str,
    ):
        """Calculate the unique burst ID (track, ESA burstId, swath) of a burst.

        Accounts for equator crossing frames, where the current track number of
        a burst may change mid-frame. Uses the ESA convention defined in the
        Sentinel-1 Level 1 Detailed Algorithm Definition.

        Parameters
        ----------
        sensing_time : datetime
            Sensing time of the first input line of this burst [UTC]
            The XML tag is sensingTime in the annotation file.
        ascending_node_dt : datetime
            Time of the ascending node prior to the start of the scene.
        start_track : int
            Relative orbit number at the start of the acquisition, from 1-175.
        end_track : int
            Relative orbit number at the end of the acquisition.
        subswath : str, {'IW1', 'IW2', 'IW3'}
            Name of the subswath of the burst (not case sensitive).

        Returns
        -------
        S1BurstId
            The burst ID object containing track number + ESA's burstId number + swath ID.

        Notes
        -----
        The `start_track` and `end_track` parameters are used to determine if the
        scene crosses the equator. They are the same if the frame does not cross
        the equator.

        References
        ----------
        ESA Sentinel-1 Level 1 Detailed Algorithm Definition
        https://sentinels.copernicus.eu/documents/247904/1877131/S1-TN-MDA-52-7445_Sentinel-1+Level+1+Detailed+Algorithm+Definition_v2-4.pdf/83624863-6429-cfb8-2371-5c5ca82907b8
        """
        swath_num = int(subswath[-1])
        # Since we only have access to the current subswath, we need to use the
        # burst-to-burst times to figure out
        #   1. if IW1 crossed the equator, and
        #   2. The mid-burst sensing time for IW2
        # IW1 -> IW2 takes ~0.83220 seconds
        # IW2 -> IW3 takes ~1.07803 seconds
        # IW3 -> IW1 takes ~0.84803 seconds
        burst_times = np.array([0.832, 1.078, 0.848])
        iw1_start_offsets = [
            0,
            -burst_times[0],
            -burst_times[0] - burst_times[1],
        ]
        offset = iw1_start_offsets[swath_num - 1]
        start_iw1 = sensing_time + datetime.timedelta(seconds=offset)

        start_iw1_to_mid_iw2 = burst_times[0] + burst_times[1] / 2
        mid_iw2 = start_iw1 + datetime.timedelta(seconds=start_iw1_to_mid_iw2)

        has_anx_crossing = (end_track == start_track + 1) or (
            end_track == 1 and start_track == 175
        )

        time_since_anx_iw1 = (start_iw1 - ascending_node_dt).total_seconds()
        time_since_anx = (mid_iw2 - ascending_node_dt).total_seconds()

        if (time_since_anx_iw1 - cls.T_orb) < 0:
            # Less than a full orbit has passed
            track_number = start_track
        else:
            track_number = end_track
            # Additional check for scenes which have a given ascending node
            # that's more than 1 orbit in the past
            if not has_anx_crossing:
                time_since_anx = time_since_anx - cls.T_orb

        # Eq. 9-89: ∆tb = tb − t_anx + (r - 1)T_orb
        # tb: mid-burst sensing time (sensing_time)
        # t_anx: ascending node time (ascending_node_dt)
        # r: relative orbit number   (relative_orbit_start)
        dt_b = time_since_anx + (start_track - 1) * cls.T_orb

        # Eq. 9-91 :   1 + floor((∆tb − T_pre) / T_beam )
        esa_burst_id = 1 + int(np.floor((dt_b - cls.T_pre) / cls.T_beam))

        return cls(track_number, esa_burst_id, subswath)

    @classmethod
    def from_str(cls, burst_id_str: str):
        """Parse a S1BurstId object from a string.

        Parameters
        ----------
        burst_id_str : str
            The burst ID string, e.g. "t123_000456_iw1"

        Returns
        -------
        S1BurstId
            The burst ID object containing track number + ESA's burstId number + swath ID.
        """
        track_number, esa_burst_id, subswath = burst_id_str.split("_")
        track_number = int(track_number[1:])
        esa_burst_id = int(esa_burst_id)
        return cls(track_number, esa_burst_id, subswath.lower())

    def __str__(self):
        # Form the unique JPL ID by combining track/burst/swath
        return (
            f"t{self.track_number:03d}_{self.esa_burst_id:06d}_{self.subswath.lower()}"
        )

    def __eq__(self, other) -> bool:
        # Allows for comparison with strings, as well as S1BurstId objects
        # e.g., you can filter down burst IDs with:
        # burst_ids = ["t012_024518_iw3", "t012_024519_iw3"]
        # bursts = [b for b in bursts if b.burst_id in burst_ids]
        if isinstance(other, str):
            return str(self) == other
        else:
            return super().__eq__(other)
