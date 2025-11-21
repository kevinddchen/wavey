import datetime
from enum import IntEnum
from typing import NamedTuple

import numpy as np
import pygrib

from wavey.common import TZ_UTC

NUM_DATA_POINTS = 145
"""Number of data points for each forecast type in the NWFS GRIB file (1 + 24 * 6 days)."""

NUM_LATS = 178
"""
Number of equally spaced latitudes for the Monterey bay forecast data,
from 36.2 to 37.0 degs (each increment is 0.00452 deg).
"""

NUM_LONS = 90
"""
Number of equally spaced longitudes for the Monterey bay forecast data,
from 237.8 to 238.3 degs (each increment is 0.00562 deg).
"""


class ForecastType(IntEnum):
    """Forecast types in the NWFS GRIB file, in order."""

    WaveHeight = 0
    """Significant height of combined wind waves and swell (m)"""
    WaveDirection = 1
    """Peak wave direction (deg)"""
    WavePeriod = 2
    """Peak wave mean period (s)"""
    SwellHeight = 3
    """Significant height of total swell (m)"""
    WindDirection = 4
    """Wind direction (deg)"""
    WindSpeed = 5
    """Wind speed (m/s)"""
    SeaSurfaceHeight = 6
    """Sea surface height (m)"""
    CurrentDirection = 7
    """Current direction (deg)"""
    CurrentSpeed = 8
    """Current speed (m/s)"""


class ForecastData(NamedTuple):
    """Data for a single forecast type from the NWFS GRIB file."""

    data: np.ma.MaskedArray
    """Array with shape (NUM_DATA_POINTS, NUM_LATS, NUM_LONS). May contain missing values."""
    lats: np.ndarray
    """Array with shape (NUM_LATS, NUM_LONS)."""
    lons: np.ndarray
    """Array with shape (NUM_LATS, NUM_LONS)."""
    analysis_date_utc: datetime.datetime
    """Date and time of analysis, i.e. start of forecast, in UTC."""


def read_forecast_data(grbs: pygrib.open, forecast_type: ForecastType) -> ForecastData:
    """
    Read forecast data from Monterey Bay NWFS GRIB file.

    Args:
        grbs: GRIB file.
        forecast_type: Type of data to read.

    Returns:
        Forecast data of the specified type.
    """

    grbs.seek(forecast_type * NUM_DATA_POINTS)  # message offset

    data_list: list[np.ma.MaskedArray] = []
    lats: np.ndarray | None = None
    lons: np.ndarray | None = None
    analysis_date: datetime.datetime | None = None

    for grb in grbs.read(NUM_DATA_POINTS):
        data, lats, lons = grb.data()
        data_list.append(data)

        if analysis_date is None:
            analysis_date = grb.analDate

    # assertions will fail if no messages were read
    assert lats is not None
    assert lons is not None
    assert analysis_date is not None
    analysis_date_utc = analysis_date.replace(tzinfo=TZ_UTC)
    data_collated = np.ma.stack(data_list)

    return ForecastData(
        data=data_collated,
        lats=lats,
        lons=lons,
        analysis_date_utc=analysis_date_utc,
    )
