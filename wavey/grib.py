import datetime
from enum import IntEnum
from typing import NamedTuple

import numpy as np
import pygrib

from wavey.common import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, TZ_UTC

NUM_DATA_POINTS = 145  # 1 + 24 * 6 hours
"""Number of data points for each forecast type in the NWFS GRIB file."""


class ForecastType(IntEnum):
    """Forecast types in the NWFS GRIB file, in order."""

    WaveHeight = 0
    """Significant height of combined wind waves and swell (m)"""
    WaveDirection = 1
    """Primary wave direction (deg)"""
    WavePeriod = 2
    """Primary wave mean period (s)"""
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
    """Array with shape (NUM_DATA_POINTS, LATS, LONS). May contain missing values."""
    lats: np.ndarray
    """Array with shape (LATS, LONS)."""
    lons: np.ndarray
    """Array with shape (LATS, LONS)."""
    analysis_date_utc: datetime.datetime
    """Date and time of analysis, i.e. start of forecast, in UTC."""


def read_forecast_data(grbs: pygrib.open, forecast_type: ForecastType) -> ForecastData:
    """
    Read forecast data from Monterey Bay NWFS GRIB file, zoomed-in near the
    peninsula (see {LAT|LON}_{MIN|MAX} values).

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
        data, lats, lons = grb.data(lat1=LAT_MIN, lat2=LAT_MAX, lon1=LON_MIN, lon2=LON_MAX)
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
