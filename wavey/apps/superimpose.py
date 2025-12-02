import datetime
import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pygrib

from wavey.__main__ import (
    BREAKWATER_LAT_IDX,
    BREAKWATER_LON_IDX,
    DPI,
    FEET_PER_METER,
    MONASTERY_LAT_IDX,
    MONASTERY_LON_IDX,
    NUM_DATA_POINTS,
    utc_to_pt,
)
from wavey.common import setup_logging
from wavey.grib import ForecastData, ForecastType, read_forecast_data
from wavey.nwfs import download_forecast, get_all_forecasts

LOG = logging.getLogger(__name__)


def main(
    time: Literal["00", "06"] = "06",
) -> None:
    """
    Superimpose multiple forecasts from different times to see how accurate they are in retrospect.

    Args:
        time: Forecast time in UTC, either "00" or "06".
    """

    all_forecasts = get_all_forecasts(time=time)

    # Download forecast data
    wave_height_forecasts: list[ForecastData] = []
    for forecast in all_forecasts:
        grib_path = download_forecast(forecast)

        with pygrib.open(grib_path) as grbs:
            wave_height_forecasts.append(read_forecast_data(grbs, ForecastType.WaveHeight))

    LOG.info("Plotting graph")
    _, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=DPI)

    for i, wave_height_forecast in enumerate(wave_height_forecasts):
        wave_height_ft = wave_height_forecast.data * FEET_PER_METER
        analysis_date_pacific = utc_to_pt(wave_height_forecast.analysis_date_utc)

        offset = i * 24  # each forecast is 24 hours into the past

        bw_wave_height_ft = wave_height_ft[offset:, BREAKWATER_LAT_IDX, BREAKWATER_LON_IDX]
        assert not np.ma.is_masked(bw_wave_height_ft), "Unexpected: Breakwater data contains masked points"

        mon_wave_height_ft = wave_height_ft[offset:, MONASTERY_LAT_IDX, MONASTERY_LON_IDX]
        assert not np.ma.is_masked(mon_wave_height_ft), "Unexpected: Monastery data contains masked points"

        # NOTE: need to erase timezone info for mlpd3 to plot local times correctly
        time0 = analysis_date_pacific.replace(tzinfo=None)
        times = [time0 + datetime.timedelta(hours=hour_i) for hour_i in range(offset, NUM_DATA_POINTS)]

        # plot styles
        if offset == 0:
            labels = ("Breakwater", "Monastery")
            linestyle = "-"
            alpha = 1.0
        else:
            labels = (None, None)
            linestyle = "--"
            alpha = 0.5 ** ((offset / 24) - 1)

        ax.plot(times, bw_wave_height_ft, label=labels[0], color="blue", linestyle=linestyle, alpha=alpha)
        ax.plot(times, mon_wave_height_ft, label=labels[1], color="red", linestyle=linestyle, alpha=alpha)

    ax.set_ylim(0)
    ax.set_ylabel("Significant wave height (ft)")
    ax.yaxis.label.set_fontsize(14)
    ax.xaxis.set_ticks_position("bottom")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import tyro

    setup_logging()
    tyro.cli(main)
