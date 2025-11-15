import datetime
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pygrib
from jinja2 import Environment, PackageLoader, select_autoescape
from tqdm import tqdm

from wavey.common import DATETIME_FORMAT, FEET_PER_METER, TZ_PACIFIC, TZ_UTC, setup_logging
from wavey.grib import NUM_DATA_POINTS, ForecastType, read_forecast_data
from wavey.map import DEFAULT_ARROW_LENGTH, Map
from wavey.nwfs import download_forecast, get_most_recent_forecast

# Force non-interactive backend to keep consistency between local and github actions
matplotlib.rcParams["backend"] = "agg"

LOG = logging.getLogger(__name__)

# Location of San Carlos Beach (aka Breakwater)
BREAKWATER_LAT_IDX = 91  # 36.61132
BREAKWATER_LON_IDX = 55  # 238.10899

# Location of Monastery Beach
MONASTERY_LAT_IDX = 72  # 36.52544
MONASTERY_LON_IDX = 48  # 238.06966

DPI = 100
"""Matplotlib figure dpi."""


def utc_to_pt(dt: datetime.datetime) -> datetime.datetime:
    """Convert UTC to pacific time."""

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ_UTC)
    else:
        assert dt.utcoffset() == datetime.timedelta(), "datetime is not UTC"

    return dt.astimezone(tz=TZ_PACIFIC)


def main(
    grib_path: Path | None = None,
    /,
    out_dir: Path = Path("_site"),
) -> None:
    """
    Create plots for significant wave height.

    Args:
        grib_path: Path to GRIB file. These are downloaded from:
            https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwps/prod/. If none,
            will download the most recent one to the current directory.
        out_dir: Path to output directory.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Download data, if needed

    if grib_path is None:
        most_recent_forecast = get_most_recent_forecast()
        grib_path = download_forecast(most_recent_forecast)

    # Extract data

    LOG.info(f"Reading '{grib_path}'")
    with pygrib.open(grib_path) as grbs:
        wave_height_forecast = read_forecast_data(grbs, ForecastType.WaveHeight)
        wave_direction_forecast = read_forecast_data(grbs, ForecastType.WaveDirection)

    wave_height_ft = wave_height_forecast.data * FEET_PER_METER
    wave_direction_rad = wave_direction_forecast.data * np.pi / 180
    lats = wave_height_forecast.lats
    lons = wave_height_forecast.lons
    analysis_date_pacific = utc_to_pt(wave_height_forecast.analysis_date_utc)

    # Get Breakwater data

    bw_wave_heights_ft = wave_height_ft[..., BREAKWATER_LAT_IDX, BREAKWATER_LON_IDX]
    assert not np.ma.is_masked(bw_wave_heights_ft), "Unexpected: Breakwater data contains masked points"

    # Get Monastery data

    mon_wave_heights_ft = wave_height_ft[..., MONASTERY_LAT_IDX, MONASTERY_LON_IDX]
    assert not np.ma.is_masked(mon_wave_heights_ft), "Unexpected: Monastery data contains masked points"

    # Draw Breakwater graph

    LOG.info("Drawing swell graph")
    fig, ax = plt.subplots(figsize=(9, 3), dpi=DPI)

    # NOTE: need to erase timezone info for mlpd3 to plot local times correctly
    x0 = analysis_date_pacific.replace(tzinfo=None)
    x = [x0 + datetime.timedelta(hours=hour_i) for hour_i in range(NUM_DATA_POINTS)]
    for label, y in (("Breakwater", bw_wave_heights_ft), ("Monastery", mon_wave_heights_ft)):
        ax.plot(x, y, label=label)  # type: ignore[arg-type]

    ax.set_ylim(0)
    ax.set_ylabel("Significant wave height (ft)")
    ax.set_xlabel("Time (Pacific)")
    ax.legend(loc="upper right")
    ax.grid(linestyle=":")

    plt.tight_layout()
    fig_div = mpld3.fig_to_html(fig)

    # Draw figure

    fig = plt.figure(figsize=(8, 10), dpi=DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    LOG.info("Drawing Monterey bay map")
    ax_main = fig.add_subplot(gs[0, :])
    map_main = Map(
        ax=ax_main,
        wave_height_ft=wave_height_ft,
        wave_direction_rad=wave_direction_rad,
        lats=lats,
        lons=lons,
        lat_min_idx=60,
        lat_max_idx=110,
        lon_min_idx=20,
        lon_max_idx=70,
    )

    LOG.info("Drawing Breakwater map")
    ax_bw = fig.add_subplot(gs[1, 0])
    map_bw = Map(
        ax=ax_bw,
        wave_height_ft=None,
        wave_direction_rad=wave_direction_rad,
        lats=lats,
        lons=lons,
        lat_min_idx=BREAKWATER_LAT_IDX - 2,
        lat_max_idx=BREAKWATER_LAT_IDX + 3,
        lon_min_idx=BREAKWATER_LON_IDX - 2,
        lon_max_idx=BREAKWATER_LON_IDX + 3,
        draw_arrows_length=DEFAULT_ARROW_LENGTH / 3,
        draw_arrows_stride=1,
    )
    ax_bw.set_title("Breakwater")

    LOG.info("Drawing Monastery map")
    ax_mon = fig.add_subplot(gs[1, 1])
    map_mon = Map(
        ax=ax_mon,
        wave_height_ft=None,
        wave_direction_rad=wave_direction_rad,
        lats=lats,
        lons=lons,
        lat_min_idx=MONASTERY_LAT_IDX - 1,
        lat_max_idx=MONASTERY_LAT_IDX + 4,
        lon_min_idx=MONASTERY_LON_IDX - 3,
        lon_max_idx=MONASTERY_LON_IDX + 2,
        draw_arrows_length=DEFAULT_ARROW_LENGTH / 3,
        draw_arrows_stride=1,
    )
    ax_mon.set_title("Monastery")

    plt.colorbar(map_main.img, orientation="vertical", label="(ft)")

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for hour_i in tqdm(range(NUM_DATA_POINTS)):
        pacific_time = analysis_date_pacific + datetime.timedelta(hours=hour_i)
        pacific_time_str = pacific_time.strftime(DATETIME_FORMAT)

        map_main.update(hour_i)
        map_bw.update(hour_i)
        map_mon.update(hour_i)

        ax_main.set_title(
            f"Significant wave height (ft) and primary wave direction\nHour {hour_i:03} -- {pacific_time_str}"
        )
        plt.savefig(plot_dir / f"{hour_i}.png")

    # Get current time

    now_utc = datetime.datetime.now(tz=TZ_UTC)
    now_pacific = now_utc.astimezone(tz=TZ_PACIFIC)
    now_pacific_str = now_pacific.strftime(DATETIME_FORMAT)

    # Export HTML

    LOG.info(f"Saving webpage to '{out_dir}'")
    env = Environment(loader=PackageLoader("wavey"), autoescape=select_autoescape())
    template_html = env.get_template("index.html.j2")
    out_html = template_html.render(
        swell_graph=fig_div,
        last_updated=now_pacific_str,
    )
    (out_dir / "index.html").write_text(out_html)


if __name__ == "__main__":
    import tyro

    setup_logging()
    tyro.cli(main)
