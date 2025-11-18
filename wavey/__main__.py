import datetime
import io
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import PIL.Image
import pygrib
from jinja2 import Environment, PackageLoader, select_autoescape
from tqdm import tqdm

import wavey
from wavey.common import DATETIME_FORMAT, FEET_PER_METER, TZ_PACIFIC, TZ_UTC, setup_logging
from wavey.grib import NUM_DATA_POINTS, ForecastType, read_forecast_data
from wavey.map import DEFAULT_ARROW_LENGTH, RESOLUTION, Map
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


def savefig(path: Path) -> None:
    """
    Save matplotlib figure to PNG file.

    We perform a bit of optimization to make the output filesize smaller
    without sacrificing quality.

    Args:
        path: Path to output PNG file.
    """

    bts = io.BytesIO()
    plt.savefig(bts, format="png")

    with PIL.Image.open(bts) as img:
        img2 = img.convert("RGB").convert("P", palette=PIL.Image.Palette.WEB)
        img2.save(path, format="png")


def main(
    grib_path: Path | None = None,
    /,
    out_dir: Path = Path("_site"),
    resolution: RESOLUTION = "h",
) -> None:
    """
    Create plots for significant wave height.

    Args:
        grib_path: Path to GRIB file. These are downloaded from:
            https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwps/prod/. If none,
            will download the most recent one to the current directory.
        out_dir: Path to output directory.
        resolution: Resolution of the coastline map. Options are crude, low,
            intermediate, high, and full.
    """

    if resolution != "f":
        LOG.warning("Not drawing full resolution coastlines. Use the flag '--resolution f'")

    # Download data, if needed

    if grib_path is None:
        most_recent_forecast = get_most_recent_forecast()
        grib_path = download_forecast(most_recent_forecast)

    # Extract data

    LOG.info(f"Reading '{grib_path}'")
    with pygrib.open(grib_path) as grbs:
        wave_height_forecast = read_forecast_data(grbs, ForecastType.WaveHeight)
        wave_direction_forecast = read_forecast_data(grbs, ForecastType.WaveDirection)
        wave_period_forecast = read_forecast_data(grbs, ForecastType.WavePeriod)
        tide_height_forecast = read_forecast_data(grbs, ForecastType.SeaSurfaceHeight)

    wave_height_ft = wave_height_forecast.data * FEET_PER_METER
    wave_direction_rad = wave_direction_forecast.data * np.pi / 180
    wave_period_sec = wave_period_forecast.data
    tide_height_ft = tide_height_forecast.data * FEET_PER_METER
    lats = wave_height_forecast.lats
    lons = wave_height_forecast.lons
    analysis_date_pacific = utc_to_pt(wave_height_forecast.analysis_date_utc)

    # Get Breakwater data

    bw_wave_height_ft = wave_height_ft[..., BREAKWATER_LAT_IDX, BREAKWATER_LON_IDX]
    bw_wave_period_sec = wave_period_sec[..., BREAKWATER_LAT_IDX, BREAKWATER_LON_IDX]
    assert not np.ma.is_masked(bw_wave_height_ft) and not np.ma.is_masked(bw_wave_period_sec), (
        "Unexpected: Breakwater data contains masked points"
    )

    # Get Monastery data

    mon_wave_height_ft = wave_height_ft[..., MONASTERY_LAT_IDX, MONASTERY_LON_IDX]
    mon_wave_period_sec = wave_period_sec[..., MONASTERY_LAT_IDX, MONASTERY_LON_IDX]
    assert not np.ma.is_masked(mon_wave_height_ft) and not np.ma.is_masked(mon_wave_period_sec), (
        "Unexpected: Monastery data contains masked points"
    )

    # Get tide height. We measure at Breakwater for simplicity.

    tide_height_ft = tide_height_ft[..., BREAKWATER_LAT_IDX, BREAKWATER_LON_IDX]
    assert not np.ma.is_masked(tide_height_ft), "Unexpected: sea level data contains masked points"

    # Plotting graph

    LOG.info("Plotting graph")
    fig, (ax_height, ax_period, ax_tide) = plt.subplots(
        3, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}, dpi=DPI
    )

    # NOTE: need to erase timezone info for mlpd3 to plot local times correctly
    time0 = analysis_date_pacific.replace(tzinfo=None)
    times = [time0 + datetime.timedelta(hours=hour_i) for hour_i in range(NUM_DATA_POINTS)]

    labels = ("Breakwater", "Monastery")
    colors = ("blue", "red")

    for i, y in enumerate((bw_wave_height_ft, mon_wave_height_ft)):
        ax_height.plot(times, y, label=labels[i], color=colors[i])  # type: ignore[arg-type]

    ax_height.set_ylim(0)
    ax_height.set_ylabel("Significant wave height (ft)")
    ax_height.yaxis.label.set_fontsize(14)
    ax_height.xaxis.set_ticks_position("bottom")
    ax_height.legend(loc="upper right")
    ax_height.grid(True, linestyle=":", alpha=0.7)

    for i, y in enumerate((bw_wave_period_sec, mon_wave_period_sec)):
        ax_period.plot(times, y, label=labels[i], color=colors[i])  # type: ignore[arg-type]

    ax_period.set_ylabel("Primary wave period (sec)")
    ax_period.yaxis.label.set_fontsize(14)
    ax_period.xaxis.set_ticks_position("bottom")
    ax_period.legend(loc="upper right")
    ax_period.grid(True, linestyle=":", alpha=0.7)

    ax_tide.plot(times, tide_height_ft, color="black")

    ax_tide.set_ylabel("Tide height (ft)")
    ax_tide.yaxis.label.set_fontsize(14)
    ax_tide.set_xlabel("Time (Pacific)")
    ax_tide.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    fig_div = mpld3.fig_to_html(fig)

    # Draw maps

    fig = plt.figure(figsize=(8, 10), dpi=DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

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
        resolution=resolution,
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
        resolution=resolution,
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
        resolution=resolution,
        draw_arrows_length=DEFAULT_ARROW_LENGTH / 3,
        draw_arrows_stride=1,
    )
    ax_mon.set_title("Monastery")

    plt.tight_layout()
    plt.colorbar(map_main.img, orientation="vertical", label="(ft)", shrink=0.8)

    LOG.info("Creating colormap frames")
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
        savefig(plot_dir / f"{hour_i}.png")

    # Get current time and version

    now_utc = datetime.datetime.now(tz=TZ_UTC)
    now_pacific = now_utc.astimezone(tz=TZ_PACIFIC)
    now_pacific_str = now_pacific.strftime(DATETIME_FORMAT)

    version = wavey.__version__

    # Export HTML

    out_html_path = out_dir / "index.html"
    LOG.info(f"Saving webpage to '{out_html_path}'")
    env = Environment(loader=PackageLoader("wavey"), autoescape=select_autoescape())
    template_html = env.get_template("index.html.j2")
    out_html = template_html.render(
        swell_graph=fig_div,
        last_updated=now_pacific_str,
        version=version,
    )
    out_html_path.write_text(out_html)


if __name__ == "__main__":
    import tyro

    setup_logging()
    tyro.cli(main)
