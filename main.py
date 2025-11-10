import datetime
import logging
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pygrib
from jinja2 import Environment, PackageLoader, select_autoescape
from mpl_toolkits.basemap import Basemap
from tqdm import tqdm

from wavey.common import DATETIME_FORMAT, FEET_PER_METER, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, TZ_PACIFIC, TZ_UTC
from wavey.grib import NUM_DATA_POINTS, ForecastType, read_forecast_data
from wavey.nwfs import download_forecast, get_most_recent_forecast

# Force non-interactive backend to keep consistency between local and github actions
matplotlib.rcParams["backend"] = "agg"

LOG = logging.getLogger(__name__)

# Location of San Carlos Beach (aka Breakwater)
BREAKWATER_LAT = 36.611
BREAKWATER_LON = 238.108

# Location of San Carlos Beach (aka Breakwater)
MONASTERY_LAT = 36.525
MONASTERY_LON = 238.069

MAX_WAVE_HEIGHT_FT = 12.0
"""The maximum value in the wave height colormap."""

WAVE_DIRECTION_ARROW_SIZE = 0.01
"""Size of arrows indicating wave direction, in degrees lat/lon."""

DPI = 100
"""Matplotlib figure dpi."""


def utc_to_pt(dt: datetime.datetime) -> datetime.datetime:
    """Convert UTC to pacific time."""

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ_UTC)
    else:
        assert dt.utcoffset() == datetime.timedelta(), "datetime is not UTC"

    return dt.astimezone(tz=TZ_PACIFIC)


def draw_arrows(
    headings_rad: np.ma.MaskedArray,
    lats: np.ndarray,
    lons: np.ndarray,
    margin: int = 1,
    stride: int = 3,
) -> tuple[list[mpatches.FancyArrow], list[tuple[int, int]]]:
    """
    Draw the initial arrows.

    Args:
        headings_rad: Arrow headings, with shape (LATS, LONS). May contain
            missing values.
        lats: Arrow center latitudes. Array with shape (LATS, LONS).
        lons: Arrow center longitudes. Array with shape (LATS, LONS).
        margin: Omit the first and last `margin` indices.
        stride: Create an arrow for every multiple of `stride` indicies.

    Returns:
        Tuple of `matplotlib.patches.FancyArrow` instances and the
        corresponding lat/lon indices.
    """

    arrows: list[mpatches.FancyArrow] = []
    latlon_idxs: list[tuple[int, int]] = []

    for lat_idx in range(margin, len(lats[..., 0]) - margin, stride):
        for lon_idx in range(margin, len(lons[0]) - margin, stride):
            heading_rad = headings_rad[lat_idx, lon_idx]
            if np.ma.is_masked(heading_rad):
                continue

            unit_vec = np.array((np.sin(heading_rad), np.cos(heading_rad)))
            dir_vec = unit_vec * WAVE_DIRECTION_ARROW_SIZE

            lat = lats[lat_idx, 0]
            lon = lons[0, lon_idx]

            arrow_center = np.array((lon, lat))
            arrow_start = arrow_center - 0.5 * dir_vec

            arrow = plt.arrow(
                *arrow_start,
                *dir_vec,
                color="black",
                width=0.0,
                head_width=0.003,
                length_includes_head=True,
            )
            arrows.append(arrow)
            latlon_idxs.append((lat_idx, lon_idx))

    return arrows, latlon_idxs


def update_arrows(
    headings_rad: np.ma.MaskedArray,
    lats: np.ndarray,
    lons: np.ndarray,
    arrows: list[mpatches.FancyArrow],
    latlon_idxs: list[tuple[int, int]],
) -> None:
    """
    Update the arrow directions.

    Args:
        headings_rad: Arrow headings, with shape (LATS, LONS). May contain
            missing values.
        lats: Arrow center latitudes. Array with shape (LATS, LONS).
        lons: Arrow center longitudes. Array with shape (LATS, LONS).
        arrows: The `matplotlib.patches.FancyArrow` instances.
        latlon_idxs: The lat/lon indices of the arrows.
    """

    for arrow, latlon_idx in zip(arrows, latlon_idxs, strict=True):
        lat_idx, lon_idx = latlon_idx

        heading_rad = headings_rad[lat_idx, lon_idx]
        assert not np.ma.is_masked(heading_rad)

        unit_vec = np.array((np.sin(heading_rad), np.cos(heading_rad)))
        dir_vec = unit_vec * WAVE_DIRECTION_ARROW_SIZE

        lat = lats[lat_idx, 0]
        lon = lons[0, lon_idx]

        arrow_center = np.array((lon, lat))
        arrow_start = arrow_center - 0.5 * dir_vec

        arrow.set_data(x=arrow_start[0], y=arrow_start[1], dx=dir_vec[0], dy=dir_vec[1])


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

    bw_lat_idx = lats[..., 0].searchsorted(BREAKWATER_LAT)
    bw_lon_idx = lons[0].searchsorted(BREAKWATER_LON)

    bw_wave_heights_ft = wave_height_ft[..., bw_lat_idx, bw_lon_idx]
    assert not np.ma.is_masked(bw_wave_heights_ft), "Unexpected: Breakwater data contains masked points"

    # Get Monastery data

    mon_lat_idx = lats[..., 0].searchsorted(MONASTERY_LAT)
    mon_lon_idx = lons[0].searchsorted(MONASTERY_LON)

    mon_wave_heights_ft = wave_height_ft[..., mon_lat_idx, mon_lon_idx]
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

    LOG.info("Drawing map frames")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=DPI)
    map = Basemap(
        projection="cyl",
        llcrnrlat=LAT_MIN,
        llcrnrlon=LON_MIN,
        urcrnrlat=LAT_MAX,
        urcrnrlon=LON_MAX,
        resolution="h",
        ax=ax,
    )
    map.drawcoastlines()

    img = map.imshow(
        wave_height_ft[0],
        cmap="jet",
        norm=mcolors.Normalize(vmin=0, vmax=MAX_WAVE_HEIGHT_FT),
    )
    plt.colorbar(img, orientation="vertical", label="(ft)", shrink=0.8)

    # NOTE: add 180 degrees because "wind direction" is where the wind comes from
    arrow_heading_rad = wave_direction_rad + np.pi
    arrows, arrow_latlon_idxs = draw_arrows(arrow_heading_rad[0], lats=lats, lons=lons)

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for hour_i in tqdm(range(NUM_DATA_POINTS)):
        pacific_time = analysis_date_pacific + datetime.timedelta(hours=hour_i)
        pacific_time_str = pacific_time.strftime(DATETIME_FORMAT)

        img.set_data(wave_height_ft[hour_i])
        update_arrows(arrow_heading_rad[hour_i], lats=lats, lons=lons, arrows=arrows, latlon_idxs=arrow_latlon_idxs)

        plt.title(f"Significant wave height (ft) and primary wave direction\nHour {hour_i:03} -- {pacific_time_str}")
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

    logging.basicConfig(level=logging.INFO, format="[%(levelname)5s] [%(created)f] %(name)s: %(message)s")
    tyro.cli(main)
