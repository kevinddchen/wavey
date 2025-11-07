import datetime
import logging
import os
import zoneinfo
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pygrib
import requests
from mpl_toolkits.basemap import Basemap
from tqdm import tqdm

from wavey.nwfs import get_most_recent_forecast

# Force non-interactive backend to keep consistency between local and github actions
matplotlib.rcParams["backend"] = "agg"

LOG = logging.getLogger(__name__)

FEET_PER_METER = 3.28

NUM_FORECASTS = 145  # 1 + 24 * 6 hours
"""Number of forecasts for each data type in the NWFS GRIB file."""

# Lat/lon bounding box for zoom-in on Monterey peninsula
LAT_MIN = 36.4  # 36.2
LAT_MAX = 36.7  # 37.0
LON_MIN = 237.9  # 237.8
LON_MAX = 238.2  # 238.3

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


class DataType(IntEnum):
    """Data types in the NWFS GRIB file, in order."""

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
    data: np.ma.MaskedArray
    """Array with shape (NUM_FORECASTS, LATS, LONS). May contain missing values."""
    lats: np.ndarray
    """Array with shape (LATS, LONS)."""
    lons: np.ndarray
    """Array with shape (LATS, LONS)."""
    analysis_date_utc: datetime.datetime
    """Date and time of analysis, i.e. start of forecast, in UTC."""


def download_most_recent_forecast_data(dir: Path) -> Path:
    """
    Downloads the most recent NWFS GRIB file and returns the path.

    Args:
        dir: Directory to save the file in.

    Returns:
        Path to the GRIB file.
    """

    url = get_most_recent_forecast()

    file_path = dir / os.path.basename(url)
    if file_path.exists():
        LOG.info(f"'{file_path}' already exists. Skipping download")
        return file_path

    LOG.info(f"Downloading '{url}' to '{file_path}'")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(file_path, "wb") as file:
        for chunk in r.iter_content(chunk_size=8192):
            file.write(chunk)

    return file_path


def read_forecast_data(grbs: pygrib.open, data_type: DataType) -> ForecastData:
    """
    Read forecast data from Monterey Bay NWFS GRIB file, zoomed-in near the
    peninsula (see {LAT|LON}_{MIN|MAX} values).

    Args:
        grbs: GRIB file.
        data_type: Type of data to read.

    Returns:
        Forecast data of the specified type.
    """

    grbs.seek(data_type * NUM_FORECASTS)  # message offset

    data_list: list[np.ma.MaskedArray] = []
    lats: np.ndarray | None = None
    lons: np.ndarray | None = None
    analysis_date: datetime.datetime | None = None

    for grb in grbs.read(NUM_FORECASTS):
        data, lats, lons = grb.data(lat1=LAT_MIN, lat2=LAT_MAX, lon1=LON_MIN, lon2=LON_MAX)
        data_list.append(data)

        if analysis_date is None:
            analysis_date = grb.analDate

    # assertions will fail if no messages were read
    assert lats is not None
    assert lons is not None
    assert analysis_date is not None
    analysis_date_utc = analysis_date.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
    data_collated = np.ma.stack(data_list)

    return ForecastData(
        data=data_collated,
        lats=lats,
        lons=lons,
        analysis_date_utc=analysis_date_utc,
    )


def utc_to_pt(dt: datetime.datetime) -> datetime.datetime:
    """Convert UTC to pacific time."""

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        assert dt.utcoffset() == datetime.timedelta(), "datetime is not UTC"

    pdt_tz = zoneinfo.ZoneInfo("America/Los_Angeles")
    return dt.astimezone(tz=pdt_tz)


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
            will download the most recent one.
        out_dir: Path to output directory.
    """

    LOG.info(f"Saving output assets to '{out_dir}'")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download data, if needed

    if grib_path is None:
        dir = Path(".")  # download to current directory
        grib_path = download_most_recent_forecast_data(dir)

    # Extract data

    LOG.info(f"Reading '{grib_path}'")
    with pygrib.open(grib_path) as grbs:
        wave_height_forecast = read_forecast_data(grbs, DataType.WaveHeight)
        wave_direction_forecast = read_forecast_data(grbs, DataType.WaveDirection)

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
    x = [x0 + datetime.timedelta(hours=hour_i) for hour_i in range(NUM_FORECASTS)]
    for label, y in (("Breakwater", bw_wave_heights_ft), ("Monastery", mon_wave_heights_ft)):
        ax.plot(x, y, label=label)

    ax.set_ylim(0)
    ax.set_ylabel("Significant wave height (ft)")
    ax.set_xlabel("Pacific Time")
    ax.legend(loc="upper right")
    ax.grid(linestyle=":")

    plt.tight_layout()
    div_str = mpld3.fig_to_html(fig, figid="graph")
    template_html = Path("template.html").read_text()
    out_html = template_html.replace("<!-- insert swell graph here -->", div_str)
    (out_dir / "index.html").write_text(out_html)

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
    for hour_i in tqdm(range(NUM_FORECASTS)):
        pacific_time = analysis_date_pacific + datetime.timedelta(hours=hour_i)
        pacific_time_str = pacific_time.strftime("%a %b %d %H:%M")

        img.set_data(wave_height_ft[hour_i])
        update_arrows(arrow_heading_rad[hour_i], lats=lats, lons=lons, arrows=arrows, latlon_idxs=arrow_latlon_idxs)

        plt.title(f"Significant wave height (ft) and primary wave direction\nHour {hour_i:03} ({pacific_time_str} PT)")
        plt.savefig(plot_dir / f"{hour_i}.png")


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, format="[%(levelname)5s] [%(created)f] %(name)s: %(message)s")
    tyro.cli(main)
