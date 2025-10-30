import datetime
import logging
import zoneinfo
from enum import IntEnum
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pygrib
from mpl_toolkits.basemap import Basemap
from tqdm import tqdm

LOG = logging.getLogger(__name__)

FEET_PER_METER = 3.28

LAT_MIN = 36.4  # 36.2
LAT_MAX = 36.7  # 37.0
LON_MIN = 237.9  # 237.8
LON_MAX = 238.2  # 238.3

BREAKWATER_LAT = 36.61
BREAKWATER_LON = 238.105

NUM_MESSAGES = 145  # 1 + 24 * 6 hours
"""Number of messages for each data type"""


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


def _utc_to_pdt(dt: datetime.datetime) -> datetime.datetime:
    """Convert UTC to PDT."""

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        assert dt.utcoffset == datetime.timedelta(), "datetime is not UTC"

    pdt_tz = zoneinfo.ZoneInfo("America/Los_Angeles")
    return dt.astimezone(tz=pdt_tz)


def main(
    grib_path: Path,
    /,
    out_dir: Path = Path("out"),
) -> None:
    """
    Create plots for significant wave height.

    Args:
        grib_path: Path to GRIB file. These are downloaded from:
            https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwps/prod/
        out_dir: Path to output directory.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract data

    utc_start_time: datetime.datetime | None = None
    wave_heights_m_list: list[np.ma.MaskedArray] = []
    lats: np.ndarray | None = None
    lons: np.ndarray | None = None

    LOG.info(f"Loading GRIB file {grib_path}...")
    with pygrib.open(grib_path) as grbs:
        grbs.seek(DataType.WaveHeight * NUM_MESSAGES)

        # pick wave height
        for grb in grbs.read(NUM_MESSAGES):
            wave_height_m, lats, lons = grb.data(
                lat1=LAT_MIN, lat2=LAT_MAX, lon1=LON_MIN, lon2=LON_MAX
            )
            wave_heights_m_list.append(wave_height_m)

            if utc_start_time is None:
                utc_start_time = grb.analDate

    assert utc_start_time is not None, "Unexpected: no messages were read"
    assert lats is not None
    assert lons is not None

    pacific_start_time = _utc_to_pdt(utc_start_time)
    wave_heights_ft = np.ma.stack(wave_heights_m_list) * FEET_PER_METER

    # Get Breakwater data

    bw_lat_idx = lats[..., 0].searchsorted(BREAKWATER_LAT)
    bw_lon_idx = lons[0].searchsorted(BREAKWATER_LON)

    bw_wave_heights_ft = wave_heights_ft[..., bw_lat_idx, bw_lon_idx]
    assert not np.ma.is_masked(bw_wave_heights_ft), (
        "Unexpected: Breakwater data contains masked points"
    )

    # Draw Breakwater graph

    LOG.info("Drawing Breakwater swell graph...")
    fig, ax = plt.subplots(figsize=(6, 2))

    x = list(range(NUM_MESSAGES))
    x_dates = [pacific_start_time + datetime.timedelta(hours=hour_i) for hour_i in x]
    x_ticks = [hour_i for hour_i, dt in zip(x, x_dates, strict=True) if dt.hour == 0]
    x_ticklabels = [x_dates[i].strftime("%a %b %d") for i in x_ticks]
    y = bw_wave_heights_ft

    ax.plot(x, y)
    ax.set_ylim(0)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)

    div_str = mpld3.fig_to_html(fig, figid="graph")
    template_html = Path("template.html").read_text()
    out_html = template_html.replace("<!-- insert the plot here -->", div_str)
    (out_dir / "index.html").write_text(out_html)

    # Draw figure

    LOG.info("Drawing map frames...")
    fig, ax = plt.subplots(figsize=(8, 8))
    m = Basemap(
        projection="cyl",
        llcrnrlat=LAT_MIN,
        llcrnrlon=LON_MIN,
        urcrnrlat=LAT_MAX,
        urcrnrlon=LON_MAX,
        resolution="h",
        ax=ax,
    )
    m.drawcoastlines()

    img = m.imshow(
        wave_heights_ft[0],
        cmap="jet",
        norm=mcolors.Normalize(vmin=0, vmax=12),
    )
    plt.colorbar(img, orientation="vertical", label="(ft)", shrink=0.8)

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for hour_i in tqdm(range(NUM_MESSAGES)):
        pacific_time = pacific_start_time + datetime.timedelta(hours=hour_i)
        pacific_time_str = pacific_time.strftime("%a %b %d %H:%M")

        img.set_data(wave_heights_ft[hour_i])
        plt.title(f"Hour {hour_i:03} ({pacific_time_str})")
        plt.savefig(plot_dir / f"{hour_i}.png")


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(module)s: %(message)s")
    tyro.cli(main)
