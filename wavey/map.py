import logging
from typing import Iterable, Literal, NamedTuple

import matplotlib
import matplotlib.axes
import matplotlib.colors
import matplotlib.image
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

LOG = logging.getLogger(__name__)

DEFAULT_ARROW_LENGTH = 0.01
"""Length of arrows indicating wave direction, in degrees lat/lon."""

MAX_WAVE_HEIGHT_FT = 12.0
"""The maximum value in the wave height colormap."""


class _Arrow(NamedTuple):
    """Container for data associated to an arrow on the figure."""

    arrow: matplotlib.patches.FancyArrow
    """matplotlib `FancyArrow` instance."""
    length: float
    """Length of arrow."""
    lat: float
    """Latitude of arrow center."""
    lon: float
    """Longitude of arrow center."""
    lat_idx: int
    """Index of arrow's latitude."""
    lon_idx: int
    """Index of arrow's longitude."""


def _draw_arrows(
    headings_rad: np.ma.MaskedArray,
    lats: np.ndarray,
    lons: np.ndarray,
    length: float = DEFAULT_ARROW_LENGTH,
    margin: int = 0,
    stride: int = 3,
) -> list[_Arrow]:
    """
    Draw the initial arrows.

    Args:
        headings_rad: Arrow headings, with shape (LATS, LONS). May contain
            missing values.
        lats: Arrow center latitudes. Array with shape (LATS, LONS).
        lons: Arrow center longitudes. Array with shape (LATS, LONS).
        length: Length of arrows.
        margin: Omit the first and last `margin` indices.
        stride: Create an arrow for every multiple of `stride` indicies.

    Returns:
        List of `Arrow` instances.
    """

    arrows: list[_Arrow] = []

    for lat_idx in range(margin, len(lats[..., 0]) - margin, stride):
        for lon_idx in range(margin, len(lons[0]) - margin, stride):
            heading_rad = headings_rad[lat_idx, lon_idx]
            if np.ma.is_masked(heading_rad):
                continue

            unit_vec = np.array((np.sin(heading_rad), np.cos(heading_rad)))
            dir_vec = unit_vec * length

            lat = lats[lat_idx, 0]
            lon = lons[0, lon_idx]

            arrow_center = np.array((lon, lat))
            arrow_start = arrow_center - 0.5 * dir_vec

            arrow = plt.arrow(
                *arrow_start,
                *dir_vec,
                color="black",
                width=0.0,
                head_width=length / 3,
                length_includes_head=True,
            )
            arrows.append(
                _Arrow(
                    arrow=arrow,
                    length=length,
                    lat=lat,
                    lon=lon,
                    lat_idx=lat_idx,
                    lon_idx=lon_idx,
                )
            )

    return arrows


def _update_arrows(
    headings_rad: np.ma.MaskedArray,
    arrows: Iterable[_Arrow],
) -> None:
    """
    Update the arrow directions.

    Args:
        headings_rad: New arrow headings, with shape (LATS, LONS). May contain
            missing values.
        arrows: List of `Arrow` instances to update.
    """

    for arrow in arrows:
        heading_rad = headings_rad[arrow.lat_idx, arrow.lon_idx]
        if np.ma.is_masked(heading_rad):
            LOG.warning(f"Could not update arrow at {arrow.lat, arrow.lon}")
            continue

        unit_vec = np.array((np.sin(heading_rad), np.cos(heading_rad)))
        dir_vec = unit_vec * arrow.length

        arrow_center = np.array((arrow.lon, arrow.lat))
        arrow_start = arrow_center - 0.5 * dir_vec

        arrow.arrow.set_data(x=arrow_start[0], y=arrow_start[1], dx=dir_vec[0], dy=dir_vec[1])


RESOLUTION = Literal["c", "l", "i", "h", "f"]


class Map:
    wave_height_ft: np.ma.MaskedArray | None
    wave_direction_rad: np.ma.MaskedArray | None
    map: Basemap
    img: matplotlib.image.AxesImage | None
    arrows: list[_Arrow]

    def __init__(
        self,
        ax: matplotlib.axes.Axes,
        wave_height_ft: np.ma.MaskedArray | None,
        wave_direction_rad: np.ma.MaskedArray | None,
        lats: np.ndarray,
        lons: np.ndarray,
        lat_min_idx: int,
        lat_max_idx: int,
        lon_min_idx: int,
        lon_max_idx: int,
        resolution: RESOLUTION = "h",
        draw_arrows_length: float = DEFAULT_ARROW_LENGTH,
        draw_arrows_stride: int = 3,
        water_color: str | None = "lightcyan",
        land_color: str | None = "wheat",
    ) -> None:
        """
        Map plotting wave height as a colormap and wave direction with arrows.

        Args:
            ax: matplotlib axes.
            wave_height_ft: If provided, will create wave height colormap.
                Should be the full (NUM_DATA_POINTS, NUM_LATS, NUM_LONS) array.
            wave_direction_rad: If provided, will draw wave direction with
                arrows. Should be the full (NUM_DATA_POINTS, NUM_LATS, NUM_LONS)
                array.
            lats: Full (NUM_LATS, NUM_LONS) array of latitudes.
            lons: Full (NUM_LATS, NUM_LONS) array of longitudes.
            lat_min_idx: For zooming-in to a smaller lat/lon bounding box.
            lat_max_idx: For zooming-in to a smaller lat/lon bounding box.
            lon_min_idx: For zooming-in to a smaller lat/lon bounding box.
            lon_max_idx: For zooming-in to a smaller lat/lon bounding box.
            resolution: Resolution of the coastline map. Options are crude,
                low, intermediate, high, and full.
            draw_arrows_length: Length of arrows.
            draw_arrows_stride: Create an arrow for every multiple of `stride`
                indicies.
            water_color: Color of water.
            land_color: Color of land.
        """

        # NOTE: not sure why this offset is needed, but looks better with it
        lat_min = lats[lat_min_idx - 1, 0]
        lat_max = lats[lat_max_idx, 0]
        lon_min = lons[0, lon_min_idx - 1]
        lon_max = lons[0, lon_max_idx]

        self.wave_height_ft = (
            wave_height_ft[..., lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx]
            if wave_height_ft is not None
            else None
        )
        self.arrow_heading_rad = (
            # NOTE: add 180 degrees because "wind direction" is where the wind comes from
            wave_direction_rad[..., lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx] + np.pi
            if wave_direction_rad is not None
            else None
        )
        lats = lats[lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx]
        lons = lons[lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx]

        self.map = Basemap(
            projection="cyl",
            llcrnrlat=lat_min,
            llcrnrlon=lon_min,
            urcrnrlat=lat_max,
            urcrnrlon=lon_max,
            resolution=resolution,
            ax=ax,
        )
        self.map.drawcoastlines()

        if water_color is not None:
            self.map.drawmapboundary(fill_color=water_color)
        if land_color is not None:
            self.map.fillcontinents(color=land_color)

        if self.wave_height_ft is not None:
            self.img = self.map.imshow(
                self.wave_height_ft[0],
                cmap="jet",
                norm=matplotlib.colors.Normalize(vmin=0, vmax=MAX_WAVE_HEIGHT_FT),
            )
        else:
            self.img = None

        if self.arrow_heading_rad is not None:
            self.arrows = _draw_arrows(
                self.arrow_heading_rad[0],
                lats=lats,
                lons=lons,
                length=draw_arrows_length,
                stride=draw_arrows_stride,
            )
        else:
            self.arrows = []

    def update(self, hour_i: int) -> None:
        """Update map to the given hour (index)."""

        if self.wave_height_ft is not None and self.img is not None:
            self.img.set_data(self.wave_height_ft[hour_i])

        if self.arrow_heading_rad is not None:
            _update_arrows(self.arrow_heading_rad[hour_i], self.arrows)
