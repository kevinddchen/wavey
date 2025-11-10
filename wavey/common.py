from zoneinfo import ZoneInfo

TZ_UTC = ZoneInfo("UTC")
TZ_PACIFIC = ZoneInfo("America/Los_Angeles")

DATETIME_FORMAT = "%a %b %d %H:%M (Pacific)"
"""Format used when formatting datetimes."""

FEET_PER_METER = 3.28

# Lat/lon bounding box for zoom-in on Monterey peninsula
LAT_MIN = 36.4
LAT_MAX = 36.7
LON_MIN = 237.9
LON_MAX = 238.2
