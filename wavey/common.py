import logging
from zoneinfo import ZoneInfo

TZ_UTC = ZoneInfo("UTC")
TZ_PACIFIC = ZoneInfo("America/Los_Angeles")

DATETIME_FORMAT = "%a %b %d %H:%M (Pacific)"
"""Format used when formatting datetimes."""

FEET_PER_METER = 3.28


def setup_logging() -> None:
    """Setup logging. Should only be called once per Python session."""

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)5s] [%(created)f] %(name)s: %(message)s",
    )
