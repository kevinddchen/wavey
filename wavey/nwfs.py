import logging
import os
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from wavey.common import setup_logging

LOG = logging.getLogger(__name__)

_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwps/prod"
_MTR = "mtr"
_CG3 = "CG3"

_CHUNK_SIZE = 8192


def get_most_recent_forecast() -> str:
    """
    Get most recent NWFS forecast data for Monterey bay.

    Returns:
        URL to the GRIB file.
    """

    # 1. List dates with forecasts
    dates = _list_dates()
    LOG.info(f"Found NWFS forecasts: {dates}")

    most_recent_date: str | None = None
    most_recent_time: str | None = None

    # 2. For each date, look for Monterey bay forecasts
    for date in dates:
        LOG.info(f"Looking in '{date}'...")
        try:
            times = _list_times(date)
        except requests.HTTPError:
            continue

        # 3. For each time, check if "CG3" forecast is available
        for time in times:
            if _check_time(date=date, time=time):
                most_recent_date = date
                most_recent_time = times[0]
                break

        # propagate break out of for-loop above
        if most_recent_date is not None:
            break

    assert most_recent_date is not None and most_recent_time is not None, (
        "Unexpected: could not find any forecasts for Monterey bay."
    )

    # parse date and time
    date_match = re.search(r"\d{8}", most_recent_date)
    assert date_match, f"Unexpected date: {most_recent_date}"
    time_match = re.search(r"\d{2}", most_recent_time)
    assert time_match, f"Unexpected time: {most_recent_time}"

    yyyymmdd = date_match.group(0)
    hh = time_match.group(0)
    filename = f"{_MTR}_nwps_{_CG3}_{yyyymmdd}_{hh}00.grib2"

    url = os.path.join(_BASE_URL, most_recent_date, _MTR, most_recent_time, _CG3, filename)
    LOG.info(f"Found most recent forecast: {url}")
    return url


def _get_hrefs(url: str, regex: str | None = None) -> list[str]:
    """
    Navigate to URL and return all hrefs on the webpage.

    Args:
        url: URL of webpage.
        regex: If provided, will only return matching hrefs.

    Returns:
        List of strings.

    Raises:
        HTTPError: If accessing URL returns error.
    """

    r = requests.get(url)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    hrefs: list[str] = [link["href"] for link in soup.find_all("a", href=True)]  # type: ignore[misc]

    if regex is not None:
        hrefs = list(filter(lambda x: re.match(regex, x), hrefs))

    return hrefs


def _list_dates() -> list[str]:
    """
    List dates with Western Region (wr) forecasts.

    Returns:
        List of strings like "wr.YYYYMMDD/"; sorted (most recent first).
    """

    url = _BASE_URL
    dates = _get_hrefs(url, r"wr\.\d{8}")  # hrefs look like "wr.YYYYMMDD/"
    return sorted(dates, reverse=True)


def _list_times(date: str) -> list[str]:
    """
    List times with forecasts for Monterey bay on the given date.

    Args:
        date: A string like "wr.YYYYMMDD/".

    Returns:
        List of strings like "HH/"; sorted (most recent first).

    Raises:
        HTTPError: If no forecasts for the given date.
    """

    url = os.path.join(_BASE_URL, date, _MTR)
    times = _get_hrefs(url, r"\d{2}")  # hrefs look like "HH/"
    return sorted(times, reverse=True)


def _check_time(date: str, time: str) -> bool:
    """
    Check if "CG3" forecast is available for the given time.

    Args:
        date: A string like "wr.YYYYMMDD/".
        time: A string like "HH/".

    Returns:
        True if a "CG3" forecast is available, else False.
    """

    url = os.path.join(_BASE_URL, date, _MTR, time, _CG3)
    r = requests.get(url)
    return r.ok


def download_forecast(url: str, dir: Path | None = None) -> Path:
    """
    Download NWFS forecast data to disk.

    Args:
        url: URL to the GRIB file.
        dir: Directory to save the file in. If none, will download to the
            current directory.

    Returns:
        Path to the GRIB file.
    """

    if dir is None:
        dir = Path(".")

    file_path = dir / os.path.basename(url)
    if file_path.exists():
        LOG.info(f"'{file_path}' already exists. Skipping download")
        return file_path

    LOG.info(f"Downloading '{url}' to '{file_path}'")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(file_path, "wb") as file:
        for chunk in r.iter_content(chunk_size=_CHUNK_SIZE):
            file.write(chunk)

    return file_path


if __name__ == "__main__":
    setup_logging()
    get_most_recent_forecast()
