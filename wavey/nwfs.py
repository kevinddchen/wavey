import logging
import os
import re

import requests
from bs4 import BeautifulSoup

LOG = logging.getLogger(__name__)

_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwps/prod"
_MTR = "mtr"
_CG3 = "CG3"


def _list_dates() -> list[str]:
    """
    List dates with completed runs for the Western Region (wr).

    Returns:
        List of strings like "wr.YYYYMMDD/", sorted in descending order (i.e.
        most recent first).
    """

    url = _BASE_URL

    r = requests.get(url)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    hrefs = [link["href"] for link in soup.find_all("a", href=True)]

    # find all hrefs that look like "wr.YYYYMMDD"
    dates = filter(lambda x: re.match(r"wr\.\d{8}", x), hrefs)
    return sorted(dates, reverse=True)


def _list_times(date: str) -> list[str]:
    """
    List times with completed runs for Monterey bay on the given date.

    Args:
        date: A string like "wr.YYYYMMDD/".

    Returns:
        List of strings like "HH/", sorted in descending order (i.e. most
        recent first).

    Raises:
        HTTPError: If no completed runs for the given date.
    """

    url = os.path.join(_BASE_URL, date, _MTR)

    r = requests.get(url)
    r.raise_for_status()  # if 404, means no runs have been completed

    soup = BeautifulSoup(r.text, "html.parser")
    hrefs = [link["href"] for link in soup.find_all("a", href=True)]

    # find all links that look like "HH"
    times = filter(lambda x: re.match(r"\d{2}", x), hrefs)
    return sorted(times, reverse=True)


def get_most_recent_forecast() -> str:
    """
    Get most recent NWFS forecast data for Monterey bay.

    Returns:
        URL to a GRIB file.

    Raises:
        HTTPError: If forecast data could not be found.
    """

    dates = _list_dates()
    LOG.info(f"Found NWFS Western Region forecasts: {dates}")

    most_recent_date: str | None = None
    most_recent_time: str | None = None
    for date in dates:
        LOG.info(f"Looking in {date}...")
        try:
            times = _list_times(date)
        except requests.HTTPError:
            # no completed runs on that date
            LOG.info(f"No Monterey bay forecasts in {date}")
            continue

        if len(times) > 0:
            most_recent_date = date
            most_recent_time = times[0]
            break

    if most_recent_date is None or most_recent_time is None:
        raise requests.HTTPError("Could not find most recent forecast.")

    # parse date and time
    date_match = re.search(r"\d{8}", most_recent_date)
    assert date_match, f"Unexpected date: {most_recent_date}"
    time_match = re.search(r"\d{2}", most_recent_time)
    assert time_match, f"Unexpected time: {most_recent_time}"

    yyyymmdd = date_match.group(0)
    hh = time_match.group(0)
    filename = f"{_MTR}_nwps_{_CG3}_{yyyymmdd}_00{hh}.grib2"

    url = os.path.join(_BASE_URL, most_recent_date, _MTR, most_recent_time, _CG3, filename)
    LOG.info(f"Most recent forecast: {url}")
    return url


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(module)s: %(message)s")

    get_most_recent_forecast()
