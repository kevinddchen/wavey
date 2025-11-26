# Wavey

[_--> live website <--_](https://kevinddchen.github.io/wavey/)

## Manifesto

**The goal of this project is to combine the accuracy of NWPS with the user interface of Windy.**

 -  The [Nearshore Wave Prediction System (NWPS)](https://polar.ncep.noaa.gov/nwps/nwpsloop.php?site=MTR&loop=sigwaveheight&cg=3)
    has high-resolution data and is quite accurate because it uses a sophisticated forecasting model.
    However, I don't like the user interface; you have to click through individual frames and it is difficult to get numerical values.
 -  [Windy](https://www.windy.com/36.616/-121.889/gfsWaves/waves?gfs,36.515,-121.898,11) has a great user interface that also displays wind, tides, etc.
    However, the data doesn't feel very accurate since forecasts don't make much sense between dive sites; why is a 6 ft swell at Breakwater no big deal, but a 6 ft swell at Monastery un-divable?!

## Data

 -  All wave and tide forecast data comes from [NWPS](https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwps/prod/).
    We make a very similar map showing the wave height in color and wave direction with arrows.
    We also plot the wave height over time for two dive sites, [Breakwater (San Carlos beach)](https://maps.app.goo.gl/wHzyiZY1mi4THkto8) and [Monastery beach](https://maps.app.goo.gl/nZdXUZvYriEUVF8z9).
    This is more similar to Windy's user interface, and is useful for tracking how conditions for a particular site develop over time.

     -  The website updates daily when data from new NWPS runs are available.

 -  We use [basemap](https://matplotlib.org/basemap/stable/) to render the coastline map.

## Local Development

This is a Python project, and the easiest way to get started is with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```
uv venv
uv sync
```

To generate the webpage,

```
uv run -m wavey
```

This will create a directory "_site/" which contains the webpage.

We use `mypy` to perform static analysis and `ruff` to perform linting and formatting.

```
uv run mypy .
uv run ruff check .
uv run ruff format --check .
```
