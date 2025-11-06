# Wavey

[_--> live website <--_](https://kevinddchen.github.io/wavey/)

## Manifesto

 -  I don't like the [Nearshore Wave Prediction System (NWPS) model viewer](https://polar.ncep.noaa.gov/nwps/nwpsloop.php?site=MTR&loop=sigwaveheight&cg=3)
    because you have to click through the frames and it is difficult to get a numerical value for the wave height.
 -  I don't like [Windy](https://www.windy.com/36.616/-121.889/gfsWaves/waves?gfs,36.515,-121.898,11) because the forecasts don't make much sense
    between dive sites; why is a 6 ft swell at Breakwater no big deal, but a 6 ft swell at Monastery un-divable?!

**The goal of this project is to combine the accuracy of NWPS with the user interface of Windy.**

## Summary

The wave and swell forecast data for [NWPS](https://polar.ncep.noaa.gov/nwps/) is obtained from https://nomads.ncep.noaa.gov/.
The map visuals were kept largely intact, with the major modifications being zooming into the Monterey peninsula, and adding a slider to scrub through time.
A graph was added to plot the wave height over time for two dive sites, Breakwater (San Carlos beach) and Monastery beach.
Similar to Windy's user interface, this can be useful for tracking how conditions for a particular site develop over time.

We use [basemap-data-hires](https://matplotlib.org/basemap/stable/) to render the coastline map.

## Local Development

This is a Python project, and the easiest way to get started is with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```
uv venv
uv sync
```

To generate the webpage,

```
uv run main.py
```

This will create a directory "_site/" which contains the webpage.
