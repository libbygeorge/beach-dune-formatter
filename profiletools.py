#-------------------------------------------------------------------------------
# Name:        profile.py
# Version:     Python 3.7, Pandas 0.24
#
# Purpose:     Functions for performing data analysis on a single profile.
#
# Authors:     Ben Chittle, Alex Smith
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np

def identify_shore(profile_xy):
    """Returns the x coordinate of the shoreline."""
    x = profile_xy["x"]
    y = profile_xy["y"]
    slope = (y - y.shift(1)) / (x - x.shift(1))

    filt = ((y > 0)
            # Current y value is the largest so far
            & (y > y.shift(1).expanding(min_periods=1).max())
            # Current value and next 4 slope values are positive
            & (slope.rolling(5).min().shift(-4) >= 0))

    x_coord = filt.idxmax()
    if x_coord == filt.index[0]:
        return None
    else:
        return x_coord


def identify_crest(profile_xy, shore_x):
    """Returns the x coordinate of the dune crest."""
    y = profile_xy.loc[shore_x:]["y"]
            # Current y value is the largest so far
    filt = ((y > y.shift(1).expanding(min_periods=1).max())
            # Difference between current y value and minimum of next 20 > 0.6
            & (y - y.rolling(20).min().shift(-20) > 0.6)
            # Current y value > next 10
            & (y > y.rolling(10).max().shift(-10)))

    x_coord = filt.idxmax()
    if x_coord == filt.index[0]:
        return None
    else:
        return x_coord


### COMPARE TO NEW
def identify_toe_old(profile_xy, shore_x, crest_x):
    """Returns the x coordinate of the dune toe."""
    subset = profile_xy.loc[shore_x:crest_x].iloc[1:-2]
    x = subset["x"]
    y = subset["y"]

    # Polynomial coefficients
    A, B, C, D = np.polyfit(x=x, y=y, deg=3)
    differences = y - ((A * x ** 3) + (B * x ** 2) + (C * x) + D)
    x_coord = differences.idxmin()
    return x_coord


def identify_toe(profile_xy, shore_x, crest_x):
    """Returns the x coordinate of the dune toe."""
    subset = profile_xy.loc[shore_x:crest_x]
    x = subset["x"]
    y = subset["y"]

    # Polynomial coefficients
    A, B, C, D = np.polyfit(x=x, y=y, deg=3)
    differences = y - ((A * x ** 3) + (B * x ** 2) + (C * x) + D)
    x_coord = differences.idxmin()
    return x_coord


def identify_heel(profile_df, crest_x):
    """Returns the x coordinate of the dune heel."""
    subset = profile_df.loc[crest_x:]
    y = subset["y"]
             # Difference between current y value and minimum of next 10 > 0.6
    filt = ~((y - y.rolling(10).min().shift(-10) > 0.6)
             # Current y value > max of previous 10 y values
             & (y > y.rolling(10).max())
             & (y > y.rolling(20).max().shift(-20)))

    x_coord = y[filt].idxmin()
    if x_coord == filt.index[0]:
        return None
    else:
        return x_coord


def grouped_mean(profiles, n):
    """Returns a new DataFrame with the mean of every n rows for each column."""
    return profiles.groupby(np.arange(len(profiles)) // n).mean()


def identify_features(profile_xy):
    """
    Returns the coordinates of the shoreline, dune toe, dune crest, and
    dune heel for a given profile as:
    (shore_x, shore_y, toe_x, toe_y, crest_x, crest_y, heel_x, heel_y)
    """
    shore_x = identify_shore(profile_xy)
    if shore_x is None:
        return None

    crest_x = identify_crest(profile_xy, shore_x)
    if crest_x is None:
        return None

    toe_x = identify_toe(profile_xy, shore_x, crest_x)
    if toe_x is None:
        return None

    heel_x = identify_heel(profile_xy, crest_x)
    if heel_x is None:
        return None

    shore_y, toe_y, crest_y, heel_y = profile_xy.loc[[shore_x, toe_x, crest_x, heel_x], "y"]

    return shore_x, shore_y, toe_x, toe_y, crest_x, crest_y, heel_x, heel_y


def measure_volume(profile_xy, start_x, end_x, profile_spacing, base_elevation=0):
    """
    Returns an approximation of the volume of the beach between two points.

    ARGUMENTS
    profile_xy: DataFrame
      xy data for a particular profile.
    start_x: float
      The x coordinate of the start of the range.
    end_x: float
      The x coordinate of the end of the range.
    profile_spacing: float
      The distance between consecutive profiles.
    base_elevation: float
      Set the height of the horizontal axis to measure volume from. Change this
      if y values are relative to an elevation other than y=0.
    """
    subset = profile_xy.loc[start_x:end_x]
    x = subset["x"]
    y = subset["y"]

    # Make all elevation values relative to the base elevation.
    y -= base_elevation

    # The area under the profile curve is calculated using the trapezoidal rule
    # and multiplized by the distance between consecutive profiles to
    # approximate the volume.
    return np.trapz(y=y, x=x) * profile_spacing