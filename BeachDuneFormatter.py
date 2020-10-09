#-------------------------------------------------------------------------------
# Name:        BeachDuneFormatter.py
# Version:     Python 3.7, Pandas 0.24
#
# Purpose:     Processing beach profile data.
#
# Authors:     Ben Chittle, Alex Smith
#-------------------------------------------------------------------------------

"""
3 parts:
    A) input checking and formatting
    B) data processing (create a library of functions)
    C) output (formatting)
A)


B)
Function to identify primary features for a given profile
1. START
2. READ (xy_data)
3. shore <- identify_shore(xy_data)
4. toe <- identify_toe(xy_data, shore)
5. crest <- identify_crest(xy_data, toe)
6. heel <- identify_heel(xy_data, crest)
7. RETURN (shore, toe, crest, heel)
8. END

Function to

C)


"""
import os, re, time
import pandas as pd
import numpy as np




def read_mask_csvs(path_to_dir):
    """
    Reads all .csv files in the given directory and concatenates them into a
    single DataFrame. Each .csv file name should end with a number specifying
    the segment its data corresponds to.
    """
    # Default value since all testing data is from one state.
    STATE = 29
    INPUT_COLUMNS = ["LINE_ID", "FIRST_DIST", "FIRST_Z"]
    OUTPUT_COLUMNS = ["profile", "x", "y"]

    if not path_to_dir.endswith("\\"):
        path_to_dir += "\\"

    # Read each .csv file into a DataFrame and append it to a list.
    print("\nReading .csv's...")
    csvs = []
    for file_name in os.listdir(path_to_dir):
        head, extension = os.path.splitext(file_name)
        if extension == ".csv":
            # Look for a segment number at the end of the file name.
            segment = re.search(r"\d+$", head)

            # If a number was found, read the file and append it to the list.
            if segment is not None:
                segment = int(segment.group())

                # Read only the specified columns and reorder them in the
                # DataFrame.
                csv = pd.read_csv(path_to_dir + file_name,
                                  usecols=INPUT_COLUMNS)[INPUT_COLUMNS]
                csv.rename(columns=dict(zip(INPUT_COLUMNS, OUTPUT_COLUMNS)),
                           inplace=True)
                # Insert a segment and state column.
                csv.insert(loc=0, column="state", value=STATE)
                csv.insert(loc=1, column="segment", value=segment)
                csvs.append(csv)

                print("\tRead {} rows of data from '{}'".format(len(csv), file_name))
            else:
                print("\tSkipping '{}' (no segment number found)".format(file_name))
        else:
            print("\tSkipping '{}' (not a .csv)".format(file_name))

    return pd.concat(csvs)


def identify_shore(profile_xy):
    """Returns a filter to identify the shoreline."""
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
    """Returns a filter to identify the dune crest."""
    y = profile_xy.loc[shore_x:]["y"]

    filt = (# Current y value is the largest so far
            (y > y.shift(1).expanding(min_periods=1).max())
            # Difference between current y value and minimum of next 20 > 0.6
            & (y - y.rolling(20).min().shift(-20) > 0.6)
            # Current y value > next 20
            & (y > y.rolling(10).max().shift(-10)))

    x_coord = filt.idxmax()
    if x_coord == filt.index[0]:
        return None
    else:
        return x_coord


def identify_toe(profile_xy, shore_x, crest_x):
    subset = profile_xy.loc[shore_x:crest_x]
    x = subset["x"]
    y = subset["y"]

    # Polynomial coefficients
    A, B, C, D = np.polyfit(x=x, y=y, deg=3)
    differences = y - ((A * x ** 3) + (B * x ** 2) + (C * x) + D)
    x_coord = differences.idxmin()
    return x_coord


def identify_heel(profile_df, crest_x):
    subset = profile_df.loc[crest_x:]
    y = subset["y"]

    filt = ~(# Difference between current y value and minimum of next 10 > 0.6
            (y - y.rolling(10).min().shift(-10) > 0.6)
             # Current y value > max of previous 10 y values
             & (y > y.rolling(10).max())
             & (y > y.rolling(20).max().shift(-20)))

    x_coord = y[filt].idxmin()
    if x_coord == filt.index[0]:
        return None
    else:
        return x_coord



'''
def identify_features_1(profile_xy):
    """
    Identifies the coordinates of the shoreline, dune toe, dune crest, and
    dune heel for a given profile.
    """
    state, segment, profile = profile_xy.iloc[0][["state", "segment", "profile"]]
    profile_xy = profile_xy.set_index("x", drop=False)
    filter_funcs=[("shore", shore_filter), ("crest", crest_filter),
                  ("toe", toe_filter), ("heel", heel_filter)]

    feature_coords = {}
    for feature, filt_func in filter_funcs:
        filt = filt_func(profile_xy, feature_coords)
        if any(filt):
            # The first row to pass the filter is used to identify the feature.
            coords = tuple(profile_xy[filt].iloc[0][["x", "y"]])
            feature_coords[feature + "_x"], feature_coords[feature + "_y"] = coords
        else:
            print("\tFailed to identify {} for {}".format(feature, (state, segment, profile)))
            return None

    return feature_coords'''

'''
def identify_features_2(profile_xy):
    """
    Identifies the coordinates of the shoreline, dune toe, dune crest, and
    dune heel for a given profile.
    """
    state, segment, profile = profile_xy.iloc[0][["state", "segment", "profile"]]
    profile_xy = profile_xy.set_index("x", drop=False)

    shore_x = identify_shore(profile_xy)
    if shore_x is None:
        print("\tNo shore for {}".format((state, segment, profile)))
        return None, None, None, None
    crest_x = identify_crest(profile_xy, shore_x)
    if crest_x is None:
        print("\tNo crest for {}".format((state, segment, profile)))
        return None, None, None, None
    toe_x = identify_toe(profile_xy, shore_x, crest_x)
    if toe_x is None:
        print("\tNo toe for {}".format((state, segment, profile)))
        return None, None, None, None
    heel_x = identify_heel(profile_xy, crest_x)
    if heel_x is None:
        print("\tNo heel for {}".format((state, segment, profile)))
        return None, None, None, None

    return shore_x, toe_x, crest_x, heel_x
'''
def identify_features(profile_xy):
    """
    Identifies the coordinates of the shoreline, dune toe, dune crest, and
    dune heel for a given profile.
    """
    #profile_xy = profile_xy.set_index("x", drop=False)

    shore_x = identify_shore(profile_xy)
    if shore_x is None:
        print("\tNo shore for {}".format(tuple(profile_xy.iloc[0]["state":"profile"])))
        return None, None, None, None
    crest_x = identify_crest(profile_xy, shore_x)
    if crest_x is None:
        print("\tNo crest for {}".format(tuple(profile_xy.iloc[0]["state":"profile"])))
        return None, None, None, None
    toe_x = identify_toe(profile_xy, shore_x, crest_x)
    if toe_x is None:
        print("\tNo toe for {}".format(tuple(profile_xy.iloc[0]["state":"profile"])))
        return None, None, None, None
    heel_x = identify_heel(profile_xy, crest_x)
    if heel_x is None:
        print("\tNo heel for {}".format(tuple(profile_xy.iloc[0]["state":"profile"])))
        return None, None, None, None

    return shore_x, toe_x, crest_x, heel_x

def test_1():
    BENPATH = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\data"
    pd.options.display.max_columns = 12
    pd.options.display.width = 120

    xy_data = read_mask_csvs(BENPATH)
    #xy_data.set_index(["state", "segment", "profile"], inplace=True)
    print(xy_data)



def test_2():
    BENPATH = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\data"
    FEATURE_COLUMNS = ["state", "segment", "profile",  "shore_x",  "shore_y",
                       "toe_x", "toe_y", "crest_x", "crest_y", "heel_x",
                       "heel_y"]
    pd.options.display.max_columns = 12
    pd.options.display.width = 120

    xy_data = read_mask_csvs(BENPATH)
    print("\nStarting data manipulation")
    feature_data = {feature:list() for feature in FEATURE_COLUMNS}
    # Iterate over the data grouped first by state, then by segment, then by
    # profile.
    t1 = time.perf_counter()
    for (state, segment, profile), profile_xy in xy_data.groupby(["state", "segment", "profile"]):
        if profile == 0:
            print("\tLast segment took {}".format(time.perf_counter() - t1))
            print("\tBeginning segment {}".format(segment))
            t1 = time.perf_counter()
        feature_data["state"].append(state)
        feature_data["segment"].append(segment)
        feature_data["profile"].append(profile)

        features = identify_features(profile_xy)
        if features is not None:
            for feature, coord in identify_features(profile_xy).items():
                feature_data[feature].append(coord)
        else:
            for feature in feature_data:
                if feature not in ["state", "segment", "profile"]:
                    feature_data[feature].append(None)


def test_3():
    BENPATH = r"C:\Users\BenPc\Documents\GitHub\beach-dune-formatter\data"
    FEATURE_COLUMNS = ["state", "segment", "profile",  "shore_x",  "shore_y",
                       "toe_x", "toe_y", "crest_x", "crest_y", "heel_x",
                       "heel_y"]
    pd.options.display.max_columns = 12
    pd.options.display.width = 120

    xy_data = read_mask_csvs(BENPATH).set_index("x", drop=False)
    print("\nIdentifying features...")
    t1 = time.perf_counter()
    profile_data = xy_data.groupby(["state", "segment", "profile"]).apply(identify_features)
    print("\tTook {}".format(time.perf_counter() - t1))
    print(profile_data)



def main():
    test_3()

if __name__ == "__main__":
    main()


