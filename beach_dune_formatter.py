#-------------------------------------------------------------------------------
# Name:        beach-dune-formatter.py
# Version:     Python 3.9, pandas 1.2.0, numpy 1.19.3, matplotlib 3.3.3
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
"""
import os, re, time
import pandas as pd
import numpy as np
import profiletools


BEN_IN = r"C:\Users\Ben2020\Documents\GitHub\beach-dune-formatter\sample_data"
BEN_OUT = r"C:\Users\Ben2020\Documents\GitHub\beach-dune-formatter\out.xlsx"
UNI_IN = r"E:\SA\Runs\Poly\tables"
UNI_OUT =  r"E:\SA\Runs\Poly\tables\b_poly.xlsx"


####################### PATH SETTINGS #######################
# Change these variables to modify the input and output paths
# (type the path directly using the format above if needed).
current_input = BEN_IN
current_output = BEN_OUT
#############################################################


def read_mask_csvs(path_to_dir):
    """
    Reads all .csv files in the given directory and concatenates them into a
    single DataFrame. Each .csv file name should end with a number specifying
    the segment its data corresponds to (i.e. 'data_file19.csv' would be
    interpreted to contain data for segment 19). Returns a DataFrame with the
    following columns: "state", "segment", "profile", "x", "y".
    """
    ### Default value since all testing data is from one state.
    STATE = np.uint8(29)

    # For mapping the input column names to the desired output column names.
    INPUT_COLUMNS = ["LINE_ID", "FIRST_DIST", "FIRST_Z"]
    OUTPUT_COLUMNS = ["profile", "x", "y"]

    # Data types corresponding to each input column.
    DTYPES = dict(zip(INPUT_COLUMNS, [np.uint16, np.float32, np.float32]))

    if not path_to_dir.endswith("\\"):
        path_to_dir += "\\"

    # Read each .csv file at the provided path into a DataFrame and append it to
    # a list.
    csvs = []
    for file_name in os.listdir(path_to_dir):
        head, extension = os.path.splitext(file_name)
        if extension == ".csv":
            # Look for a segment number at the end of the file name.
            segment = re.search(r"\d+$", head)

            # If a number was found, read the file and append it to the list.
            if segment is not None:
                segment = np.int16(segment.group())

                # Read only the desired columns and reorder them in the
                # DataFrame.
                csv_data = pd.read_csv(path_to_dir + file_name,
                                       usecols=INPUT_COLUMNS,
                                       dtype=DTYPES)[INPUT_COLUMNS]
                csv_data.rename(columns=dict(zip(INPUT_COLUMNS, OUTPUT_COLUMNS)),
                                inplace=True)
                # Insert a column for the segment and state values.
                csv_data.insert(loc=0, column="state", value=STATE)
                csv_data.insert(loc=1, column="segment", value=segment)

                csvs.append(csv_data)

                print("\tRead {} rows of data from file '{}'".format(len(csv_data), file_name))
            else:
                print("\tSkipping file '{}' (no segment number found)".format(file_name))
        else:
            print("\tSkipping file '{}' (not a .csv)".format(file_name))

    # Combine the .csvs into a single DataFrame and set the index to the x
    # values column.
    return pd.concat(csvs)


###ASSUMES THAT DATA CAN BE GROUPED BY state, segment, profile.
###RENAME.
def measure_feature_volumes(xy_data, start_values, end_values, base_elevations):
    """
    Returns a list of volumes calculated between the start and end point given
    for each profile.

    ARGUMENTS
    xy_data: DataFrame
      xy data of a collection of profiles.
    start_values: iterable
      The x value to start measuring volume from (a value must be provided 
      for each profile).
    end_values: iterable
      The x value to stop measuring volume from (a value must be provided 
      for each profile).
    base_elevations: iterable
      The elevation to use as zero (a value must be provided for each profile.)
    """
    # The distance between consecutive profiles. Uses the distance between the
    # first two consecutive x values, which assumes the profiles were taken from
    # a square grid.
    profile_spacing = xy_data["x"].iat[1] - xy_data["x"].iat[0]

    grouped_xy = xy_data.groupby(["state", "segment", "profile"])
    data = []
    # Measure the volume for each profile between the corresponding start and
    # end x value in start_values and end_values.
    for (index, profile_xy), start_x, end_x, base_y in zip(grouped_xy, start_values, end_values, base_elevations):
        data.append(profiletools.measure_volume(profile_xy, start_x, end_x, profile_spacing, base_y))

    return data


def write_data_excel(path_to_file, dataframes, names):
    """
    Write data to an Excel file.

    ARGUMENTS
    path_to_file: string
    dataframes: iterable
      Sequence of DataFrames to write. Each element is written on a new sheet.
    names: iterable
      The name of each sheet, corresponding to each element in 'dataframes'.
    """
    with pd.ExcelWriter(path_to_file) as writer:
        for name, data in zip(names, dataframes):
            try:
                data.to_excel(writer, name)
            except IndexError:
                print("\tFailed to write '{}' to file (the DataFrame may be"
                      " empty).".format(name))


### HAVE THE USER DECLARE HOW THEIR DATA IS CATEGORIZED / ORGANIZED
### (need to know for groupby operations)
def main(input_path, output_path):
    FEATURE_COLUMNS = ["shore_x", "shore_y", "toe_x", "toe_y", "crest_x",
                       "crest_y", "heel_x", "heel_y"]

    pd.options.display.max_columns = 12
    pd.options.display.width = 100

    initial_start_time = time.perf_counter()

    print("\nReading .csv's...")
    start_time = time.perf_counter()
    xy_data = read_mask_csvs(input_path)

    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))


    print("\nIdentifying features...")
    start_time = time.perf_counter()

    # Identify the shoreline, dune toe, dune crest, and dune heel for each
    # profile in the data. This data will be returned as a Pandas Series
    # containing tuples of the 4 pairs of coordinates for each profile.
    profile_data = xy_data.groupby(["state", "segment", "profile"]).apply(profiletools.identify_features)

    # Expand the Series of tuples into a DataFrame where each column contains an
    # x or y componenent of a feature.
    profile_data = pd.DataFrame(profile_data.to_list(), columns=FEATURE_COLUMNS, index=profile_data.index)
    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))

    print("\nCalculating beach data...")
    start_time = time.perf_counter()

    # Use the feature data for each profile to calculate additional
    # characteristics of the beach. This includes the dune height, beach width,
    # dune toe height, dune crest height, dune length, beach slope, dune slope,
    # beach volume, dune volume, and beach-dune volume ratio.
    beach_data = pd.DataFrame(
        data={"dune_height" : profile_data["crest_y"] - profile_data["toe_y"],
              "beach_width" : profile_data["toe_x"] - profile_data["shore_x"],
              "dune_toe" : profile_data["toe_y"],
              "dune_crest" : profile_data["crest_y"],
              "dune_length" : profile_data["crest_x"] - profile_data["toe_x"]},
        index=profile_data.index)

    # Approximates the beach slope as the total change in height of the beach
    # divided by the length of the beach.
    beach_data["beach_slope"] = (profile_data["toe_y"] - profile_data["shore_y"]) / beach_data["beach_width"]

    # Approximates the dune slope as the change in height of the dune divided by
    # the length of the dune.
    beach_data["dune_slope"] = beach_data["dune_height"] / beach_data["dune_length"]

    beach_data["beach_vol"] = measure_feature_volumes(
                                  xy_data,
                                  start_values=profile_data["shore_x"],
                                  end_values=profile_data["toe_x"],
                                  base_elevations=profile_data["shore_y"])
    ### ARE ELEVATIONS RELATIVE TO TOE Y OR SHORE Y?
    beach_data["dune_vol"] = measure_feature_volumes(
                                 xy_data,
                                 start_values=profile_data["toe_x"],
                                 end_values=profile_data["crest_x"],
                                 base_elevations=profile_data["toe_y"])
    ### SHOULD THIS BE db_ratio?
    beach_data["bd_ratio"] = beach_data["dune_vol"] / beach_data["beach_vol"]
    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))


    print("\nFiltering data...")
    start_time = time.perf_counter()

    # Replace values outside of the desired bounds with NaNs.
    filtered_beach_data = beach_data.copy()
    filtered_beach_data.loc[beach_data["dune_vol"] >= 300, "dune_vol"] = np.nan
    filtered_beach_data.loc[beach_data["beach_vol"] >= 500, "beach_vol"] = np.nan
    filtered_beach_data.loc[(beach_data["dune_height"] > 1) & (beach_data["dune_height"] < 10), "dune_height"] = np.nan
    filtered_beach_data.loc[(beach_data["dune_length"] > 5) & (beach_data["dune_length"] < 25), "dune_length"] = np.nan
    filtered_beach_data.loc[beach_data["dune_crest"] < 20, "dune_crest"] = np.nan
    filtered_beach_data.loc[(beach_data["beach_width"] > 10) & (beach_data["beach_width"] < 60), "beach_width"] = np.nan
    filtered_beach_data.loc[beach_data["dune_toe"] > 2 * beach_data["dune_toe"].rolling(10)]
 
    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))


    print("\nAveraging data...")
    start_time = time.perf_counter()

    # Takes the mean of each column for every 10 profiles.
    averaged_beach_data = beach_data.groupby(["state", "segment"]).apply(profiletools.grouped_mean, 10)
    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))


    print("\nCorrelating data...")
    start_time = time.perf_counter()
    corr1 = beach_data.corr(method="pearson")
    corr2 = filtered_beach_data.corr(method="pearson")
    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))


    print("\nWriting to file...")
    start_time = time.perf_counter()
    write_data_excel(path_to_file=output_path,
                     dataframes=(profile_data, beach_data, corr1, filtered_beach_data,
                                 corr2, averaged_beach_data),
                     names=("profile_data", "unfiltered", "corr_1", "filtered",
                            "corr_2", "averages"))
    print("\tTook {:.2f} seconds".format(time.perf_counter() - start_time))

    print("\nTotal time: {:.2f} seconds".format(time.perf_counter() - initial_start_time))

    return xy_data, profile_data, beach_data, filtered_beach_data, averaged_beach_data


if __name__ == "__main__":
    xy_data, profile_data, beach_data, filtered_beach_data, avg = main(current_input, current_output)
    #main(current_input, current_output)

