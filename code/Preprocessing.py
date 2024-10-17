import numpy as np
import dask.dataframe as dd
import math
import xarray
import pandas as pd
import os
import json
from datetime import date, datetime
from math import sin, cos, pi
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import logging, sys
from datetime import timedelta

logging.getLogger().setLevel(logging.INFO)
# pd.set_option('future.no_silent_downcasting', True)

import warnings
import pandas as pd
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))


class FileExtractor:
    """Helps at reading and extracting files for the preprocessing. It expects the file formats .nc, .csv or .json."""

    def __init__(self):
        pass


    def extract_json_files(self, path):
        """Combines all JSON files in a given path (only on the surface level, no subfolders) into a single pandas.Dataframe.+
        
        Parameters:
            - path: a String object containing the location of the files.

        Returns:
            - json_data: pandas.DataFrame containing the JSON data.
        """

        json_data = None

        for file in os.listdir(path):
            if file.split(".")[-1] == "json":
                with open(path + "/" + file) as f:
                    d = json.load(f)
                    d = pd.json_normalize(d)
                    if type(json_data) == type(None):
                        json_data = d
                    else:
                        json_data = pd.concat([json_data, d])

        return json_data
    

    def combine_files(self, path, file_name, file_format:str=".nc"):
        """Combine files of the same type of file name pattern with different timestamps. Converts them into a dataframe.
        
        Parameters:
            - path: a String object containing the location of the files.
            - file_name: file name pattern of the files (they should have one in common, e.g. 'dwd_icon_eu_hornsea')
            - file_format: the format of the files to extract.

        Returns:
            - df_total: pandas.DataFrame containing the extracted data.
        
        """

        # outage data is encoded in .json files
        if file_format.lower() == ".json":
            return self.extract_json_files(path)

        df = None
        df_new = None

        all_dfs = list()

        for file in os.listdir(path):
            # check for files with the file name pattern
            if file.endswith(file_format) and file_name.lower() in file.lower(): 
                # weather data is encoded in .nc files
                if file_format == ".nc":
                    ds = xarray.open_dataset(path + "/" + file)
                    df_new = ds.to_dataframe()
                    # the weather data has different indices and orders of index names which need to be sorted.
                    try:
                        if len(df_new.index[0]) == 3:
                            df_new = df_new.reorder_levels(["ref_datetime", "valid_datetime", "point"])
                        elif len(df_new.index[0]) == 4:
                            df_new = df_new.reorder_levels(["ref_datetime", "valid_datetime", "latitude", "longitude"])
                        elif len(df_new.index[0]) == 5:
                            df_new = df_new.reorder_levels(["ref_datetime", "valid_datetime", "latitude", "longitude", "point"])
                        else:
                            df_new = df_new.reorder_levels(["ref_datetime", "valid_datetime"])
                    except:
                        if len(df_new.index[0]) == 3:
                            df_new = df_new.reorder_levels(["reference_time", "valid_time", "point"])
                        elif len(df_new.index[0]) == 4:
                            df_new = df_new.reorder_levels(["reference_time", "valid_time", "latitude", "longitude"])
                        elif len(df_new.index[0]) == 5:
                            df_new = df_new.reorder_levels(["reference_time", "valid_time", "latitude", "longitude", "point"])
                        else:
                            df_new = df_new.reorder_levels(["reference_time", "valid_time"])

                    all_dfs.append(df_new)
                # energy data is encoded in .csv files
                elif file_format == ".csv":
                    df = pd.read_csv(path + "/" + file)
                    all_dfs.append(df)

        df_total = pd.concat(all_dfs)

        return df_total
    


class Preprocessing:
    """Contains the data cleaning and feature extraction for the datasets. Includes various preprocessing steps like handling invalid values and outliers, computing aggregations and combining different datasets."""

    def __init__(self):
        self.irrelevant_features = list()


    def perform_preprocessing_pipeline(self, geo_data_dict:dict, 
                                       aggregate_by:str = "val_time", aggregate_by_ref_time_too:bool = True,
                                       merge_with_outage_data:bool = True, json_file_path:str = "nc_files/REMIT",
                                       non_numerical_columns:list = ["unavailabilityType", "affectedUnit"],
                                       fft:bool = False, columns_to_fft:list = ["temp_diff", "solar_down_rad_diff", "wind_speed_diff", "wind_speed_100_diff"],
                                       deployment:bool = True, energy_data_dict:dict = dict(), left_merge:str = "val_time", right_merge:str = "dtm"):
        """The main method which calls upon all the other methods necessary for completing the preprocessing pipeline.
        
        Parameters:
            - geo_data_dict: dictionary object which contains the file name pattern as a key and the corresponding file path as a value. Since multiple files might be in the same path, the path is considered here as the value.
                            The values can also consist of pandas.DataFrames, which will lead to skipping the file reading step with the FIleExtractor class.
            - aggregate_by_ref_time_too: if True, aggregates additionally by the reference time of weather forecasts, otherwise the aggregation will only be by the valid time of the forecast.
            - merge_with_outage_data: if True, merges with the REMIT data containing information about outages at the HORNSEA wind farm (stored as JSON files).
            - json_file_path: file location of the REMIT data. Should be a valid path if merge_with_outage_data is set to True. Will be ignored, if merge_with_outage_data is set to False.
            - non_numerical_columns: list of non-numerical columns. Necessary to specify since the aggregation steps for non-numerical columns occur differently from the aggregation of numerical ones.
            - fft: if True, adds features which are the result of a Fast Fourier Transformation (FFT) on specific features in the dataset.
            - columns_to_fft: columns which the FFT is supposed to be computed on. If fft is set to False, this will be ignored.
            - deployment: if True, the preprocessing will expect no available label data and adjusts the preprocessing.
            - energy_data_dict: same structure as the geo_data_dict, except it does not expect pandas DataFrames as values. Contains file name patterns and file paths for the energy data.
            - left_merge: column to use for combining the weather data with other datasets (REMIT data and energy data)
            - right_merge: column to use for combining the energy data with the weather and REMIT data

        Returns:
            - df: preprocessed pandas.DataFrame object
        """

        weather_data = list()
        extractor = FileExtractor()
        self.non_numerical_columns = non_numerical_columns

        # if the values in the weather data dict contains string values, it is interpreted as a file path. Read the files from the path
        if type(list(geo_data_dict.values())[0]) == type(str()):
            for file_name_pattern, file_path in geo_data_dict.items():
                weather_data.append(extractor.combine_files(file_path, file_name_pattern, ".nc"))
        # otherwise, the values will be interpreted as DataFrames
        elif type(list(geo_data_dict.values())[0]) == type(pd.DataFrame()):
            assert "dwd" in geo_data_dict.keys(), "geo_data_dict expects 'dwd' and 'ncep' as corresponding keys for the DWD and NCEP weather data."
            assert "ncep" in geo_data_dict.keys(), "geo_data_dict expects 'dwd' and 'ncep' as corresponding keys for the DWD and NCEP weather data."

            weather_data.append(geo_data_dict["dwd"])
            weather_data.append(geo_data_dict["ncep"])
        else:
            logging.critical("Input must be either a file path or a pandas.DataFrame object.")
            sys.exit(0)
            
        logging.info("Perform data cleaning on the weather data...")
        for index in range(0, len(weather_data)):
            weather_data[index] = self.preprocess_geo_data(weather_data[index])

        logging.info("Merge weather stations...")
        df = self.merge_weather_stations_data(weather_data_1 = weather_data[0], weather_data_2 = weather_data[1], aggregate_by = aggregate_by, aggregate_by_ref_time_too = aggregate_by_ref_time_too)
        df = self.add_difference_features(df)

        # the outage data contains information only relevant about the HORNSEA wind park. Thus, it is only relevant for data regarding the wind power forecast.
        if merge_with_outage_data and "wind_speed" in df.columns:
            logging.info("Merge with outages data (REMIT)...")
            outage_data = extractor.extract_json_files(json_file_path)
            outage_data = self.preprocess_outage_data(outage_data, deployment = deployment)
            df = self.merge_with_outages(df, outage_data)

        if fft:
            df = self.add_fft_features(df, columns_to_fft = columns_to_fft)

        if (deployment != True) | ("solar_down_rad" in df.columns):
            logging.info("Merge with energy data...")
            key = next(iter(energy_data_dict))
            energy_df = extractor.combine_files(energy_data_dict[key], key, ".csv")
            if "wind_speed" in df.columns:
                only_labels = True
            else:
                only_labels = False
            energy_df = self.preprocess_energy_data(energy_df, deployment = deployment, only_labels = only_labels)
            df = self.merge_geo_energy_outage_data(df, energy_df, left_merge = left_merge, right_merge = right_merge, deployment = deployment)

        # sort the columns alphabetically to avoid errors during the Feature Engineering (aka keep the same order)
        df = df.reindex(sorted(df.columns), axis=1)
        logging.info("Preprocessing done!")

        return df


    def cyclic_sin(self, n):
        """Performs a cyclic sine conversion on a number.
        
        Parameters:
            - n: input value
        Returns:
            - theta: output value.
        
        """
        theta = 2 * pi * n
        return sin(theta)


    def cyclic_cos(self, n):
        """Performs a cyclic cosine conversion on a number.
        
        Parameters:
            - n: input value
        Returns:
            - theta: output value.
        
        """
        theta = 2 * pi * n
        return cos(theta)


    def get_cycles(self, d, sin_or_cos:int, time_information:str):
        """
        Get the cyclic properties of a datetime,represented as points on the unit circle.

        Parameters:
            - d: datetime object
            - sin_or_cos: value to decide if cyclic_sin (if 0) or cyclic_cos (if 1) shall be called
            - time_information: name of the time information (e.g. 'month')
        Returns:
            transformed values.

        Source of the implementation of this method: https://medium.com/@dan.allison/how-to-encode-the-cyclic-properties-of-time-with-python-6f4971d245c0
        """
        month = d.month - 1
        day = d.day - 1

        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        if sin_or_cos == 0:
            if time_information.lower() == "month":
                return self.cyclic_sin(month / 12)
            elif time_information.lower() == "day":
                return self.cyclic_sin(day / days_in_month[month])
            # elif time_information.lower() == "dayofweek":
            #     return self.cyclic_sin(d.weekday() / 7)
            elif time_information.lower() == "hour":
                return self.cyclic_sin(d.hour / 60)
            elif time_information.lower() == "minute":
                return self.cyclic_sin(d.minute / 60)
            else:
                return
        elif sin_or_cos == 1:
            if time_information.lower() == "month":
                return self.cyclic_cos(month / 12)
            elif time_information.lower() == "day":
                return self.cyclic_cos(day / days_in_month[month])
            elif time_information.lower() == "dayofweek":
                return self.cyclic_cos(d.weekday() / 7)
            elif time_information.lower() == "hour":
                return self.cyclic_cos(d.hour / 24)
            elif time_information.lower() == "minute":
                return self.cyclic_cos(d.minute / 60)
            else:
                return
    

    def preprocess_outage_data(self, json_data, deployment:bool = False):
        """Performs data cleaning and feature extraction on the REMIT data.
        
        Parameters:
            - json_data: pandas.DataFrame containing the data which has been extracted from the JSON files by a FileExtractor class object.
            - deployment: handles the data as unknown (test) data if set to True.
        
        Returns:
            - outages_df: preprocessed REMIT data as a pandas.DataFrame object.
        """
        # dismissed outages are not relevant
        json_data = json_data[json_data.eventStatus != "Dismissed"]
        # drop those columns since they provide no value as features
        json_data.drop(columns = ["id", "outageProfile", "assetId", "affectedUnitEIC", "dataset", "eventStatus", "cause", "publishTime", "createdTime", "relatedInformation", "revisionNumber", "mrid"], axis = 1, inplace = True)

        for col in ["affectedArea", "participantId", "assetType", "biddingZone", "fuelType", "eventType", "messageHeading", "messageType", "registrationCode"]:
            if col in json_data:
                json_data.drop(columns = [col], axis = 1, inplace = True)
        
        # drop string value columns with only one unique value. If it´s not used for the deployment, we are in the training phase instead. Search through every column and save their names so during deployment, they will get removed instantly next time.
        # if the preprocessing is used for the deployment, we already know which features provide no information. During deployment we might have to use interference on a single row, so just searching for columns with one unique value can only work
        # during the preprocessing of training data.
        if deployment == False:
            for col in json_data.columns:
                # remove columns which contain only one unique value.
                if json_data[col].nunique() == 1:
                        try:
                            json_data.drop(columns = col, inplace = True, axis = 1)
                            self.irrelevant_features.append(col)
                        except:
                            continue
        else:
            # irrelevant_features stores any features which contained only one unique value in the training data. They will be removed during deployment.
            if len(self.irrelevant_features) > 0:
                json_data.drop(columns = self.irrelevant_features, axis = 1, inplace = True)

        # convert the datetime information to the right format
        for col in ["eventStartTime", "eventEndTime"]:
            json_data[col] = pd.to_datetime(json_data[col])

        outages_df = pd.DataFrame(columns = json_data.columns)

        # the outages contain start and end times for each outage. With this for loop we create for each 30min interval a new row and add it to the dataframe.
        # We basically include every time period inside of an outage in 30min periods.
        for row in json_data.iterrows():
            dates_30min_freq = pd.date_range(start = row[1]["eventStartTime"], end = row[1]["eventEndTime"], freq = "30min", inclusive = "both")
            json_data_with_date_ranges = pd.DataFrame(index = dates_30min_freq, columns = json_data.columns)
            json_data_with_date_ranges.iloc[:] = row[1]
            json_data_with_date_ranges = json_data_with_date_ranges.infer_objects(copy=False)
            outages_df = pd.concat([outages_df, json_data_with_date_ranges])

        # create new features for the outages
        outages_df["hoursSinceOutage"] = (outages_df.index - outages_df.eventStartTime).div(pd.Timedelta("1h"))
        outages_df["hoursUntilOutageEnd"] = (outages_df.eventEndTime - outages_df.index).div(pd.Timedelta("1h"))
        outages_df["outage"] = True
        #outages_df["mrid"] = outages_df["mrid"].str.split("-").str[-1]

        outages_df["unavailableCapacity"] = pd.to_numeric(outages_df["unavailableCapacity"])
        outages_df["availableCapacity"] = pd.to_numeric(outages_df["availableCapacity"])

        outages_df.drop(columns = ["eventStartTime", "eventEndTime", "normalCapacity"], axis = 1, inplace = True)

        return outages_df
    

    def merge_with_outages(self, df, outage_data):
        """Merge weather data with REMIT data and extract some new features.
        
        Parameters:
            - df: weather data as a pandas.DataFrame object.
            - outage_data: REMIT data as a pandas.DataFrame object.
        Returns:
            - df: merged data as a pandas.DataFrame object.
        """
        
        # the REMIT data has not the same time intervals as the weather data. Thus, an approximative merge is used.
        df = pd.merge_asof(left = df, right = outage_data.sort_index(), left_index = True, right_index = True, direction = "nearest", tolerance = pd.Timedelta("30m"))

        # the merge will result in NA values (since outages are not present all the time), thus they have to be filled with replacement values
        df[["affectedUnit", "unavailabilityType"]] = df[["affectedUnit", "unavailabilityType"]].fillna("None")
        df[["unavailableCapacity", "hoursSinceOutage", "hoursUntilOutageEnd"]] =df[["unavailableCapacity", "hoursSinceOutage", "hoursUntilOutageEnd"]].fillna(0)
        df[["availableCapacity"]] = df[["availableCapacity"]].fillna(400)
        df["outage"] = df["outage"].fillna(False).astype(int)

        return df
    

    def preprocess_energy_data(self, df_energy, deployment:bool = False, only_labels:bool = True):
        """Data cleaning and feature extraction of the energy data.
        
        Parameters:
            - df_energy: energy data as a pandas.DataFrame object which is the result of the extracted files by the FileExtractor class.
            - deployment: if True, will assume there is no available label data and only extract the features.
            - only_labels: if True, will make sure that during the training phase for wind energy production, only the relevant labels and time information will be extracted.

        Returns:
            - df_energy: preprocessed energy data as a pandas.DataFrame object.
        """
        # convert the datetime information to the right format
        df_energy.rename(columns = {"timestamp_utc":"dtm"}, inplace = True)

        if "Unnamed: 0" in df_energy.columns:
            df_energy.drop(["Unnamed: 0"], axis = 1, inplace = True)

        df_energy["dtm"] = pd.to_datetime(df_energy["dtm"])
        df_energy = df_energy.sort_values("dtm")

        # Group by year, month, and hour, then calculate the mean
        grouped_means = df_energy.groupby([df_energy.dtm.dt.year, df_energy.dtm.dt.month, df_energy.dtm.dt.hour]).transform('mean')

        # Fill missing values in df_energy with the corresponding grouped means
        df_energy = df_energy.fillna(grouped_means)

        if deployment == False:
            # convert MW to MWh (30min periods --> multiply by 0.5)
            for col in ["Solar_MW", "Wind_MW"]:
                df_energy[col] = 0.5 * df_energy[col]

            df_energy["Wind_MWh_credit"] = df_energy["Wind_MW"] - df_energy["boa_MWh"]
            df_energy.rename(columns = {"Solar_MW": "Solar_MWh_credit", "Wind_MW": "Wind_MWh", "Solar_installedcapacity_mwp": "installed_capacity_mwp", "Solar_capacity_mwp": "capacity_mwp"}, inplace = True)
            df_energy = df_energy.drop_duplicates()
            df_energy.drop(["Wind_MWh"], axis = 1, inplace = True)

            # for the case, if the preprocessing is set during the training phase, but the weather data is only relevant for the wind energy forecast
            if only_labels:
                df_energy = df_energy[["dtm", "Wind_MWh_credit", "Solar_MWh_credit"]]
            else:
                df_energy["unused_capacity_mwp"] = df_energy["installed_capacity_mwp"] - df_energy["capacity_mwp"]
                df_energy = df_energy[["dtm", "Wind_MWh_credit", "Solar_MWh_credit", "installed_capacity_mwp", "capacity_mwp", "unused_capacity_mwp"]]
        else:
            df_energy.rename(columns = {"Solar_installedcapacity_mwp": "installed_capacity_mwp", "Solar_capacity_mwp": "capacity_mwp"}, inplace = True)
            df_energy = df_energy.drop_duplicates()
            df_energy["unused_capacity_mwp"] = df_energy["installed_capacity_mwp"] - df_energy["capacity_mwp"]
            df_energy = df_energy[["dtm", "installed_capacity_mwp", "capacity_mwp", "unused_capacity_mwp"]]

        return df_energy


    def preprocess_geo_data(self, df):
        """Data cleaning and feature extraction of the weather data.
        
        Parameters:
            - df: weather data as a pandas.DataFrame object which is the result of the extracted files by the FileExtractor class.

        Returns:
            - df: preprocessed weather data as a pandas.DataFrame object.
        """

        df = self.clean_geo_data(df)
        df = self.add_statistical_data(df)
        df = self.add_other_features(df)
        # some new features will have NaN value, e.g. due to not enough past values to calculate a rolling mean
        df = df.bfill()
        df = df.ffill()

        if "index" in df.columns:
            df.drop(["index"], axis = 1, inplace = True)

        return df


    def clean_geo_data(self, df):
        """Data cleaning of the weather data. Includes correct type formatting, renaming columns and removing outliers and missing data.
        
        Parameters:
            - df: weather data.
        
        Returns:
            - df: cleaned weather data.
        """
        # reset the index (reference_time, valid_time, latitude, longitude)
        df.reset_index(inplace = True)

        if "index" in df.columns:
            df.drop(columns = ["index"], axis = 1, inplace = True)
        if "point" in df.columns:
                df = df.drop("point", axis=1)

        df = df.rename(columns = 
                       {"level_0": "ref_time",
                        "level_1": "val_time",
                        "reference_datetime": "ref_time",
                        "ref_datetime":"ref_time",
                        "valid_datetime": "val_time",
                        "valid_time": "val_time",
                        "latitude": "lat",
                        "longitude": "long",
                        "RelativeHumidity": "rel_hum",
                        "Temperature": "temp",
                        "TotalPrecipitation": "total_prec",
                        "WindDirection" : "wind_dir",
                        "WindDirection:100": "wind_dir_100",
                        "WindSpeed": "wind_speed",
                        "WindSpeed:100" : "wind_speed_100",
                        "CloudCover": "cloud_cover",
                        "SolarDownwardRadiation": "solar_down_rad"})
        
        if "wind_speed" in df.columns:
            # convert wind speed from m/s to km/h
            df["wind_speed"] = df["wind_speed"] * 3.6
        if "wind_speed_100" in df.columns:
            # convert wind speed from m/s to km/h
            df["wind_speed_100"] = df["wind_speed_100"] * 3.6
        
        # convert the datetime information to the right format

        df["ref_time"] = pd.to_datetime(df["ref_time"])

        if df["ref_time"].dt.tz is None:
            df["ref_time"] = df["ref_time"].dt.tz_localize("UTC")

        if not pd.api.types.is_datetime64_any_dtype(df['val_time']):
            if df["val_time"].max() < 1000:
                df["forecast_horizon"] = df["val_time"]
                df["val_time"] = df["ref_time"] + pd.to_timedelta(df["val_time"], unit = "hour")
            else:
                df["val_time"] = pd.to_datetime(df["val_time"])
                if df["val_time"].dt.tz is None:
                    df["val_time"] = df["val_time"].dt.tz_localize("UTC")
                df["forecast_horizon"] = (df["val_time"] - df["ref_time"]).div(pd.Timedelta("1h"))
        else:
            df["forecast_horizon"] = (df["val_time"] - df["ref_time"]).div(pd.Timedelta("1h"))


        # remove forecasts which extend beyond the day ahead, since they will be outdated the next day anyway
        df = df[(df["val_time"] - df["ref_time"]).div(pd.Timedelta("1h")) < 50]
        # some data points have a miscalculation at their coordinates (e.g. ncep_gfs_demand). The actual coordinates can be identified by their value of the feature "point"

        df["long"] = df["long"].astype(float)
        df["lat"] = df["lat"].astype(float)

        df.loc[df.long > 90, "long"] -= 360
        df.loc[df.long < -90, "long"] += 360

        # there are anomalies of the solar down radiation being above 1000 in a short time period. The maximum threshold is to be believed to be about 1000 W/m^2
        # source: https://www.researchgate.net/post/Are_there_minimum_and_maximum_threshold_of_solar_irradiance
        if "solar_down_rad" in df.columns:
            df = df[df["solar_down_rad"] <= 1000]
            df.loc[df["solar_down_rad"] < 0, "solar_down_rad"] = 0
            # convert W/m^2 to kW/km^2
            # df["solar_down_rad"] = df["solar_down_rad"] * 1000

        if "rel_hum" in df.columns:
            df.loc[df["rel_hum"] > 100, "rel_hum"] = 100
            df.loc[df["rel_hum"] < 0, "rel_hum"] = 0
                
        if "total_prec" in df.columns:
            df.loc[df["total_prec"] < 0, "total_prec"] = 0

        df = self.remove_outliers(df)
        df = self.handle_missing_data(df)

        df = df.groupby(["ref_time", "val_time"]).mean().reset_index()
        df.drop(columns = ["lat", "long"], axis = 1, inplace = True)

        return df


    def remove_outliers(self, df, replace=True):
        """Removes or replaces outliers of the weather data.
        
        Parameters:
            - df: weather data.
            - replace: if True, replaces outliers with other values.
        
        Returns:
            - df: cleaned weather data.
        """
        features = list(df.columns)
        for i in ["ref_time", "val_time", "lat", "long", "forecast_horizon"]:
            if i in features:
                features.remove(i)
                
        for column in features:
            df[column] = df.groupby("val_time")[column].transform(lambda group: self.remove_outliers_group(group, replace))

        if not replace:
            df = df.dropna()

        return df
    
    
    def remove_outliers_group(self, group, replace):
        """Replaces outliers within a group object.
        
        Parameters:
            - df: weather data.
            - replace: if True, will replace outliers with other values.
        
        Returns:
            - df: cleaned weather data.
        """
        Q1 = group.quantile(0.25)
        Q3 = group.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if replace:
            mean = group[(group >= lower_bound) & (group <= upper_bound)].mean()
            group = group.where((group >= lower_bound) & (group <= upper_bound), mean)
        else:
            group = group.where((group >= lower_bound) & (group <= upper_bound), np.nan)

        return group


    def handle_missing_data(self, df, performance = False):
        """Replace missing values in the data.
        
        Parameters:
            - df: weather data.
            - performance: if True, will approach a different handling of the missing values with a lower computation time.
        
        Returns:
            - df: cleaned weather data.
        """
        
        if not performance:
            
            df['lat_lon_combination'] = df['lat'].astype(str) + '_' + df['long'].astype(str)
            cols_with_nan = df.columns[df.isna().any()].tolist()
            
            for col in cols_with_nan:
                # group and interpolate each column individually
                df[col] = df.groupby(['forecast_horizon', 'lat_lon_combination'])[col].transform(
                    lambda group: group.interpolate(method='linear') if group.notna().sum() > 1 else group
                )

            df.drop("lat_lon_combination", axis=1, inplace=True)
            mask = df.isna().any(axis=1)
        
            if mask.any():
                # group and aggregate values by the year, month and hour
                grouped_means = df.groupby([df.val_time.dt.year, df.val_time.dt.month, df.val_time.dt.hour])[cols_with_nan].transform('mean')

                # fill NA values with the means
                df[mask] = df[mask].fillna(grouped_means)

        if performance:

            mask = df.isna().any(axis=1)
            # group by year, month, and hour, then calculate the mean
            grouped_means = df.groupby([df.val_time.dt.year, df.val_time.dt.month, df.val_time.dt.hour]).transform('mean')
            # fill missing values using the grouped means
            df[mask] = df[mask].fillna(grouped_means)

        return df
    

    def merge_weather_stations_data(self, weather_data_1, weather_data_2, aggregate_by:str = "val_time", aggregate_by_ref_time_too:bool = True):
        """Merge the weather data from the DWD and NCEP weather stations and perform a resampling and interpolation for 30min intervals.
        
        Parameters:
            - weather_data_1: first weather data set.
            - weather_data_2: second weather data set.
            - aggregate_by: column to to use for the aggregation of the combined data set.
            - aggregate_by_ref_time_too: if True, will aggregate by the reference time of a weather forecast as well.

        Returns:
            - weather_data: combined and aggregated dataset.
        """

        assert aggregate_by in weather_data_1.columns, f"Dimension {aggregate_by} to aggregate by was not found in the first dataset."
        assert aggregate_by in weather_data_2.columns, f"Dimension {aggregate_by} to aggregate by was not found in the second dataset."
        assert "datetime" in str(weather_data_1[aggregate_by].dtype), f"First input's dimension to aggregate by ({aggregate_by}) is not properly formatted to datetime."
        assert "datetime" in str(weather_data_2[aggregate_by].dtype), f"Second input's dimension to aggregate by ({aggregate_by}) is not properly formatted to datetime."

        # merge forecasts from different locations to one aggregated value per reference and valid time
        if aggregate_by_ref_time_too:
            weather_data = pd.concat([weather_data_1, weather_data_2]).groupby(["ref_time", aggregate_by]).mean()
        else:
            weather_data = pd.concat([weather_data_1, weather_data_2]).groupby([aggregate_by]).mean()

        if "lat" in weather_data.columns and "long" in weather_data.columns:
                weather_data = weather_data.drop(["lat", "long"], axis = 1)
        if "ref_time" in weather_data.columns:
            weather_data = weather_data.drop(["ref_time"], axis = 1)

        # merge forecasts on valid time
        # resampling will lead to every 2nd row being empty, thus, an interpolation is required
        weather_data = weather_data.resample("30min", level = 1).mean().interpolate("time")

        return weather_data


    def add_statistical_data(self, df):
        """Add rolled averages, standard deviations, minimum and maximum values.
        
        Parameters:
            - df: data set as a pandas.DataFrame object.
        Returns:
            - df: data set as a pandas.DataFrame object.
        """

        # add the rolling mean/std/min/max of the last data points (each data point represents a 30min period) as features
        for feature in ["temp", "wind_speed", "wind_speed_100", "wind_direction", "wind_direction_100", "solar_down_rad", "cloud_cover", "total_prec"]:
            if feature in df.columns:

                df[f"{feature}_mean_6h"] = df[f"{feature}"].rolling(12).mean()
                df[f"{feature}_std_6h"] = df[f"{feature}"].rolling(12).std()
                df[f"{feature}_min_6h"] = df[f"{feature}"].rolling(12).min()
                df[f"{feature}_max_6h"] = df[f"{feature}"].rolling(12).max()

                df[f"{feature}_mean_3h"] = df[f"{feature}"].rolling(6).mean()
                df[f"{feature}_std_3h"] = df[f"{feature}"].rolling(6).std()
                df[f"{feature}_min_3h"] = df[f"{feature}"].rolling(6).min()
                df[f"{feature}_max_3h"] = df[f"{feature}"].rolling(6).max()

                df[f"{feature}_next_forecast"] = df[f"{feature}"].shift(-1)

        df = df.sort_values("val_time")

        return df


    def add_other_features(self, df):
        """Extract further features from the data.
        
        Parameters:
            - df: data set as a pandas.DataFrame object.
        Returns:
            - df: data set as a pandas.DataFrame object.
        """

        if "wind_speed" in df.columns:
            df["wind_speed_range_3h"] = df["wind_speed_max_3h"] - df["wind_speed_min_3h"]

        if "wind_speed_100" in df.columns:
            df["wind_speed_100_range_3h"] = df["wind_speed_100_max_3h"] - df["wind_speed_100_min_3h"]
            # add the altitude difference in wind speed
            df["wind_speed_altitude_diff"] = df["wind_speed_100"] - df["wind_speed"]

        if "wind_dir" in df.columns:
            df["wind_dir_sin"] = df["wind_dir"].apply(self.convert_wind_directions_to_sin)
            df["wind_dir_cos"] = df["wind_dir"].apply(self.convert_wind_directions_to_cos)
            df.drop(columns = ["wind_dir"], axis = 1, inplace = True)

        if "wind_dir_100" in df.columns:
            df["wind_dir_100_sin"] = df["wind_dir_100"].apply(self.convert_wind_directions_to_sin)
            df["wind_dir_100_cos"] = df["wind_dir_100"].apply(self.convert_wind_directions_to_cos)
            df.drop(columns = ["wind_dir_100"], axis = 1, inplace = True)

        if "solar_down_rad" in df.columns:
            df["solar_down_rad_range_3h"] = df["solar_down_rad_max_3h"] - df["solar_down_rad_min_3h"]
            df["interaction_solar_down_rad_temp"] = df["solar_down_rad"] * df["temp"]

        if "temp" in df.columns:
            df["temp_range_3h"] = df["temp_max_3h"] - df["temp_min_3h"]

        # add the cyclic encoded features to the dataset.
        for col in ["month", "day", "dayofweek", "hour"]:
            time_col_sin = "sin_" + col
            time_col_cos = "cos_" + col

            df[time_col_sin] = df["val_time"].apply(self.get_cycles, args = (0, col))
            df[time_col_cos] = df["val_time"].apply(self.get_cycles, args = (1, col))

        return df
    

    def convert_wind_directions_to_sin(self, data):
        """Convert a degree (°) value into a radiant sine value.
        
        Parameters:
            - data: input data.
        Returns:
            - data: output data.
        
        """
        data = np.deg2rad(data)
        return math.sin(data)


    def convert_wind_directions_to_cos(self, data):
        """Convert a degree (°) value into a radiant cosine value.
        
        Parameters:
            - data: input data.
        Returns:
            - data: output data.
        
        """
        data = np.deg2rad(data)
        return math.cos(data)
        

    def merge_geo_energy_outage_data(self, geo_data, energy_data, left_merge:str = "val_time", right_merge:str = "dtm", deployment:bool = False):
        """Combine the geo data, which is combined with the REMIT data already, with the energy data.
        
        Parameters:
            - geo_data: combined weather and REMIT data.
            - energy_data: energy data.
            - left_merge: column on the weather data to perform the merge.
            - right_merge: column on the energy data to perform the merge.
        Returns:
            - merged_data: combined data.
        """

        geo_data = geo_data.reset_index()

        if deployment:
            main_merge_column = left_merge
            how_to_merge = "left"
            merge_column_to_drop = right_merge
        else:
            main_merge_column = right_merge
            how_to_merge = "right"
            merge_column_to_drop = left_merge

        assert left_merge in geo_data.columns, f"{left_merge} not found in geo data."
        assert right_merge in energy_data.columns, f"{right_merge} not found in energy and outage data."

        if "index" in geo_data.columns:
            geo_data.drop(columns = ["index"], axis = 1, inplace = True)

        assert type(geo_data) == type(pd.DataFrame()), "Geo data is not a pandas dataframe object."
        assert type(energy_data) == type(pd.DataFrame()), "Data with energy and outages is not a pandas dataframe object."
        assert geo_data.shape[0] > 0, "Geo data is empty."
        assert energy_data.shape[0] > 0, "Energy and outage data is empty."
        assert "datetime" in str(geo_data[left_merge].dtype), f"First input's dimension to aggregate by ({right_merge}) is not properly formatted to datetime."
        assert "datetime" in str(energy_data[right_merge].dtype), f"Second input's dimension to aggregate by ({right_merge}) is not properly formatted to datetime."
        merged_data = geo_data.merge(energy_data, left_on = left_merge, right_on = right_merge, how = how_to_merge)
        merged_data = merged_data.bfill()
        merged_data = merged_data.ffill()

        for col in self.non_numerical_columns[:]:
            # check for valid input
            if (col not in geo_data.columns) & (col not in energy_data.columns):
                self.non_numerical_columns.remove(col)
        # perform this only if the data contains non-numerical columns
        if len(self.non_numerical_columns) > 0:
            # we aggregate the numerical and non-numerical columns seperately and then combine them
            # mean does not support non-numerical columns
            aggregated_numerical_cols = merged_data.drop(self.non_numerical_columns, axis = 1).groupby(main_merge_column).mean()
            aggregated_string_cols = merged_data[[*self.non_numerical_columns, main_merge_column]].groupby(main_merge_column).first()
            merged_data = aggregated_numerical_cols.merge(aggregated_string_cols, left_index = True, right_index = True)
        # perform this if the data contains only numerical columns
        else:
            merged_data = merged_data.groupby(main_merge_column).mean()
            if main_merge_column in merged_data.columns:
                merged_data = merged_data.set_index(main_merge_column)

        merged_data = merged_data.drop_duplicates()
        merged_data = merged_data.dropna(axis = 0)

        if merge_column_to_drop in merged_data.columns:
            merged_data.drop(columns = [merge_column_to_drop], axis = 1, inplace = True)

        return merged_data
    

    def add_difference_features(self, data):
        """Add features based on the difference of values between data points.
        
        Parameters:
            - data: input data.
        Returns:
            - data: output data.
        """

        for col in ['rel_hum', 'temp', 'total_precipitation',
                    'wind_direction', 'wind_speed', "wind_speed_100",
                    'cloud_cover', 'solar_down_rad',
                    "unused_capacity_mwp"]:
    
            new_col = col + "_diff"
            if col in data.columns:
                data[new_col] = data[col].diff()

        data.fillna(0, inplace = True)
        data = data[data.index != 0]

        return data
    

    def add_fft_features(self, data, columns_to_fft:list = list()):
        """Perform a Fast Fourier Transformation (FFT) on selected features (defined in columns_to_fft). Apply a FFT for each unique date in the dataset (one FFT per day).
        
        Parameters:
            - data: input data.
            - columns_to_fft: columns to compute the FFT on.
        Returns:
            - data: output data.
        """
        for column in columns_to_fft:
            logging.info(f"Apply FFT on {column}...")
            if column in data.columns:
                for date in data.groupby(data.index.date).max().index:
                    data.loc[data.index.date == date, f"{column}_fft"] = np.abs(np.fft.fft(data.loc[data.index.date == date, f"{column}"]))

        return data
    


class FeatureEngineerer:
    """Performs the feature engineering steps such as train-val-test-splits, onehotencoding non-numerical columns and scaling the data. """
    
    def __init__(self, label:str = "Solar_MWh_credit", labels_to_remove:list = ["Solar_MWh_credit", "Wind_MWh_credit"], 
                 columns_to_ohe:str = list(), train_ratio:float = 0.7, val_ratio:float = 0.2, test_ratio:float = 0.1, scaler_name:str = "standard"):
        assert train_ratio + val_ratio + test_ratio <= 1, "Train, validation and test data ratio can only equal to 1 as a sum."

        self.label = label
        self.labels_to_remove = labels_to_remove
        self.columns_to_ohe = columns_to_ohe
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.scaler_name = scaler_name

    def perform_feature_engineering(self, data, deployment:bool = False, labels_to_remove:list = ["Solar_MWh_credit", "Wind_MWh_credit"]):
        """Main method to call the methods necessary for the feature engineering process.
        
        Parameters:
            - data: output data from the preprocessing steps.
            - deployment: if True, will skip train-val-test-splits and avoid fitting the OneHotEncoder and Scaler.
            - labels_to_remove: list of columns which should be treated as labels. Will be ignored if deployment is set to True.
        Returns:
            None. Will store data instead under self.X_train, self.X_val, self.X_test, self.y_train, self.y_val and self.y_test if deployment is set to False and otherwise under self.deployment_data.
        """

        if deployment:
            self.deployment_data = data.copy()
        else:
            # relevant for plotting methods in model_utils.py
            self.features_after_fe = [*data.drop(labels_to_remove, axis = 1).columns]

            if type(labels_to_remove) != type(list()):
                labels_to_remove = [labels_to_remove]
            self.labels_to_remove = labels_to_remove
            self.columns_to_ohe = self.columns_to_ohe
            self.train_ratio = self.train_ratio
            self.test_ratio = self.test_ratio

            if self.scaler_name.lower() == "standard":
                self.scaler = StandardScaler()
            elif self.scaler_name.lower() == "minmax":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = RobustScaler()

            #self.train_val_test_split(data)

            self.train_val_test_split_hardcoded(data)

        if len(self.columns_to_ohe) > 0:
            self.onehotencode(data, deployment = deployment)

        self.scale(deployment = deployment)


    def train_val_test_split(self, data):
        """Perform a train-validation-test split on given input data.
        
        Parameters:
            - data: input data.
        Returns:
            None.
        """
        ts = TimeSeriesSplit(n_splits = 5)
        # the last 10% of the dataset will be used for the test data set
        X_train_val = data.drop(self.labels_to_remove, axis = 1).iloc[:int(data.shape[0] * (1 - self.test_ratio)), :]
        y_train_val = data[self.label].iloc[:int(data.shape[0] * 0.9)]

        self.X_test = data.drop(self.labels_to_remove, axis = 1).iloc[int(data.shape[0] * (1 - self.test_ratio)):, :]
        self.y_test = data[self.label].iloc[int(data.shape[0] * 0.9):]

        for train_index, val_index in ts.split(X_train_val):
            self.X_train, self.X_val = X_train_val.iloc[train_index, :], X_train_val.iloc[val_index, :]
            self.y_train, self.y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]



    def train_val_test_split_by_year(self, data, test_duration=12, val_duration=12):
        """Perform a train-val-test split by the timespan of a year."""

        data = data.sort_index()

        max_date = data.index.max()
        test_start_date = max_date - timedelta(weeks=52*test_duration/12)
        test_end_date = test_start_date + timedelta(weeks=26)
        val_end_date = test_start_date - timedelta(days=1)
        val_start_date = val_end_date - timedelta(weeks=52*val_duration/12)

        train_data = data[data.index  <= val_start_date]
        val_data = data[(data.index > val_start_date) & (data.index <= val_end_date)]
        test_data = data[(data.index > test_start_date) & (data.index <= test_end_date)]
        
        # Entferne die Spalten, die nicht benötigt werden
        self.X_train = train_data.drop(self.labels_to_remove, axis=1)
        self.y_train = train_data[self.label]

        self.X_val = val_data.drop(self.labels_to_remove, axis=1)
        self.y_val = val_data[self.label]

        self.X_test = test_data.drop(self.labels_to_remove, axis=1)
        self.y_test = test_data[self.label]

    def train_val_test_split_hardcoded(self, data):
        
        train_data = data[data.index.date < date(2023, 1, 1)]
        val_data = data[(data.index.date >= date(2023, 1, 1)) & (data.index.date < date(2023, 7, 1))]
        test_data = data[(data.index.date >= date(2023, 7, 1)) & (data.index.date < date(2024, 1, 1))]

        # Entferne die Spalten, die nicht benötigt werden
        self.X_train = train_data.drop(self.labels_to_remove, axis=1)
        self.y_train = train_data[self.label]

        self.X_val = val_data.drop(self.labels_to_remove, axis=1)
        self.y_val = val_data[self.label]

        self.X_test = test_data.drop(self.labels_to_remove, axis=1)
        self.y_test = test_data[self.label]

    def onehotencode(self, data, deployment:bool = False):
        """Perform a onehotencoding. For the training phase, it will create and fit a OneHotEncoder. During deployment, it will assume a OneHotEncoder has been fitted and will be called.
        
        Parameters:
            - data: input data.
            - deployment: if True, will assume a OneHotEncoder exists and will be used. Otherwise will create and fit a OneHotEncoder.
        Returns:
            - data: output data, if no columns are found to be onehotencoded. Otherwise None.
        """
        
        try:
            true_ohe_columns = list(data.dtypes[data.dtypes == "object"].index)
            for col in true_ohe_columns:
                if col not in self.columns_to_ohe:
                    logging.warning(f"Column {col} was recognized as a numerical feature. Will be converted and ignored for onehotencoding.")
                    data[col] = pd.to_numeric(data[col])     
        except Exception as e:
            logging.warning(e)
            self.columns_to_ohe = []
        
        if len(self.columns_to_ohe) > 0:
            if deployment == False:
                self.ohe = OneHotEncoder(sparse_output = False, handle_unknown = "infrequent_if_exist")
                self.X_train[self.ohe.get_feature_names_out()] = self.ohe.fit_transform(self.X_train[self.columns_to_ohe])
                self.X_train.drop(columns = self.columns_to_ohe, axis = 1, inplace = True)
                self.X_train = self.X_train.reindex(sorted(self.X_train.columns), axis=1)

                # store the adjusted feature names after fitting the onehotencoder
                self.features_after_fe = [*data.drop(columns = [*self.labels_to_remove, *self.columns_to_ohe], axis = 1).columns, *self.ohe.get_feature_names_out()]

                self.X_val[self.ohe.get_feature_names_out()] = self.ohe.transform(self.X_val[self.columns_to_ohe])
                self.X_val.drop(columns = self.columns_to_ohe, axis = 1, inplace = True)
                self.X_val = self.X_val.reindex(sorted(self.X_val.columns), axis=1)

                self.X_test[self.ohe.get_feature_names_out()] = self.ohe.transform(self.X_test[self.columns_to_ohe])
                self.X_test.drop(columns = self.columns_to_ohe, axis = 1, inplace = True)
                self.X_test = self.X_test.reindex(sorted(self.X_test.columns), axis=1)
            else:
                self.deployment_data = data.copy()
                self.deployment_data[self.ohe.get_feature_names_out()] = self.ohe.transform(self.deployment_data[self.columns_to_ohe])
                self.deployment_data.drop(columns = self.columns_to_ohe, axis = 1, inplace = True)
                self.deployment_data = self.deployment_data.reindex(sorted(self.deployment_data.columns), axis=1)
        else:
            logging.info("No features found to onehotencode.")
            return data

    
    def scale(self, deployment:bool = False):
        """Scale the data. For the training phase, it will create and fit a Scaler. During deployment, it will assume a Scaler has been fitted and will be called.
        
        Parameters:
            - data: input data.
            - deployment: if True, will assume a Scaler exists and will be used. Otherwise will create and fit a Scaler.
        Returns:
            - None.
        """
        if deployment == False:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_val = self.scaler.transform(self.X_val)
            self.X_test = self.scaler.transform(self.X_test)
        else:
            self.deployment_data = self.scaler.transform(self.deployment_data)
