import numpy as np
import dask.dataframe as dd
import math
import xarray
import pandas as pd
import os
import json
from datetime import datetime
from math import sin, cos, pi
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit


pd.set_option('future.no_silent_downcasting', True)


class FileExtractor:
    def __init__(self):
        pass


    def extract_json_files(self, path):

        json_data = None

        for file in os.listdir(path):
            with open(path + "/" + file) as f:
                d = json.load(f)
                d = pd.json_normalize(d)
            if type(json_data) == type(None):
                json_data = d
            else:
                json_data = pd.concat([json_data, d])

        return json_data
    

    def combine_files(self, path, file_name, file_format:str=".nc"):
        """Combine files of the same dataset with different timestamps. Converts them into a dataframe."""
        df = None
        df_new = None

        all_dfs = list()

        for file in os.listdir(path):
            if file.endswith(file_format) and file_name.lower() in file.lower(): 
                if file_format == ".nc":
                    ds = xarray.open_dataset(path + "/" + file)
                    df_new = ds.to_dataframe()

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

                elif file_format == ".csv":
                    df = pd.read_csv(path + "/" + file)
                    all_dfs.append(df)

        df_total = pd.concat(all_dfs)

        return df_total
    


class Preprocessing:

    def __init__(self, for_deployment:bool = False):
        self.for_deployment = for_deployment
        self.irrelevant_features = list()


    def perform_preprocessing_pipeline(self, geo_data_dict:dict, 
                                       aggregate_by:str = "val_time", aggregate_by_ref_time_too:bool = True,
                                       fft:bool = False, columns_to_fft:list = ["temp_diff", "solar_down_rad_diff", "wind_speed_diff", "wind_speed_100_diff"],
                                       deployment:bool = True, energy_data_dict:dict = dict(), left_merge:str = "val_time", right_merge:str = "dtm"):
        weather_data = list()
        extractor = FileExtractor()
        for file_name_pattern, file_path in geo_data_dict.items():
            weather_data.append(extractor.combine_files(file_path, file_name_pattern, ".nc"))

        for index in range(0, len(weather_data)):
            weather_data[index] = self.preprocess_geo_data(weather_data[index])

        df = self.merge_weather_stations_data(weather_data_1 = weather_data[0], weather_data_2 = weather_data[1], aggregate_by = aggregate_by, aggregate_by_ref_time_too = aggregate_by_ref_time_too)
        df = self.add_difference_features(df)

        if fft:
            df = self.add_fft_features(df, columns_to_fft = columns_to_fft)

        if deployment == False:
            key = next(iter(energy_data_dict))
            energy_df = extractor.combine_files(energy_data_dict[key], key, ".csv")
            energy_df = self.preprocess_energy_data(energy_df)
            df = self.merge_geo_energy_outage_data(df, energy_df, left_merge = left_merge, right_merge = right_merge)

        return df


    def cyclic_sin(self, n):
        theta = 2 * pi * n
        return sin(theta)


    def cyclic_cos(self, n):
        theta = 2 * pi * n
        return cos(theta)


    def get_cycles(self, d, sin_or_cos:int, time_information:str):
        '''
        Get the cyclic properties of a datetime,
        represented as points on the unit circle.
        Arguments
        ---------
        d : datetime object
        Returns
        -------
        dictionary of sine and cosine tuples

        source: https://medium.com/@dan.allison/how-to-encode-the-cyclic-properties-of-time-with-python-6f4971d245c0
        '''
        month = d.month - 1
        day = d.day - 1

        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        if sin_or_cos == 0:
            if time_information.lower() == "month":
                return self.cyclic_sin(month / 12)
            elif time_information.lower() == "day":
                return self.cyclic_sin(day / days_in_month[month])
            elif time_information.lower() == "dayofweek":
                return self.cyclic_sin(d.weekday() / 7)
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
    

    def preprocess_outage_data(self, json_data):
        # dismissed outages are not relevant
        json_data = json_data[json_data.eventStatus != "Dismissed"]
        # drop those columns since they provide no value as features
        json_data.drop(columns = ["id", "outageProfile", "assetId", "affectedUnitEIC", "dataset", "eventStatus", "cause", "publishTime", "createdTime", "relatedInformation", "revisionNumber", "mrid"], axis = 1, inplace = True)

        # drop string value columns with only one unique value. If it´s not used for the deployment, we are in the training phase instead. Search through every column and save their names so during deployment, they will get removed instantly next time.
        # if the preprocessing is used for the deployment, we already know which features provide no information. During deployment we might have to use interference on a single row, so just searching for columns with one unique value can only work
        # during the preprocessing of training data.
        if self.for_deployment == False:
            for col in json_data.columns:
                if json_data[col].nunique() == 1:
                        try:
                            json_data.drop(columns = col, inplace = True, axis = 1)
                            self.irrelevant_features.append(col)
                        except:
                            continue
        else:
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

        outages_df.drop(columns = ["eventStartTime", "eventEndTime"], axis = 1, inplace = True)

        return outages_df
    

    def preprocess_energy_data(self, df_energy):
        # convert the datetime information to the right format
        df_energy["dtm"] = pd.to_datetime(df_energy["dtm"])
        df_energy = df_energy.sort_values("dtm")

        # Group by year, month, and hour, then calculate the mean
        grouped_means = df_energy.groupby([df_energy.dtm.dt.year, df_energy.dtm.dt.month, df_energy.dtm.dt.hour]).transform('mean')

        # Fill missing values in df_energy with the corresponding grouped means
        df_energy = df_energy.fillna(grouped_means)

        # convert MW to MWh (30min periods --> multiply by 0.5)
        for col in ["Solar_MW", "Wind_MW"]:
            df_energy[col] = 0.5 * df_energy[col]

        df_energy["Wind_MWh_credit"] = df_energy["Wind_MW"] - df_energy["boa_MWh"]
        df_energy.rename(columns = {"Solar_MW": "Solar_MWh_credit", "Wind_MW": "Wind_MWh"}, inplace = True)
        df_energy.drop(["Wind_MWh"], axis = 1, inplace = True)

        #df_energy["unused_Solar_capacity_mwp"] = df_energy["Solar_installedcapacity_mwp"] = df_energy["Solar_capacity_mwp"]

        df_energy = df_energy.drop_duplicates()
        df_energy = df_energy[["dtm", "Wind_MWh_credit", "Solar_MWh_credit"]]

        return df_energy


    def merge_energy_with_outages(self, energy_data, outage_data):
        energy_with_outages = pd.merge_asof(left = energy_data, right = outage_data, left_on = "dtm", right_on = outage_data.sort_index().index, direction = "nearest", tolerance = pd.Timedelta("30m"))

        # the merge will result in NA values (since outages are not present all the time), thus they have to be filled with replacement values
        energy_with_outages[["affectedUnit", "unavailabilityType"]] = energy_with_outages[["affectedUnit", "unavailabilityType"]].fillna("None")
        energy_with_outages[["unavailableCapacity", "hoursSinceOutage", "hoursUntilOutageEnd"]] = energy_with_outages[["unavailableCapacity", "hoursSinceOutage", "hoursUntilOutageEnd"]].fillna(0)
        energy_with_outages[["availableCapacity"]] = energy_with_outages[["availableCapacity"]].fillna(400)
        energy_with_outages["outage"] = energy_with_outages["outage"].fillna(False).astype(int)

        return energy_with_outages


    def preprocess_geo_data(self, df):
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
    

    # def remove_outliers(self, df):
    #     # remove outliers
    #     df["year"] = df["val_time"].dt.year
    #     df["month"] = df["val_time"].dt.month
    #     df["day"] = df["val_time"].dt.day
    #     df["hour"] = df["val_time"].dt.hour

    #     features = list(df.columns)
    #     for i in ["ref_time", "val_time", "lat", "long", "month", "day", "hour", "forecast_horizon"]:
    #         if i in features:
    #             features.remove(i)

    #     for column in features:
    #         q1 = df[column].quantile(0.25)
    #         q3 = df[column].quantile(0.75)
    #         iqr = q3 - q1

    #         lower_bound = q1 - 1.5*iqr
    #         upper_bound = q3 + 1.5*iqr

    #         df[column] = df[column].where((df[column] >= lower_bound) | (df[column] <= upper_bound),
    #                                         other=np.nan)
    #         group_means = df.groupby(["year", "month", "day", "hour"])[column].transform("mean")
    #         df[column] = df[column].fillna(group_means)
            
    #     df = df.drop(["year", "month", "day", "hour"], axis=1)

    #     return df


    def remove_outliers(self, df, replace=True):
        """Removes or replaces outliers of the weather data."""
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
        """Replaces outliers within a group object."""
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
        
        if not performance:
            
            df['lat_lon_combination'] = df['lat'].astype(str) + '_' + df['long'].astype(str)
            cols_with_nan = df.columns[df.isna().any()].tolist()
            
            for col in cols_with_nan:
                # Group and interpolate each column individually
                df[col] = df.groupby(['forecast_horizon', 'lat_lon_combination'])[col].transform(
                    lambda group: group.interpolate(method='linear') if group.notna().sum() > 1 else group
                )

            df.drop("lat_lon_combination", axis=1, inplace=True)
            mask = df.isna().any(axis=1)
        
            if mask.any():
                # Gruppiere nach Jahr, Monat und Stunde und berechne den Mittelwert für numerische Spalten
                grouped_means = df.groupby([df.val_time.dt.year, df.val_time.dt.month, df.val_time.dt.hour])[cols_with_nan].transform('mean')

                # Fülle die verbliebenen NaN-Werte mit den berechneten Mittelwerten
                df[mask] = df[mask].fillna(grouped_means)

        if performance:

            mask = df.isna().any(axis=1)
            # Group by year, month, and hour, then calculate the mean
            grouped_means = df.groupby([df.val_time.dt.year, df.val_time.dt.month, df.val_time.dt.hour]).transform('mean')
            # Fill missing values using the grouped means
            df[mask] = df[mask].fillna(grouped_means)

        return df
    

    def merge_weather_stations_data(self, weather_data_1, weather_data_2, aggregate_by:str = "val_time", aggregate_by_ref_time_too:bool = True):
        """Merge the weather data from the DWD and NCEP weather stations."""

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
        # add the rolling mean/std/min/max of the last 48 data points (each data point represents a 30min period) as a feature

        for feature in ["temp", "wind_speed", "wind_speed_100", "wind_direction", "wind_direction_100", "solar_down_rad", "cloud_cover", "total_prec"]:
            if feature in df.columns:
                df[f"{feature}_mean"] = df[f"{feature}"].rolling(48).mean()
                df[f"{feature}_std"] = df[f"{feature}"].rolling(48).std()
                df[f"{feature}_min"] = df[f"{feature}"].rolling(48).min()
                df[f"{feature}_max"] = df[f"{feature}"].rolling(48).max()

        df = df.sort_values("val_time")

        return df


    def add_other_features(self, df):

        if "wind_speed" in df.columns:
            # convert wind speed from m/s to km/h
            df["wind_speed"] = df["wind_speed"] * 3.6
            df["wind_speed_range"] = df["wind_speed_max"] - df["wind_speed_min"]

        if "wind_speed_100" in df.columns:
            # convert wind speed from m/s to km/h
            df["wind_speed_100"] = df["wind_speed_100"] * 3.6
            df["wind_speed_100_range"] = df["wind_speed_100_max"] - df["wind_speed_100_min"]
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
            df["solar_down_rad_range"] = df["solar_down_rad_max"] - df["solar_down_rad_min"]
            df["interaction_solar_down_rad_temp"] = df["solar_down_rad"] * df["temp"]

        if "temp" in df.columns:
            df["temp_range"] = df["temp_max"] - df["temp_min"]


        for col in ["month", "day", "dayofweek", "hour"]:
            time_col_sin = "sin_" + col
            time_col_cos = "cos_" + col

            df[time_col_sin] = df["val_time"].apply(self.get_cycles, args = (0, col))
            df[time_col_cos] = df["val_time"].apply(self.get_cycles, args = (1, col))

        return df
    

    def convert_wind_directions_to_sin(self, data):
        data = np.deg2rad(data)
        return math.sin(data)


    def convert_wind_directions_to_cos(self, data):
        data = np.deg2rad(data)
        return math.cos(data)
        

    def merge_geo_energy_outage_data(self, geo_data, energy_outage_data, left_merge:str = "val_time", right_merge:str = "dtm"):
        """Combine the geo data from the weather stations with the energy and outage data (CSV and JSON files combined)."""

        geo_data = geo_data.reset_index()

        assert left_merge in geo_data.columns, f"{left_merge} not found in geo data."
        assert right_merge in energy_outage_data.columns, f"{right_merge} not found in energy and outage data."

        if "index" in geo_data.columns:
            geo_data.drop(columns = ["index"], axis = 1, inplace = True)

        assert type(geo_data) == type(pd.DataFrame()), "Geo data is not a pandas dataframe object."
        assert type(energy_outage_data) == type(pd.DataFrame()), "Data with energy and outages is not a pandas dataframe object."
        assert geo_data.shape[0] > 0, "Geo data is empty."
        assert energy_outage_data.shape[0] > 0, "Energy and outage data is empty."
        assert "datetime" in str(geo_data[left_merge].dtype), f"First input's dimension to aggregate by ({right_merge}) is not properly formatted to datetime."
        assert "datetime" in str(energy_outage_data[right_merge].dtype), f"Second input's dimension to aggregate by ({right_merge}) is not properly formatted to datetime."

        merged_data = geo_data.merge(energy_outage_data, left_on = left_merge, right_on = right_merge, how = "right")
        merged_data = merged_data.drop_duplicates().groupby(right_merge).mean()
        merged_data = merged_data.dropna(axis = 0)

        if left_merge in merged_data.columns:
            merged_data.drop(columns = [left_merge], axis = 1, inplace = True)

        return merged_data
    

    def add_difference_features(self, data):
        """Add features based on the difference of values between data points."""

        for col in ['rel_hum', 'temp', 'total_precipitation',
                    'wind_direction', 'wind_speed', "wind_speed_100",
                    'cloud_cover', 'solar_down_rad']:
    
            new_col = col + "_diff"
            if col in data.columns:
                data[new_col] = data[col].diff()

        data.fillna(0, inplace = True)
        data = data[data.index != 0]

        return data
    

    def add_fft_features(self, data, columns_to_fft:list = ["temp_diff", "solar_down_rad_diff", "wind_speed_diff", "wind_speed_100_diff"]):
        """Perform a Fast Fourier Transformation (FFT) on selected features (defined in columns_to_fft). Apply a FFT for each unique date in the dataset (one FFT per day)."""
        for column in columns_to_fft:
            print(f"Apply FFT on {column}...")
            if column in data.columns:
                for date in data.groupby(data.index.date).max().index:
                    data.loc[data.index.date == date, f"{column}_fft"] = np.abs(np.fft.fft(data.loc[data.index.date == date, f"{column}"]))

        return data
    


class FeatureEngineerer:
    def __init__(self, data, label:str = "Solar_MWh_credit", labels_to_remove:list = ["Solar_MWh_credit", "Wind_MWh_credit"], 
                 columns_to_ohe:list = list(), train_ratio:float = 0.7, val_ratio:float = 0.2, test_ratio:float = 0.1, scaler:str = "standard"):
        assert train_ratio + val_ratio + test_ratio <= 1, "Train, validation and test data ratio can only equal to 1 as a sum."

        self.label = label
        if type(labels_to_remove) != type(list()):
            labels_to_remove = [labels_to_remove]
        self.labels_to_remove = labels_to_remove
        self.columns_to_ohe = columns_to_ohe
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        if scaler.lower() == "standard":
            self.scaler = StandardScaler()
        elif scaler.lower() == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()

        self.train_val_test_split(data)

        if len(self.columns_to_ohe) > 0:
            self.onehotencode(data)

        self.scale()


    def train_val_test_split(self, data):
        ts = TimeSeriesSplit(n_splits = 5)
        # the last 10% of the dataset will be used for the test data set
        X_train_val = data.drop(self.labels_to_remove, axis = 1).iloc[:int(data.shape[0] * (1 - self.test_ratio)), :]
        y_train_val = data[self.label].iloc[:int(data.shape[0] * 0.9)]

        self.X_test = data.drop(self.labels_to_remove, axis = 1).iloc[int(data.shape[0] * (1 - self.test_ratio)):, :]
        self.y_test = data[self.label].iloc[int(data.shape[0] * 0.9):]

        for train_index, val_index in ts.split(X_train_val):
            self.X_train, self.X_val = X_train_val.iloc[train_index, :], X_train_val.iloc[val_index, :]
            self.y_train, self.y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]


    def onehotencode(self, data, columns_to_ohe):
        # check for valid input for columns to onehotencode
        if len(columns_to_ohe) > 0:
            for column in columns_to_ohe:
                if column not in data.columns:
                    self.columns_to_ohe.remove(column)

        if len(columns_to_ohe) > 1:
            self.ohe = OneHotEncoder()
            self.X_train[columns_to_ohe] = self.ohe.fit_transform(self.X_train[columns_to_ohe])
            self.X_test[columns_to_ohe] = self.ohe.transform(self.X_test[columns_to_ohe])        
            self.X_train[columns_to_ohe] = self.scaler.fit_transform(self.X_train[columns_to_ohe])
            self.X_test[columns_to_ohe] = self.scaler.transform(self.X_test[columns_to_ohe])
        else:
            print("No features found to onehotencode.")

    
    def scale(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

