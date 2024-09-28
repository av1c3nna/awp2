import numpy as np
import dask.dataframe as dd
import math
import xarray
import pandas as pd
import os
import json
from datetime import datetime
from math import sin, cos, pi


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
        df = "a"
        df_new = "a"

        for file in os.listdir(path):
            if file.endswith(file_format) and file_name.lower() in file.lower(): 
                print(file)
                if file_format == ".nc":
                    ds = xarray.open_dataset(path + "/" + file)

                    if type(df) == type(str()):
                        df = ds.to_dataframe()
                    else:
                        df_new = ds.to_dataframe()
                elif file_format == ".csv":
                    if type(df) == type(str()):
                        df = pd.read_csv(path + "/" + file)
                    else:
                        df_new = pd.read_csv(path + "/" + file)

                if type(df_new) != type(str()):
                    df_total = pd.concat([df, df_new])

        return df_total
    

class Preprocessing:

    def __init__(self):
        pass


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
        json_data.drop(columns = ["id", "outageProfile", "assetId", "affectedUnitEIC", "dataset", "eventStatus", "cause", "publishTime", "createdTime", "relatedInformation", "revisionNumber"], axis = 1, inplace = True)

        # drop string value columns with only one unique value
        for col in json_data.columns:
            if json_data[col].nunique() == 1 and json_data[col].dtypes == "O":
                try:
                    json_data.drop(columns = col, inplace = True, axis = 1)
                except:
                    continue

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
        outages_df["mrid"] = outages_df["mrid"].str.split("-").str[-1]

        outages_df.drop(columns = ["eventStartTime", "eventEndTime"], axis = 1, inplace = True)

        return outages_df
    

    def preprocess_energy_data(self, df_energy):
        # convert the datetime information to the right format
        df_energy["dtm"] = pd.to_datetime(df_energy["dtm"])

        # Group by year, month, and hour, then calculate the mean
        grouped_means = df_energy.groupby([df_energy.dtm.dt.year, df_energy.dtm.dt.month, df_energy.dtm.dt.hour]).transform('mean')

        # Fill missing values in df_energy with the corresponding grouped means
        df_energy = df_energy.fillna(grouped_means)

        # convert MW to MWh (30min periods --> multiply by 0.5)
        for col in ["Solar_MW", "Wind_MW"]:
            df_energy[col] = 0.5 * df_energy[col]

        df_energy["Wind_MWh_credit"] = df_energy["Wind_MW"] - df_energy["boa_MWh"]
        df_energy.rename(columns = {"Solar_MW": "Solar_MWh_credit", "Wind_MW": "Wind_MWh"}, inplace = True)

        df_energy["unused_Solar_capacity_mwp"] = df_energy["Solar_installedcapacity_mwp"] = df_energy["Solar_capacity_mwp"]

        for col in ["month", "day", "dayofweek", "hour", "minute"]:
            time_col_sin = "sin_" + col
            time_col_cos = "cos_" + col

            df_energy[time_col_sin] = df_energy.dtm.apply(self.get_cycles, args = (0, col))
            df_energy[time_col_cos] = df_energy.dtm.apply(self.get_cycles, args = (1, col))

        return df_energy


    def merge_energy_with_outages(self, energy_data, outage_data):
        energy_with_outages = pd.merge_asof(left = energy_data, right = outage_data, left_on = "dtm", right_on = outage_data.sort_index().index, direction = "nearest", tolerance = pd.Timedelta("30m"))

        # the merge will result in NA values (since outages are not present all the time), thus they have to be filled with replacement values
        energy_with_outages[["mrid", "affectedUnit", "unavailabilityType"]] = energy_with_outages[["mrid", "affectedUnit", "unavailabilityType"]].fillna("None")
        energy_with_outages[["unavailableCapacity", "hoursSinceOutage", "hoursUntilOutageEnd"]] = energy_with_outages[["unavailableCapacity", "hoursSinceOutage", "hoursUntilOutageEnd"]].fillna(0)
        energy_with_outages[["normalCapacity", "availableCapacity"]] = energy_with_outages[["normalCapacity", "availableCapacity"]].fillna(400)
        energy_with_outages["outage"] = energy_with_outages["outage"].fillna(False).astype(int)

        return energy_with_outages


    def preprocess_geo_data(self, df):
        df = self.clean_geo_data(df)
        df = self.handle_missing_data(df)
        df = self.add_statistical_data(df)
        df = self.add_other_features(df)

        if "index" in df.columns:
            df.drop(["index"], axis = 1, inplace = True)

        return df


    def clean_geo_data(self, df):
        # reset the index (reference_time, valid_time, latitude, longitude)
        df.reset_index(inplace = True)
        # rename the columns properly
        df = df.rename(columns = {"level_0": "reference_time", "level_1": "valid_time"})
        if "index" in df.columns:
            df.drop(columns = ["index"], axis = 1, inplace = True)
        # convert the datetime information to the right format
        df["reference_time"] = pd.to_datetime(df.reference_time).dt.tz_localize("UTC")
        df["valid_time"] = df["reference_time"] + pd.to_timedelta(df["valid_time"], unit = "hour")
        # remove forecasts which extend beyond the day ahead, since they will be outdated the next day anyway
        df = df[(df["valid_time"] - df["reference_time"]).div(pd.Timedelta("1h")) < 50]
        # some data points have a miscalculation at their coordinates (e.g. ncep_gfs_demand). The actual coordinates can be identified by their value of the feature "point"
        df.loc[df.longitude > 90, "longitude"] -= 360
        df.loc[df.longitude < -90, "longitude"] += 360

        # there are anomalies of the solar down radiation being above 1000 in a short time period. The maximum threshold is to be believed to be about 1000 W/m^2
        # source: https://www.researchgate.net/post/Are_there_minimum_and_maximum_threshold_of_solar_irradiance
        if "SolarDownwardRadiation" in df.columns:
            df = df[df["SolarDownwardRadiation"] <= 1000]
            df.loc[df["SolarDownwardRadiation"] < 0, "SolarDownwardRadiation"] = 0
            # convert W/m^2 to kW/km^2
            # df["SolarDownwardRadiation"] = df["SolarDownwardRadiation"] * 1000

        if "RelativeHumidity" in df.columns:
            df.loc[df["RelativeHumidity"] > 100, "RelativeHumidity"] = 100
            df.loc[df["RelativeHumidity"] < 0, "RelativeHumidity"] = 0
            
        if "TotalPrecipitation" in df.columns:
            df.loc[df["TotalPrecipitation"] < 0, "TotalPrecipitation"] = 0

        df = df.groupby(["reference_time", "valid_time"]).mean().reset_index()
        df.drop(columns = ["latitude", "longitude"], axis = 1, inplace = True)

        if "point" in df.columns:
            df.drop(["point"], axis = 1, inplace = True)

        return df


    def handle_missing_data(self, df):
        # Fill missing values by using the mean of other data points at a similiar time (same year, month and hour)
        mask = df.isna().any(axis=1)
        # Group by year, month, and hour, then calculate the mean
        grouped_means = df.groupby([df.valid_time.dt.year, df.valid_time.dt.month, df.valid_time.dt.hour]).transform('mean')
        # Fill missing values using the grouped means
        df[mask] = df[mask].fillna(grouped_means)
        
        return df


    def add_statistical_data(self, df):

        df_std = df.set_index("valid_time").resample("24h").std().sort_values("valid_time").drop(["reference_time"], axis = 1)
        df_mean = df.set_index("valid_time").resample("24h").mean().sort_values("valid_time").drop(["reference_time"], axis = 1)
        df_min = df.set_index("valid_time").resample("24h").min().sort_values("valid_time").drop(["reference_time"], axis = 1)
        df_max = df.set_index("valid_time").resample("24h").max().sort_values("valid_time").drop(["reference_time"], axis = 1)

        df_std.columns = [x + "_std" for x in df_std.columns]
        df_mean.columns = [x + "_mean" for x in df_mean.columns]
        df_min.columns = [x + "_min" for x in df_min.columns]
        df_max.columns = [x + "_max" for x in df_max.columns]

        df = df.sort_values("valid_time")

        for data in [df_std, df_mean, df_min, df_max]:
            df = pd.merge(df, data, on = "valid_time", how = "left")

        # Convert pandas dataframe to dask dataframe to enable a faster operation
        ddf = dd.from_pandas(df, npartitions=10)
        # fill the missing values, since the aggregations were computed on rows whose datetime values were at 12am.
        df = ddf.ffill().compute()

        return df


    def add_other_features(self, df):

        # bins = [0, 1, 5, 11, 19, 28, 38, 49, 61, 74, 88, 102, 117, float("inf")]
        # labels = [x for x in range(0,13,1)]

        if "WindSpeed" in df.columns:
            # convert wind speed from m/s to km/h
            df["WindSpeed"] = df["WindSpeed"] * 3.6
            df["WindSpeed_range"] = df["WindSpeed_max"] - df["WindSpeed_min"]
            # add the beaufort scala for the wind speed
            # df["BeaufortScale"] = pd.cut(df["WindSpeed"], bins = bins, labels = labels, right = False)
            # df["BeaufortScale"] = pd.to_numeric(df["BeaufortScale"])

        if "WindSpeed:100" in df.columns:
            # convert wind speed from m/s to km/h
            df["WindSpeed:100"] = df["WindSpeed:100"] * 3.6
            df["WindSpeed:100_range"] = df["WindSpeed:100_max"] - df["WindSpeed:100_min"]
            # add the beaufort scala for the wind speed
            # df["BeaufortScale:100"] = pd.cut(df["WindSpeed:100"], bins = bins, labels = labels, right = False)
            # df["BeaufortScale:100"] = pd.to_numeric(df["BeaufortScale:100"])
            # add the altitude difference in wind speed
            df["WindSpeedAltitudeDiff"] = df["WindSpeed:100"] - df["WindSpeed"]

        if "WindDirection" in df.columns:
            df["WindDirection_sin"] = df["WindDirection"].apply(self.convert_wind_directions_to_sin)
            df["WindDirection_cos"] = df["WindDirection"].apply(self.convert_wind_directions_to_cos)
            df.drop(columns = ["WindDirection"], axis = 1, inplace = True)

        if "WindDirection:100" in df.columns:
            df["WindDirection:100_sin"] = df["WindDirection:100"].apply(self.convert_wind_directions_to_sin)
            df["WindDirection:100_cos"] = df["WindDirection:100"].apply(self.convert_wind_directions_to_cos)
            df.drop(columns = ["WindDirection:100"], axis = 1, inplace = True)

        if "SolarDownwardRadiation" in df.columns:
            df["SolarDownwardRadiation_range"] = df["SolarDownwardRadiation_max"] - df["SolarDownwardRadiation_min"]
            df["Interaction_SolarDownwardRadiation_Temperature"] = df["SolarDownwardRadiation"] * df["Temperature"]

        if "Temperature" in df.columns:
            df["Temperature_range"] = df["Temperature_max"] - df["Temperature_min"]

        return df
    

    def convert_wind_directions_to_sin(self, data):
        data = np.deg2rad(data)
        return math.sin(data)


    def convert_wind_directions_to_cos(self, data):
        data = np.deg2rad(data)
        return math.cos(data)
    

    def merge_weather_stations_data(self, weather_data_1, weather_data_2, aggregate_by:str = "valid_time"):
        """Merge the weather data from the DFD and NCEP weather stations."""

        assert(aggregate_by in weather_data_1.columns, f"Dimension {aggregate_by} to aggregate by was not found in the first dataset.")
        assert(aggregate_by in weather_data_2.columns, f"Dimension {aggregate_by} to aggregate by was not found in the second dataset.")
        assert("datetime" in str(weather_data_1[aggregate_by].dtype), f"First input's dimension to aggregate by ({aggregate_by}) is not properly formatted to datetime.")
        assert("datetime" in str(weather_data_2[aggregate_by].dtype), f"Second input's dimension to aggregate by ({aggregate_by}) is not properly formatted to datetime.")

        weather_data = pd.concat([weather_data_1, weather_data_2]).groupby([aggregate_by]).mean()

        if "reference_time" in weather_data.columns:
            return weather_data.drop(["reference_time"], axis = 1)
        else:
            return weather_data
        

    def merge_geo_energy_outage_data(self, geo_data, energy_outage_data, left_merge:str = "valid_time", right_merge:str = "dtm"):
        """Combine the geo data from the weather stations with the energy and outage data (CSV and JSON files combined)."""

        assert(left_merge in geo_data.columns, f"{left_merge} not found in geo data.")
        assert(right_merge in energy_outage_data.columns, f"{right_merge} not found in energy and outage data.")

        if left_merge in geo_data.index.name:
            geo_data = geo_data.reset_index()

        assert(type(geo_data) == type(pd.DataFrame()), "Geo data is not a pandas dataframe object.")
        assert(type(energy_outage_data) == type(pd.DataFrame()), "Data with energy and outages is not a pandas dataframe object.")
        assert(geo_data.shape[0] > 0, "Geo data is empty.")
        assert(energy_outage_data.shape[0] > 0, "Energy and outage data is empty.")
        assert("datetime" in str(geo_data[left_merge].dtype), f"First input's dimension to aggregate by ({right_merge}) is not properly formatted to datetime.")
        assert("datetime" in str(energy_outage_data[right_merge].dtype), f"Second input's dimension to aggregate by ({right_merge}) is not properly formatted to datetime.")

        merged_data = pd.merge_asof(geo_data.sort_values(left_merge), energy_outage_data, left_on = left_merge, right_on = right_merge, direction = "nearest", tolerance = pd.Timedelta("15min"))
        merged_data = merged_data.dropna(axis = 0)
        merged_data.set_index(right_merge, inplace = True)

        if left_merge in merged_data.columns:
            merged_data.drop(columns = [left_merge], axis = 1, inplace = True)

        return merged_data
    

    def add_difference_features(self, data):
        """Add features based on the difference of values between data points."""

        for col in ['RelativeHumidity', 'Temperature', 'TotalPrecipitation',
                    'WindDirection', 'WindSpeed', 'MIP',
                    'SS_Price', 'boa_MWh', 'DA_Price', 'normalCapacity',
                    'availableCapacity', 'unavailableCapacity',
                    'CloudCover', 'SolarDownwardRadiation', 'Temperature']:
    
            new_col = col + "_diff"
            if col in data.columns:
                data[new_col] = data[col].diff()

        data.fillna(0, inplace = True)
        data = data[data.index != 0]

        return data

