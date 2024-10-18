# Imports
import comp_utils
from Preprocessing import Preprocessing, FeatureEngineerer
import pandas as pd
from model_utils import *
from datetime import datetime, timedelta
import os
import logging
import copy

logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)


class AutoSubmitter:
    """
    This class automates the process of fetching, preprocessing, and predicting energy production 
    (both wind and solar) for the next day. It is executed once daily to generate the predictions 
    and prepare them for submission to the Rebase API.
    """
    def __init__(self, hornsea_model, pes_model):
        """
        Initializes the AutoSubmitter with models and API client.
        """
        logger.info("Initializing AutoSubmitter with models and API client...")

        self.hornsea_model = hornsea_model
        self.pes_model = pes_model
        self.hornsea_data = {}
        self.pes_data = {}
        self.predictions = {}
        self._saved_state = None

        try:
            api_key = open("DanMaLeo_key.txt").read()
            self.rebase_api_client = comp_utils.RebaseAPI(api_key=api_key)
            logger.info("Successfully initialized Auto Submitter.")
        except FileNotFoundError:
            logger.error("API key file not found. Please check if 'DanMaLeo_key.txt' exists.")
        except Exception as e:
            logger.error(f"Failed to initialize RebaseAPI client: {e}")


    def fetch_data(self):
        """
        Fetches current weather data for Hornsea 1 and PES Region 10, and stores energy data locally.
        """
        logger.info("Fetching data...")
        self._saved_state = copy.deepcopy(self)
        
        # Create directory to save fetched data
        today = datetime.today().strftime('%Y-%m-%d')
        base_dir = os.path.join("fetched_data", today)
        
        # Create path if not exists
        def ensure_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)
                logger.info(f"Created directory: {path}")
            return path

        # Fetch weather data for Hornsea 1
        hornsea_dir = ensure_dir(os.path.join(base_dir, "hornsea"))
        hornsea_dwd_fetched = self._fetch_data(power="hornsea", model="DWD_ICON-EU")
        logger.info("Successfully fetched DWD weather data for Hornsea 1.")
        hornsea_gfs_fetched = self._fetch_data(power="hornsea", model="NCEP_GFS")
        logger.info("Successfully fetched NCEP GFS weather data for Hornsea 1.")
        self.hornsea_data.update({"dwd_fetched": hornsea_dwd_fetched, "gfs_fetched": hornsea_gfs_fetched})

        # Locally save fetched Hoensea 1 data (long-term)
        hornsea_dwd_fetched.to_csv(os.path.join(hornsea_dir, "dwd_fetched.csv"))
        hornsea_gfs_fetched.to_csv(os.path.join(hornsea_dir, "gfs_fetched.csv"))
        
        # Fetch weather data for PES Region 10
        pes_dir = ensure_dir(os.path.join(base_dir, "pes"))
        pes_dwd_fetched = self._fetch_data(power="pes", model="DWD_ICON-EU")
        logger.info("Successfully fetched DWD weather data for Hornsea 1.")
        pes_gfs_fetched = self._fetch_data(power="pes", model="NCEP_GFS")
        logger.info("Successfully fetched NCEP GFS weather data for PES Region 10.")
        self.pes_data.update({"dwd_fetched": pes_dwd_fetched, "gfs_fetched": pes_gfs_fetched})
        
        # Locally save fetched PES Region 10 data (long-term)
        pes_dwd_fetched.to_csv(os.path.join(pes_dir, "dwd_fetched.csv"))
        pes_gfs_fetched.to_csv(os.path.join(pes_dir, "gfs_fetched.csv"))

        # Create directory to save fetched energy data (short-term)
        if not os.path.exists("energy_fetched_temporary"):
            os.makedirs("energy_fetched_temporary")
            logger.info("Created directory for energy data.")

        # Fetch energy data
        today = datetime.today().strftime('%Y-%m-%d')
        energy_fetched = self.rebase_api_client.get_variable(day=today, variable="solar_total_production").rename({"timestamp_utc": "dtm"}, axis=1)
        
        # Check if at least one timestamp of fetched energy data and fetched weather data overlaps (This in important in order to correctly perform the preprocessing)
        energy_fetched_copy = energy_fetched.copy()
        energy_fetched_copy["dtm"] = pd.to_datetime(energy_fetched_copy["dtm"])
        if len(pes_dwd_fetched.merge(energy_fetched_copy["dtm"], how="inner", left_on="valid_datetime", right_on="dtm")) == 0:
            logger.warning("No overlapping timestamps of fetched energy data and fetched weather data. Capacity_mwp might be empty.")

        # Locally save fetched energy data (short-term)
        energy_fetched.to_csv("energy_fetched_temporary/energy_fetched.csv")
        logger.info("Successfully fetched and saved energy data (overwriting daily).")

        # Locally save fetched energy data (long-term)
        energy_dir = ensure_dir(os.path.join(base_dir, "energy"))
        energy_fetched.to_csv(os.path.join(energy_dir, "energy_fetched.csv"))
        
        return self
    

    def _fetch_data(self, power, model):
        """
        Fetches and converts weather data for the specified power source and weather model.
        """
        now = datetime.now()
        start_time = pd.Timestamp(datetime(now.year, now.month, now.day, 23, 0, 0), tz='UTC')
        end_time = start_time + timedelta(days=1)

        # Fetch weather data depending on the specified energy producer and weather model
        if power == "hornsea" and model == "DWD_ICON-EU":
            try:
                df_fetched = comp_utils.RebaseAPI.get_hornsea_dwd(self.rebase_api_client)
            except Exception as e:
                logger.error(f"Error occurred while fetching DWD Hornsea 1 data: {e}")
                return self._saved_state
            
        elif power == "hornsea" and model == "NCEP_GFS":
            try:
                df_fetched = comp_utils.RebaseAPI.get_hornsea_gfs(self.rebase_api_client)
            except Exception as e:
                logger.error(f"Error occurred while fetching NCEP GFS Hornsea 1 data: {e}")
                return self._saved_state

        elif power == "pes" and model == "DWD_ICON-EU":
            try:
                df_fetched = comp_utils.RebaseAPI.get_pes10_nwp(self.rebase_api_client, model=model)
            except Exception as e:
                logger.error(f"Error occurred while fetching DWD PES Region 10 data: {e}")
                return self._saved_state
            
        elif power == "pes" and model == "NCEP_GFS":
            try:
                df_fetched = comp_utils.RebaseAPI.get_pes10_nwp(self.rebase_api_client, model=model)
            except Exception as e:
                logger.error(f"Error occurred while fetching NCEP GFS PES Region 10 data: {e}")
                return self._saved_state

        df_fetched = comp_utils.weather_df_to_xr(df_fetched).to_dataframe()
        # Check if the dataframe contains all relevant timestamps for the market period of the next day
        if len(df_fetched[(df_fetched.index.get_level_values("valid_datetime") >= start_time) & 
                          (df_fetched.index.get_level_values("valid_datetime") <= end_time)]
                          .groupby("valid_datetime")) < 25:
            logger.warning("Fetched weather data does not appear to contain all relevant timestamps of the market period. Model predictions might be incomplete.")
        
        # Check if the dataframe contains enough timestamps to compute 6-hour interval rolling features
        if len(df_fetched[(df_fetched.index.get_level_values("valid_datetime") <= start_time)]
                          .groupby("valid_datetime")) < 6:
            logger.warning("Fetched weather data does not appear to contain all relevant timestamps to compute 6-hour rolling interval features. Model predictions might be inaccurate.")

        return df_fetched
    

    def prepare_data(self):
        """
        Prepares the weather data for the machine learning model.
        """
        logger.info("Preparing data...")
        self._saved_state = copy.deepcopy(self)
        
        # Preprocess weather data
        logger.info("Preprocessing data for Hornsea 1...")
        df_preprocessed_hornsea = self._preprocess_data(power="hornsea")
        logger.info("Successfully preprocessed data for Hornsea 1.")

        logger.info("Preprocessing data for PES Region 10...")
        df_preprocessed_pes = self._preprocess_data(power="pes")
        logger.info("Successfully preprocessed data for PES Region 10.")

        # Perform feature engineering on preprocessed weather data
        logger.info("Performing feature engineering on data for Hornsea 1...")
        self._engineer_data(power="hornsea", df_preprocessed=df_preprocessed_hornsea)
        logger.info("Successfully performed feature engineering on data for Hornsea 1.")

        logger.info("Performing feature engineering on data for PES Region 10...")
        self._engineer_data(power="pes", df_preprocessed=df_preprocessed_pes)
        logger.info("Successfully performed feature engineering on data for PES Region 10.")

        return self
    

    def _preprocess_data(self, power):
        """
        Preprocesses the weather data for the specified power source.
        """
        preprocessor = Preprocessing()
        
        # Preprocess weather data depending on the specified energy producer
        if power == "hornsea":
            try:
                geo_data_dict = {"dwd": self.hornsea_data["dwd_fetched"], "ncep": self.hornsea_data["gfs_fetched"]}
                df_preprocessed = preprocessor.perform_preprocessing_pipeline(geo_data_dict,
                                                                            deployment=True,
                                                                            json_file_path="REMIT")
            except Exception as e:
                logger.error(f"Error occured while preprocessing Hornsea 1 data: {e}")
                return self._saved_state
            
        elif power == "pes":
            try:
                geo_data_dict = {"dwd": self.pes_data["dwd_fetched"], "ncep": self.pes_data["gfs_fetched"]}
                energy_data_dict = {"energy_fetched": "energy_fetched_temporary"}
                df_preprocessed = preprocessor.perform_preprocessing_pipeline(geo_data_dict,
                                                                            deployment=True,
                                                                            json_file_path="REMIT",
                                                                            energy_data_dict=energy_data_dict)
            except Exception as e:
                logger.error(f"Error occured while preprocessing PES Region 10 data: {e}")
                return self._saved_state
           
        getattr(self, f"{power}_data").update({"df_preprocessed": df_preprocessed})

        return df_preprocessed
    

    def _engineer_data(self, power, df_preprocessed):
        """
        Performs feature engineering on preprocessed weather data for the specified power source.
        """
        # Set importent parameters for the feature engineering depending on the specified energy producer
        if power == "hornsea":
            logging_text = "Hornsea 1"
            try:
                fe_data = pd.read_parquet("preprocessed_hornsea_with_energy.parquet")
            except FileNotFoundError:
                logger.error("Preprocessed Hornsea 1 data not found. Please check if 'preprocessed_hornsea_with_energy.parquet' exists.")
                return self._saved_state
            except Exception as e:
                logger.error(f"Error occurred while reading 'preprocessed_hornsea_with_energy.parquet': {e}")
                return self._saved_state
            fe_fitted = FeatureEngineerer(label="Wind_MWh_credit", columns_to_ohe=['unavailabilityType', 'affectedUnit'])

        elif power == "pes":
            logging_text = "PES Region 10"
            try:
                fe_data = pd.read_parquet("preprocessed_pes_with_energy.parquet")
            except FileNotFoundError:
                logger.error("Preprocessed PES Region 10 data not found. Please check if 'preprocessed_pes_with_energy.parquet' exists.")
                return self._saved_state
            except Exception as e:
                logger.error(f"Error occurred while reading 'preprocessed_pes_with_energy.parquet': {e}")
                return self._saved_state
            fe_fitted = FeatureEngineerer(label="Solar_MWh_credit", columns_to_ohe=[])

        # Perform feature engineering on the weather data depending on the specified energy producer
        try:
            fe_fitted.perform_feature_engineering(fe_data, deployment=False)
        except Exception as e:
            logger.error(f"Error occurred while performing feature engineering on historical {logging_text} data: {e}")
            return self._saved_state
        try:
            fe_fitted.perform_feature_engineering(df_preprocessed, deployment=True)
        except Exception as e:
            logger.error(f"Error occurred while performing feature engineering on fetched {logging_text} data: {e}")
            return self._saved_state
        array_prepared = fe_fitted.deployment_data
        getattr(self, f"{power}_data").update({"array_prepared": array_prepared})


    def predict(self):
        """
        Makes predictions for the next using the Hornsea 1 and PES Region 10 models.
        """
        logger.info("Making predictions...")
        self._saved_state = copy.deepcopy(self)

        # Make a prediction for Hornsea 1 within the market time interval for the next day
        try:
            pred_hornsea = self.hornsea_model.predict(self.hornsea_data["array_prepared"])
            logger.info("Successfully completed prediction for Hornsea 1.")
        except Exception as e:
            logger.error(f"Error occurred while making a prediction on Hornsea 1 data: {e}")
            return self._saved_state
        # Make a prediction for PES Region 10 within the market time interval for the next day
        try:
            pred_pes = self.pes_model.predict(self.pes_data["array_prepared"])
            logger.info("Successfully completed prediction for PES Region 10.")
        except Exception as e:
            logger.error(f"Error occurred while making a prediction on PES Region 10 data: {e}")
            return self._saved_state
        self.predictions.update({"hornsea": pred_hornsea,
                                 "pes": pred_pes})

        return self


    def prepare_submission(self):
        """
        Prepares the submission data in the required format.
        """
        logger.info("Preparing data for submission...")
        self._saved_state = copy.deepcopy(self)

        try:
            # Create a pandas with the right market times of the next day as index
            submission_data = pd.DataFrame({"datetime":comp_utils.day_ahead_market_times()})
            # Create a dictionary of the summed predictions of the two models (Hornsea 1 and PES)
            pred = dict()
            for keys in [("q10","0.1"), ("q20","0.2"), ("q30","0.3"), ("q40","0.4"), ("q50","0.5"), ("q60","0.6"), ("q70","0.7"), ("q80","0.8"), ("q90","0.9")]:
                    pred[keys[0]] = self.predictions["hornsea"][keys[1]] + self.predictions["pes"][keys[1]]

            # Merge the prediction dictionary with the dataframe on the right market times of the next day.
            index = self.hornsea_data["df_preprocessed"].index
            submission_data = pd.DataFrame(pred, index=index).merge(submission_data, how="right", left_on="val_time", right_on="datetime")
            # Set the predictions for the market bid
            submission_data["market_bid"] = submission_data["q40"]
            # Check if submission data has the right shape
            if submission_data.shape != (48, 11):
                logger.warning("There are less predictions values than there should be. Something seems to have gone wrong.")
            # Check if submission data contains any NaN values
            if submission_data.isna().any().any():
                logger.warning("Predictions contain NaN values. Something seems to have gone wrong.")
            # Check if submission data contains any negative values
            if (submission_data.drop("datetime", axis=1) < 0).any().any():
                logger.warning("Predictions contain negative values. Something seems to have gone wrong.")
            logger.info("Successfully created dataframe of submission data.")
        except Exception as e:
            logger.error(f"Error occurred while creating the prediction dataframe: {e}")
            return self._saved_state
        
        # Prapare the submission data to the right format
        try:
            submission_data = comp_utils.prep_submission_in_json_format(submission_data)
            logger.info("Successfully converted dataframe of submission data to json format.")
            self.predictions.update({"final": submission_data})
        except Exception as e:
            logging.error(f"Error occurred while converting the prediction dataframe to json format: {e}")
            return self._saved_state


    def submit(self):
        """
        Submits the final predictions to the Rebase API.
        """

        while True:
            # Ask the user if they want to proceed with the submission
            response = input("Submit results? (y/n)")
            if response == "y":
                logger.info("Submitting results...")

                # Submit the final predictions
                try:
                    self.rebase_api_client.submit(self.predictions["final"])
                    logger.info("Successfully submitted results.")
                except Exception as e:
                    logger.error(f"Error occurred during submission: {e}")
                    return self
                break

            elif response == "n":
                logger.info("Submission canceled by user.")
                return self
            else:
                logger.info("Invalid input. Please enter 'y' or 'n'.")


# Bei Implementierung des trading modells beachten: "The volume of energy traded in a single period is limited to the range 0 MWh to 1800 MWh, the
# maximum generation output of the hybrid power plant. Bids outside of this range will be rejected by
# the submission API."

# wetter und energie daten speichern