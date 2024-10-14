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

        # Fetch weather data for Hornsea 1
        hornsea_dwd_fetched = self._fetch_data(power="hornsea", model="DWD_ICON-EU")
        logger.info("Successfully fetched DWD weather data for Hornsea 1.")
        hornsea_gfs_fetched = self._fetch_data(power="hornsea", model="NCEP_GFS")
        logger.info("Successfully fetched NCEP GFS weather data for Hornsea 1.")
        self.hornsea_data.update({"dwd_fetched": hornsea_dwd_fetched, "gfs_fetched": hornsea_gfs_fetched})
        
        # Fetch weather data for PES Region 10
        pes_dwd_fetched = self._fetch_data(power="pes", model="DWD_ICON-EU")
        logger.info("Successfully fetched DWD weather data for Hornsea 1.")
        pes_gfs_fetched = self._fetch_data(power="pes", model="NCEP_GFS")
        logger.info("Successfully fetched NCEP GFS weather data for PES Region 10.")
        self.pes_data.update({"dwd_fetched": pes_dwd_fetched, "gfs_fetched": pes_gfs_fetched})
        
        # Locally save data on the amount of power currently generated in the PES 10 region
        if not os.path.exists("energy_data_fetched"):
             os.makedirs("energy_data_fetched")
        logger.info("Created directory for energy data.")
        today = datetime.today().strftime('%Y-%m-%d')
        energy_fetched = self.rebase_api_client.get_variable(day=today, variable="solar_total_production").rename({"timestamp_utc": "dtm"}, axis=1)
        energy_fetched.to_csv("energy_data_fetched/energy_fetched.csv")
        logger.info("Successfully fetched and saved energy data.")
        
        return self
    

    def _fetch_data(self, power, model):
        """
        Fetches and converts weather data for the specified power source and weather model.
        """
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        start_time = pd.Timestamp(datetime(yesterday.year, yesterday.month, yesterday.day, 23, 0, 0), tz='UTC')
        end_time = start_time + timedelta(days=2)

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

        # Limit weather data to the relevant time interval
        df_fetched = comp_utils.weather_df_to_xr(df_fetched).to_dataframe()
        df_fetched = df_fetched[(df_fetched.index.get_level_values("valid_datetime") >= start_time) & 
                                (df_fetched.index.get_level_values("valid_datetime") <= end_time)]
        
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
        now = datetime.now()
        now = pd.Timestamp(datetime(now.year, now.month, now.day, 17, 0, 0), tz='UTC')
        market_end_time = now + timedelta(hours=29, minutes=30)
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
                energy_data_dict = {"energy_fetched": "energy_data_fetched"}
                df_preprocessed = preprocessor.perform_preprocessing_pipeline(geo_data_dict,
                                                                            deployment=True,
                                                                            json_file_path="REMIT",
                                                                            energy_data_dict=energy_data_dict)
            except Exception as e:
                logger.error(f"Error occured while preprocessing PES Region 10 data: {e}")
                return self._saved_state
           
        # Limit preprocessed weather data data to the relevant time interval
        df_preprocessed = df_preprocessed[(df_preprocessed.index >= now) & (df_preprocessed.index <= market_end_time)]
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

        # Make a prediction within the market time interval for the next day
        try:
            pred_hornsea = self.hornsea_model.predict(self.hornsea_data["array_prepared"])
            logger.info("Successfully completed prediction for Hornsea 1.")
        except Exception as e:
            logger.error(f"Error occurred while making a prediction on Hornsea 1 data: {e}")
            return self._saved_state
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

        # Create a pandas with the right market times of the next day as index
        submission_data = pd.DataFrame({"datetime":comp_utils.day_ahead_market_times()})
        # Create a dictionary of the summed predictions of the two models (Hornsea 1 and PES)
        pred = dict()
        for keys in [("q10","0.1"), ("q20","0.2"), ("q30","0.3"), ("q40","0.4"), ("q50","0.5"), ("q60","0.6"), ("q70","0.7"), ("q80","0.8"), ("q90","0.9")]:
                pred[keys[0]] = self.predictions["hornsea"][keys[1]] + self.predictions["pes"][keys[1]]

        # Merge the prediction dictionary with the dataframe on the right market times of the next day.
        index = self.hornsea_data["df_preprocessed"].index
        submission_data = pd.DataFrame(pred, index=index).merge(submission_data, how="right", left_on="val_time", right_on="datetime")
        submission_data["market_bid"] = submission_data["q50"]
        logger.info("Successfully created dataframe of submission data.")

        # Prapare the submission data to the right format
        submission_data = comp_utils.prep_submission_in_json_format(submission_data)
        logger.info("Successfully converted dataframe of submission data to json format.")
        self.predictions.update({"final": submission_data})


    def submit(self):
        """
        Submits the final predictions to the Rebase API.
        """

        while True:
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



# Zeiten checken (Prüfungen und UTC Verständnis)
# Prüfen, ob submission daten richtig sind