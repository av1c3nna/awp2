import comp_utils
from Preprocessing import Preprocessing, FeatureEngineerer
import pandas as pd
from model_utils import *
from datetime import datetime, timedelta
import os

class AutoSubmitter:

    def __init__(self, hornsea_model, pes_model):
        self.hornsea_model = hornsea_model
        self.pes_model = pes_model
        self.hornsea_data = {}
        self.pes_data = {}
        self.predictions = {}
        self.rebase_api_client = comp_utils.RebaseAPI(api_key = open("DanMaLeo_key.txt").read())


    def fetch_data(self):
        
        hornsea_dwd_fetched = self._fetch_data(power="hornsea", model="DWD_ICON-EU")
        hornsea_gfs_fetched = self._fetch_data(power="hornsea", model="NCEP_GFS")
        self.hornsea_data.update({"dwd_fetched": hornsea_dwd_fetched, "gfs_fetched": hornsea_gfs_fetched})

        pes_dwd_fetched = self._fetch_data(power="pes", model="DWD_ICON-EU")
        pes_gfs_fetched = self._fetch_data(power="pes", model="NCEP_GFS")
        self.pes_data.update({"dwd_fetched": pes_dwd_fetched, "gfs_fetched": pes_gfs_fetched})
        
        if not os.path.exists("energy_data_fetched"):
             os.makedirs("energy_data_fetched")
        today = datetime.today().strftime('%Y-%m-%d')
        energy_fetched = self.rebase_api_client.get_variable(day=today, variable="solar_total_production").rename({"timestamp_utc": "dtm"}, axis=1)
        energy_fetched.to_csv("energy_data_fetched/energy_fetched.csv")
        
        return self
    

    def _fetch_data(self, power, model):

        now = datetime.now()
        yesterday = now - timedelta(days=1)
        start_time = pd.Timestamp(datetime(yesterday.year, yesterday.month, yesterday.day, 23, 0, 0), tz='UTC')
        end_time = start_time + timedelta(days=2)

        if power == "hornsea" and model == "DWD_ICON-EU":
            df_fetched = comp_utils.RebaseAPI.get_hornsea_dwd(self.rebase_api_client)
            
        elif power == "hornsea" and model == "NCEP_GFS":
            df_fetched = comp_utils.RebaseAPI.get_hornsea_gfs(self.rebase_api_client)

        elif power == "pes" and model == "DWD_ICON-EU":
            df_fetched = comp_utils.RebaseAPI.get_pes10_nwp(self.rebase_api_client, model=model)

        elif power == "pes" and model == "NCEP_GFS":
            df_fetched = comp_utils.RebaseAPI.get_pes10_nwp(self.rebase_api_client, model=model)

        df_fetched = comp_utils.weather_df_to_xr(df_fetched).to_dataframe()
        df_fetched = df_fetched[(df_fetched.index.get_level_values("valid_datetime") >= start_time) & 
                                (df_fetched.index.get_level_values("valid_datetime") <= end_time)]
        
        return df_fetched
    

    def prepare_data(self):
        
        df_preprocessed_hornsea = self._preprocess_data(power="hornsea")
        df_preprocessed_pes = self._preprocess_data(power="pes")

        self._engineer_data(power="hornsea", df_preprocessed=df_preprocessed_hornsea)
        self._engineer_data(power="pes", df_preprocessed=df_preprocessed_pes)

        return self
    

    def _preprocess_data(self, power):
        
        now = datetime.now()
        now = pd.Timestamp(datetime(now.year, now.month, now.day, 23, 0, 0), tz='UTC')
        preprocessor = Preprocessing()

        if power == "hornsea":
            geo_data_dict = {"dwd": self.hornsea_data["dwd_fetched"], "ncep": self.hornsea_data["gfs_fetched"]}
            df_preprocessed = preprocessor.perform_preprocessing_pipeline(geo_data_dict,
                                                                      deployment=True,
                                                                      json_file_path="REMIT")
            
        elif power == "pes":
            geo_data_dict = {"dwd": self.pes_data["dwd_fetched"], "ncep": self.pes_data["gfs_fetched"]}
            energy_data_dict = {"energy_fetched": "energy_data_fetched"}
            df_preprocessed = preprocessor.perform_preprocessing_pipeline(geo_data_dict,
                                                                      deployment=True,
                                                                      json_file_path="REMIT",
                                                                      energy_data_dict=energy_data_dict)

        df_preprocessed = df_preprocessed[df_preprocessed.index >= now]
        getattr(self, f"{power}_data").update({"df_preprocessed": df_preprocessed})

        return df_preprocessed
    

    def _engineer_data(self, power, df_preprocessed):
        
        if power == "hornsea":
            fe_data = pd.read_parquet("preprocessed_hornsea_with_energy.parquet")
            fe_fitted = FeatureEngineerer(label="Wind_MWh_credit", columns_to_ohe=['unavailabilityType', 'affectedUnit', 'outage'])

        elif power == "pes":
            fe_data = pd.read_parquet("preprocessed_pes_with_energy.parquet")
            fe_fitted = FeatureEngineerer(label="Solar_MWh_credit", columns_to_ohe=[])
        
        fe_fitted.perform_feature_engineering(fe_data, deployment=False)
        fe_fitted.perform_feature_engineering(df_preprocessed, deployment=True)
        array_prepared = fe_fitted.deployment_data
        getattr(self, f"{power}_data").update({"array_prepared": array_prepared})


    def predict(self):

        pred_hornsea = self.hornsea_model.predict(self.hornsea_data["array_prepared"])
        pred_pes = self.pes_model.predict(self.pes_data["array_prepared"])
        self.predictions.update({"hornsea": pred_hornsea,
                                 "pes": pred_pes})

        return self


    def prepare_submission(self):

        submission_data = pd.DataFrame({"datetime":comp_utils.day_ahead_market_times()})
        pred = dict()
        for keys in [("q10","0.1"), ("q20","0.2"), ("q30","0.3"), ("q40","0.4"), ("q50","0.5"), ("q60","0.6"), ("q70","0.7"), ("q80","0.8"), ("q90","0.9")]:
                pred[keys[0]] = self.predictions["hornsea"][keys[1]] + self.predictions["pes"][keys[1]]

        index = self.hornsea_data["df_preprocessed"].index
        submission_data = pd.DataFrame(pred, index=index).merge(submission_data, how="right", left_on="val_time", right_on="datetime")
        submission_data["market_bid"] = submission_data["q50"]

        submission_data = comp_utils.prep_submission_in_json_format(submission_data)
        self.predictions.update({"final": submission_data})


    def submit(self):
        #self.rebase_api_client.submit(self.predictions["final"])
        pass
