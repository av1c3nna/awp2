import comp_utils
from Preprocessing import Preprocessing, FeatureEngineerer
import pandas as pd


class AutoSubmitter:

    def __init__(self, model):
        self.model = model.lower()
        self.rebase_api_client = comp_utils.RebaseAPI(api_key = open("DanMaLeo_key.txt").read())
        self.data = {}

    def fetch_data(self): # Für die Models nur Wetterdaten für nächsten Tag von 23:00 bis 23:00 Uhr fetchen

        if self.model == "hornsea_1":
            dwd_fetched = comp_utils.RebaseAPI.get_hornsea_dwd(self.rebase_api_client)
            gfs_fetched = comp_utils.RebaseAPI.get_hornsea_gfs(self.rebase_api_client)

        if self.model == "pes":
            energy_fetched = self.rebase_api_client.get_variable(day="2024-10-11", variable="solar_total_production").rename({"timestamp_utc": "dtm"}, axis=1)
            energy_fetched.to_csv("energy_data/energy_fetched.csv")
            dwd_fetched = comp_utils.RebaseAPI.get_pes10_nwp(self.rebase_api_client, model="DWD_ICON-EU")
            gfs_fetched = comp_utils.RebaseAPI.get_pes10_nwp(self.rebase_api_client, model="NCEP_GFS")
        
        dwd_fetched = comp_utils.weather_df_to_xr(dwd_fetched).to_dataframe()
        gfs_fetched = comp_utils.weather_df_to_xr(gfs_fetched).to_dataframe()
        self.data.update({"dwd_fetched": dwd_fetched, "gfs_fetched": gfs_fetched})
        
        return self
    

    def prepare_data(self):

        preprocessor = Preprocessing()
        geo_data_dict = {"dwd": self.data["dwd_fetched"], "ncep": self.data["gfs_fetched"]}
        energy_data_dict = {"energy_fetched": "energy_data"}
        json_file_path = "REMIT"

        df_prepared = preprocessor.perform_preprocessing_pipeline(geo_data_dict, 
                                                                  deployment=True, 
                                                                  merge_with_outage_data=True, 
                                                                  json_file_path=json_file_path, 
                                                                  energy_data_dict=energy_data_dict, # Energie daten nicht verfügbar für nächsten Tag, prep muss aber am Vortag laufen
                                                                  fft=False)
        
        if self.model == "hornsea_1":
            fe_data = pd.read_parquet("preprocessed_hornsea_with_energy.parquet")
            fe_fitted = FeatureEngineerer(label="Wind_MWh_credit")

        if self.model == "pes":
            fe_data = pd.read_parquet("preprocessed_pes_with_energy.parquet")
            fe_fitted = FeatureEngineerer(label="Solar_MWh_credit")

        fe_fitted.perform_feature_engineering(fe_data, deployment=False)
        fe_fitted.perform_feature_engineering(df_prepared, deployment=True)
        df_prepared = fe_fitted.deployment_data

        self.data.update({"df_prepared": df_prepared})

        return self
    

    def predict(self):
        pass


    def submit(self):
        pass



# import comp_utils

# rebase_api_client = comp_utils.RebaseAPI(api_key = open("team_key.txt").read())
# test_date = "2024-10-10"

# capacity = rebase_api_client.get_variable(day=test_date,variable="solar_total_production")


