import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.fixes import parse_version, sp_version
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import matplotlib.dates as mdates
from datetime import datetime, date
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, ParameterGrid, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.conformity_scores import GammaConformityScore


def pinball(y, q, alpha):
    return (y - q) * alpha * (y >= q) + (q - y) * (1 - alpha) * (y < q)

def pinball_score(df, quantiles):
    score = list()
    for qu in quantiles:
        # pinball loss for every quantile
        score.append(pinball(y=df["true"],
                             q=df[f"{qu}"],
                             alpha=qu).mean())
    return sum(score)/len(score)  # avg pinball score

def plot_quantile_performance(model_list, model_names, title, quantiles, df_list=False):
    # Dictionary, um Scores für alle Modelle zu speichern
    quantile_scores = {qu: [] for qu in quantiles}
    if not df_list:
        for model in model_list:
            # Flatten the predictions
            model.q_predictions = {k: v.flatten() for k, v in model.q_predictions.items() if k != "date"}
            df = pd.DataFrame(model.q_predictions)

            # Berechnung der Pinball-Scores für jedes Quantil
            for qu in quantiles:
                score = pinball(y=df["true"], q=df[str(qu)], alpha=qu).mean()
                quantile_scores[qu].append(score)  # Score zum entsprechenden Quantil hinzufügen
    if df_list:
        for df in model_list:
            
            for qu in quantiles:
                score = pinball(y=df["true"], q=df[str(qu)], alpha=qu).mean()
                quantile_scores[qu].append(score)  # Score zum entsprechenden Quantil hinzufügen
    
    num_models = len(model_list)
    bar_width = 0.15  
    index = np.arange(len(quantiles))

    
    for i, model_scores in enumerate(zip(*quantile_scores.values())):
        plt.bar(index + i * bar_width, model_scores, bar_width, label=model_names[i])

    plt.xlabel('Quantile')
    plt.ylabel('Pinball Score')
    plt.title(title)
    plt.xticks(index + bar_width * (num_models - 1) / 2, quantiles)  
    plt.legend(title='Models')
    plt.tight_layout()
    plt.show()


# Base Model Class
class BaseModel:
    """super class for pinball score and visualization"""

    def __init__(self, feature_engineerer, quantiles, model_save_dir, load_pretrained=False):
        self.feature_engineerer = feature_engineerer
        self.quantiles = quantiles
        self.q_predictions = {"true": feature_engineerer.y_test.values}
        self.model_save_dir = model_save_dir
        self.load_pretrained = load_pretrained
        self.models_loaded = False
        self._create_model_save_dir()

    def _create_model_save_dir(self):
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

    def pinball(self, y, q, alpha):
        """formula for pinball score"""

        return (y - q) * alpha * (y >= q) + (q - y) * (1 - alpha) * (y < q)

    def pinball_score(self):
        """pinball score implemetation"""

        score = []
        self.q_predictions = {k: v.flatten() for k, v in self.q_predictions.items() if k != "date"}
        df = pd.DataFrame(self.q_predictions)
        for qu in self.quantiles:
            score.append(self.pinball(y=df["true"], q=df[str(qu)], alpha=qu).mean())
        return sum(score) / len(score)
    
    def pinball_score_single(df, qu):
        score = pinball(y=df["true"],
                                q=df[f"{qu}"],
                                alpha=qu).mean()
        return score
    
    def list_pinball_scores(pred_and_true, quantiles = np.arange(0.1, 1.0, 0.1).round(2)):
        for q in quantiles:
            print(f"{q}", BaseModel.pinball_score_single(pred_and_true, q))

    def sort_quantiles(self, data, quantiles):
        """sorting quantiles so that smaller quantiles have smaller values"""
        quantile_keys = [str(qu) for qu in quantiles]
        
        # create dataframe from dictionary with quantiles as feature
        quantile_df = pd.DataFrame({key: data[key] for key in quantile_keys})
        
        # sorting quantiles
        sorted_quantiles = np.sort(quantile_df.values, axis=1)
        
        # overwrite values in dictionary
        for idx, key in enumerate(quantile_keys):
            data[key] = sorted_quantiles[:, idx]
        
        return data

    def replace_neg_values(self, data, quantiles):
        """replacing neg values with 0"""

        quantile_keys = [str(qu) for qu in quantiles] # keys

        quantile_df = pd.DataFrame({key: data[key] for key in quantile_keys}) # dict to dataframe

        quantile_df_no_negative = quantile_df.clip(lower=0) # replace values

        #overwrite dictionary
        for key in quantile_keys:
            data[key] = quantile_df_no_negative[key].values

        return data

    def plot_quantils(self, daterange, y, quantiles, year = 2023, month=8, day=False):
        """visualization for prediction"""

        warnings.filterwarnings("ignore", category=FutureWarning)

        y["date"] = daterange
        plot_df = pd.DataFrame(y)

        sns.set_style("whitegrid")

        
        data = plot_df[(plot_df.date.dt.year == year) & (plot_df.date.dt.month == month)]  # use data for specific year and month
           
        if day:
            data = plot_df[(plot_df.date.dt.year == year) & (plot_df.date.dt.month == month) & (plot_df.date.dt.day == day)] # use  data for specific year, month and day
        
        # 2. Filtere die entsprechenden Zeilen aus `y`

        plt.figure(figsize=(10,6))
        sns.lineplot( x=data.date, y=data["true"], label="True")

        for quantile in quantiles:
            sns.lineplot(
                        x=data.date,
                        y=data[str(quantile)],
                        color='gray',
                        alpha=(1-abs(1-quantile)),
                        label=f'q{quantile}')
        if day:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            date = datetime(year, month, day)
            formatted_date = date.strftime("%d/%m/%Y")
        else:
            date = datetime(year, month, 1)
            formatted_date = date.strftime("%B %Y")
        
        plt.title(f"Energy-Generation on {formatted_date}")
        plt.xlabel('Date/Time')
        plt.ylabel('Generation [MWh]')
        plt.tight_layout()


# XGBoost Model Class
class XGBoostModel(BaseModel):
    def __init__(self, feature_engineerer, quantiles, model_save_dir, hyperparams=False, load_pretrained=False, num_boost_round=250, early_stopping_rounds=10):
        super().__init__(feature_engineerer, quantiles, model_save_dir, load_pretrained)
        
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        if not hyperparams:
            self.hyperparams = {
            # Use the quantile objective function.
            "objective": "reg:quantileerror",
            "tree_method": "hist",
            "quantile_alpha": quantiles,
            "learning_rate": 0.04,
            "max_depth": 8,
        }
            
        else:
            self.hyperparams = hyperparams

        if self.load_pretrained:
            self._load_model()

    def _load_model(self):
        """load the pretrained model"""

        model_filename = os.path.join(self.model_save_dir, f"xgboost_model.json")
        if os.path.exists(model_filename):
            self.booster = xgb.Booster()
            self.booster.load_model(model_filename)
            self.models_loaded = True
            print(f"Loaded pretrained XGBoost model from {model_filename}")
        else:
            print(f"No pretrained model found at {model_filename}, training a new model instead.")

    def train_xgboost_model(self, x_train, y_train, x_val, y_val, feature_name):
        """fit xgboost model to data"""

        evals_result = {}
        Xy_train = xgb.QuantileDMatrix(x_train, y_train, feature_names=feature_name)
        Xy_val = xgb.QuantileDMatrix(x_val, y_val, ref=Xy_train, feature_names=feature_name)

        # Training XGBoost model
        booster = xgb.train(
            self.hyperparams,  
            Xy_train,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            evals=[(Xy_train, "Train"), (Xy_val, "Val")],
            evals_result=evals_result,
        )
        return booster

    def train_and_predict(self):
        """Train the xgboost model or use the pretrained one."""
        if not self.load_pretrained or not self.models_loaded:
            # Train a new model if not loading pretrained
            self.booster = self.train_xgboost_model(
                self.feature_engineerer.X_train,
                self.feature_engineerer.y_train,
                self.feature_engineerer.X_val,
                self.feature_engineerer.y_val,
                self.feature_engineerer.features_after_fe
            )

            # Save the model
            model_filename = os.path.join(self.model_save_dir, f"xgboost_model.json")
            self.booster.save_model(model_filename)
            print(f"Saved new XGBoost model to {model_filename}")

            self.models_loaded = True

        else:
            print("Using the loaded pretrained XGBoost model for prediction.")

        # Predict for all quantiles at once
        scores = self.booster.inplace_predict(self.feature_engineerer.X_test)

        # Assuming the number of quantiles corresponds to the output shape
        assert scores.shape[1] == len(self.quantiles), "Mismatch in the number of quantiles."

        for i, quantile in enumerate(self.quantiles):
            self.q_predictions[str(quantile)] = scores[:, i]

    def predict(self, X_test):
        """use trained or loaded model to predict"""

        if not self.models_loaded:
            raise ValueError("Model not loaded. You need to load or train the model first.")
        
        # Predict for all quantiles at once
        scores = self.booster.inplace_predict(X_test)
        predictions = {}
        
        # Assuming that the number of quantiles is the same as the output shape
        assert scores.shape[1] == len(self.quantiles), "Mismatch in quantile predictions."

        for i, quantile in enumerate(self.quantiles):
            predictions[str(quantile)] = scores[:, i]

        return predictions
    
    def plot_feature_importance(self):
        """plot feature importance"""

        self.importance_dict = self.booster.get_score(importance_type='weight')
        self.importance_df = pd.DataFrame(list(self.importance_dict.items()), columns=['Feature', 'Importance'])
        # self.importance_df = self.importance_df.sort_values(by='Importance')
        plt.figure(figsize=(12, 8))  # Größe anpassen
        plt.barh(self.importance_df['Feature'], self.importance_df['Importance'])
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.grid(True)
        plt.title('Manuelle Feature Importance Visualisierung')
        plt.show()



# Quantile Regressor Model Class
class QuantileRegressorModel(BaseModel):
    """basic quantile regression model"""

    def __init__(self, feature_engineerer, quantiles, model_save_dir, load_pretrained=False):
        super().__init__(feature_engineerer, quantiles, model_save_dir, load_pretrained)
        self.solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
        self.models = {}

        if self.load_pretrained:
            self._load_models()

    def _load_models(self):
        """Load the pretrained models from disk."""
        for quantile in self.quantiles:
            model_filename = os.path.join(self.model_save_dir, f"qr_model_quantile_{quantile}.pkl")
            if os.path.exists(model_filename):
                self.models[quantile] = joblib.load(model_filename)
                self.models_loaded = True  # Set models_loaded to True when models are loaded
                print(f"Loaded pretrained Quantile Regressor model for quantile {quantile} from {model_filename}")
            else:
                print(f"No pretrained model found for quantile {quantile}. Training a new model instead.")

    def train_and_predict(self):
        """Train the Quantile Regressor models or use the pretrained ones."""
        for quantile in self.quantiles:
            if not self.load_pretrained or quantile not in self.models:
                # Train a new model for this quantile
                qr_model = QuantileRegressor(quantile=quantile, alpha=0, solver=self.solver)
                qr_model.fit(self.feature_engineerer.X_train, self.feature_engineerer.y_train)

                # Save the model
                model_filename = os.path.join(self.model_save_dir, f"qr_model_quantile_{quantile}.pkl")
                joblib.dump(qr_model, model_filename)
                print(f"Saved Quantile Regressor model for quantile {quantile} to {model_filename}")

                # Store the model for prediction
                self.models[quantile] = qr_model
            else:
                print(f"Using the loaded pretrained Quantile Regressor model for quantile {quantile}")

            # Predict and store the results
            self.q_predictions[str(quantile)] = self.models[quantile].predict(self.feature_engineerer.X_test)

        #sort results
        self.q_predictions = self.sort_quantiles(self.q_predictions, self.quantiles)
        self.q_predictions = self.replace_neg_values(self.q_predictions, self.quantiles)

        # **Set models_loaded to True after training so predict can be called immediately**
        self.models_loaded = True

    def predict(self, X_test):
        """Use the trained or loaded models to make predictions."""
        if not self.models_loaded:
            raise ValueError("Models not loaded. You need to load or train the models first.")

        predictions = {}
        for quantile in self.quantiles:
            if quantile not in self.models:
                raise ValueError(f"Model for quantile {quantile} not available. Train or load the model first.")
            
            predictions[str(quantile)] = self.models[quantile].predict(X_test)

        #sort results
        predictions = self.sort_quantiles(predictions, self.quantiles)
        predictions = self.replace_neg_values(predictions, self.quantiles)

        return predictions
    
# Define pinball loss function
def pinball_loss(y_true, y_pred, quantile=0.9):
    delta = y_true - y_pred
    loss = np.where(delta > 0, quantile * delta, (1 - quantile) * - delta)
    return np.mean(loss)

class LGBMRegressorModel(BaseModel):
    """lightgbm quantile regression"""
    
    def __init__(self, feature_engineerer, quantiles, model_save_dir, load_pretrained=False):
        super().__init__(feature_engineerer, quantiles, model_save_dir, load_pretrained)
        self.models = {}

        if self.load_pretrained:
            self._load_models()

    def _load_models(self):
        """Load the pretrained models from disk."""
        for quantile in self.quantiles:
            model_filename = os.path.join(self.model_save_dir, f"lgbm_model_quantile_{quantile}.pkl")
            if os.path.exists(model_filename):
                self.models[quantile] = joblib.load(model_filename)
                self.models_loaded = True  # Set models_loaded to True when models are loaded
                print(f"Loaded pretrained Quantile Regressor model for quantile {quantile} from {model_filename}")
            else:
                print(f"No pretrained model found for quantile {quantile}. Training a new model instead.")

    def train_and_predict(self):
        """Train the LGBMRegressor models or use the pretrained ones."""
        for quantile in self.quantiles:
            if not self.load_pretrained or quantile not in self.models:
                # Train a new model for this quantile
                qr_lgbm = lgb.LGBMRegressor(objective='quantile', alpha=quantile, n_estimators=1000, force_col_wise=True)
                qr_lgbm.fit(
                    self.feature_engineerer.X_train, 
                    self.feature_engineerer.y_train,
                    eval_set=[(self.feature_engineerer.X_train, self.feature_engineerer.y_train), 
                              (self.feature_engineerer.X_val, self.feature_engineerer.y_val)],
                    eval_names=['train', 'valid'],
                    eval_metric='quantile',
                    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(50)]
                )

                # Save the model
                model_filename = os.path.join(self.model_save_dir, f"lgbm_model_quantile_{quantile}.pkl")
                joblib.dump(qr_lgbm, model_filename)
                print(f"Saved Quantile Regressor model for quantile {quantile} to {model_filename}")

                # Store the model for prediction
                self.models[quantile] = qr_lgbm
            else:
                print(f"Using the loaded pretrained Quantile Regressor model for quantile {quantile}")

            # Predict and store the results
            self.q_predictions[str(quantile)] = self.models[quantile].predict(self.feature_engineerer.X_test)

        #sort results
        self.q_predictions = self.sort_quantiles(self.q_predictions, self.quantiles)
        self.q_predictions = self.replace_neg_values(self.q_predictions, self.quantiles)

        # **Set models_loaded to True after training so predict can be called immediately**
        self.models_loaded = True

    def train_and_predict_hyperparametertuning(self, parameters, search="GridSearch"):
        """Train the LGBMRegressor models or use the pretrained ones."""
        
        cv = TimeSeriesSplit(n_splits=3)

        for quantile in self.quantiles:
            scorer = make_scorer(pinball_loss, greater_is_better=False, quantile=quantile)
            print(f"--------Train model for Quantile {quantile}: ")
            if not self.load_pretrained or quantile not in self.models:
                # Train a new model for this quantile
                lgbm = lgb.LGBMRegressor(objective='quantile', alpha=quantile, verbose=-1)

                if search == "GridSearch":
                    grid_lgbm = GridSearchCV(estimator=lgbm, param_grid=parameters, cv=cv, n_jobs=-1, scoring = scorer)
                    grid_lgbm.fit(
                        self.feature_engineerer.X_train, 
                        self.feature_engineerer.y_train,
                        eval_set=[(self.feature_engineerer.X_train, self.feature_engineerer.y_train), 
                                (self.feature_engineerer.X_val, self.feature_engineerer.y_val)],
                        eval_names=['train', 'valid'],
                        eval_metric='quantile',
                        callbacks=[early_stopping(stopping_rounds=50),log_evaluation(25)]
                    )
                if search == "RandomSearch":
                    grid_lgbm = RandomizedSearchCV(estimator=lgbm, param_distributions=parameters, cv=cv, n_jobs=-1, scoring = scorer, n_iter=20)
                    grid_lgbm.fit(
                        self.feature_engineerer.X_train, 
                        self.feature_engineerer.y_train,
                        eval_set=[(self.feature_engineerer.X_train, self.feature_engineerer.y_train), 
                                (self.feature_engineerer.X_val, self.feature_engineerer.y_val)],
                        eval_names=['train', 'valid'],
                        eval_metric='quantile',
                        callbacks=[early_stopping(stopping_rounds=50),log_evaluation(25)]
                    )

                # Save the model
                model_filename = os.path.join(self.model_save_dir, f"lgbm_model_quantile_{quantile}.pkl")
                joblib.dump(grid_lgbm.best_estimator_, model_filename)
                print(f"Saved Quantile Regressor model for quantile {quantile} to {model_filename}")

                # Store the model for prediction
                print(grid_lgbm.best_params_)
                self.models[quantile] = grid_lgbm.best_estimator_
            else:
                print(f"Using the loaded pretrained Quantile Regressor model for quantile {quantile}")

            # Predict and store the results
            self.q_predictions[str(quantile)] = self.models[quantile].predict(self.feature_engineerer.X_test)

        #sort results
        self.q_predictions = self.sort_quantiles(self.q_predictions, self.quantiles)
        self.q_predictions = self.replace_neg_values(self.q_predictions, self.quantiles)

        # **Set models_loaded to True after training so predict can be called immediately**
        self.models_loaded = True

    def predict(self, X_test):
        """Use the trained or loaded models to make predictions."""
        if not self.models_loaded:
            raise ValueError("Models not loaded. You need to load or train the models first.")

        predictions = {}
        for quantile in self.quantiles:
            if quantile not in self.models:
                raise ValueError(f"Model for quantile {quantile} not available. Train or load the model first.")
                
            predictions[str(quantile)] = self.models[quantile].predict(X_test)
            
        #sort results
        predictions = self.sort_quantiles(predictions, self.quantiles)
        predictions = self.replace_neg_values(predictions, self.quantiles)

        return predictions
    
    def plot_feature_importance(self, feature_dataset):
        self.importance_df = pd.DataFrame({
            # 'Feature': qr_lgbm.feature_name_,
            # 'Importance': qr_lgbm.feature_importances_
        })
        self.importance_df["feature_name"] = feature_dataset.drop(['Solar_MWh_credit', 'Wind_MWh_credit'], axis=1).columns
        # Plotten der Feature Importance
        plt.figure(figsize=(12, 8))  # Plot-Größe anpassen
        plt.barh(self.importance_df['feature_name'], self.importance_df['Importance'])
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Manuelle Feature Importance Visualisierung Lightgbm')
        plt.grid(True)
        plt.show()


class ConformalQuantilePredictionLGBM(BaseModel):
    """lightgbm quantile regression"""
    
    def __init__(self, feature_engineerer, quantiles, model_save_dir, load_pretrained=False):
        super().__init__(feature_engineerer, quantiles, model_save_dir, load_pretrained)
        self.models = {}

        if self.load_pretrained:
            self._load_models()

    def _load_models(self):
        """Load the pretrained models from disk."""
        point_model_filename = os.path.join(self.model_save_dir, "point_prediction_lgbm.pkl")
        if os.path.exists(point_model_filename):
            self.point_model = joblib.load(point_model_filename)
            print(f"Loaded pretrained Point Prediction model (50%-Quantile) from {point_model_filename}")
        else:
            print("No pretrained Point Prediction model found. Training a new model instead.")

        for quantile in self.quantiles:
            model_filename = os.path.join(self.model_save_dir, f"conformal_quantile_prediction_lgbm{quantile}.pkl")
            if os.path.exists(model_filename):
                self.models[quantile] = joblib.load(model_filename)
                self.models_loaded = True  # Set models_loaded to True when models are loaded
                print(f"Loaded pretrained Quantile Regressor model for quantile {quantile} from {model_filename}")
            else:
                print(f"No pretrained model found for quantile {quantile}. Training a new model instead.")

    def train_point_prediction(self, param_distribution):
        """trains 50% Quantile modell for point prediction"""
        estimator = lgb.LGBMRegressor(objective='quantile', alpha=0.5, random_state=7,verbose=-1)
        
        #hyperparametertuning
        optim_model = RandomizedSearchCV(
                                            estimator,
                                            param_distributions=param_distribution,
                                            n_jobs=-1,
                                            n_iter=30,
                                            cv=TimeSeriesSplit(n_splits=2),
                                            random_state=7
                                        )
        optim_model.fit(self.feature_engineerer.X_train, self.feature_engineerer.y_train)
        estimator = optim_model.best_estimator_
        # Speichere das 50%-Quantil-Modell (Punktvorhersage)
        point_model_filename = os.path.join(self.model_save_dir, "point_prediction_lgbm.pkl")
        joblib.dump(estimator, point_model_filename)
        print(f"Saved Point Prediction model (50%-Quantile) to {point_model_filename}")
        return estimator
    
    def train_and_predict(self, param_distribution):
        if not self.load_pretrained or not hasattr(self, 'point_model'):
            # Trainiere ein neues Modell, falls kein vortrainiertes vorhanden ist
            self.point_model = self.train_point_prediction(param_distribution=param_distribution)
        else:
            print("Using the loaded pretrained Point Prediction model (50%-Quantile).")


        for quantile in self.quantiles:
            if not self.load_pretrained or quantile not in self.models:
                mapie = MapieRegressor(estimator=self.point_model, cv="prefit")
                mapie.fit(self.feature_engineerer.X_train, self.feature_engineerer.y_train, X_calib=self.feature_engineerer.X_val, y_calib=self.feature_engineerer.y_val)
                model_filename = os.path.join(self.model_save_dir, f"conformal_quantile_prediction_lgbm{quantile}.pkl")
                joblib.dump(mapie, model_filename)
                print(f"Saved Quantile Regressor model for quantile {quantile} and {1-quantile} to {model_filename}")

                self.models[quantile] = mapie
            else:
                print(f"Using the loaded pretrained Conformal Quantile Regressor model for quantile {quantile}")
            
            y_pred, y_pis = self.models[quantile].predict(self.feature_engineerer.X_test, alpha=[quantile])
            self.q_predictions[str(quantile)] = y_pis.reshape(-1,2)[:,0]
            self.q_predictions[str(1-quantile)] = y_pis.reshape(-1,2)[:,1]
        self.q_predictions["0.5"] = self.point_model.predict(self.feature_engineerer.X_test)

        #sort results
        self.q_predictions = self.sort_quantiles(self.q_predictions, self.quantiles)
        self.q_predictions = self.replace_neg_values(self.q_predictions, self.quantiles)
        self.q_predictions = self.replace_neg_values(self.q_predictions, [0.5])

        # **Set models_loaded to True after training so predict can be called immediately**
        self.models_loaded = True

    
    def predict(self, X_test):
        """Use the trained or loaded models to make predictions."""
        if not self.models_loaded:
            raise ValueError("Models not loaded. You need to load or train the models first.")

        predictions = {}
        for quantile in self.quantiles:
            if quantile not in self.models:
                raise ValueError(f"Model for quantile {quantile} not available. Train or load the model first.")
                
            y_pred, y_pis = self.models[quantile].predict(X_test)
            predictions[str(quantile)] = y_pis.reshape(-1,2)[:,0]
            predictions[str(1-quantile)] = y_pis.reshape(-1,2)[:,1]
        predictions["0.5"] = self.point_model.predict(X_test)
            
        #sort results
        predictions = self.sort_quantiles(predictions, self.quantiles)
        predictions = self.replace_neg_values(predictions, self.quantiles)
        predictions = self.replace_neg_values(predictions, [0.5])

        return predictions
    