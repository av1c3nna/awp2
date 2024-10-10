import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.fixes import parse_version, sp_version
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def pinball(y, q, alpha):
    return (y - q) * alpha * (y >= q) + (q - y) * (1 - alpha) * (y < q)

def pinball_score(df, quantiles):
    score = list()
    for qu in quantiles:
        # Berechne den Pinball Loss für jedes Quantil
        score.append(pinball(y=df["true"],
                             q=df[f"{qu}"],
                             alpha=qu/100).mean())
    return sum(score)/len(score)  # Durchschnittlicher Pinball Score


# Base Model Class
class BaseModel:
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
        return (y - q) * alpha * (y >= q) + (q - y) * (1 - alpha) * (y < q)

    def pinball_score(self):
        score = []
        df = pd.DataFrame(self.q_predictions)
        for qu in self.quantiles:
            score.append(self.pinball(y=df["true"], q=df[str(qu)], alpha=qu).mean())
        return sum(score) / len(score)

    def plot_quantils(self, x, y, quantiles):

        first_month = x.index[0].month  # Den Monat des ersten Datums nehmen
        first_month_data = x[x.index.month == first_month]  # Nur die Daten für den ersten Monat filtern
        filter = len(first_month_data.index)
        
        # 2. Filtere die entsprechenden Zeilen aus `y`

        plt.figure(figsize=(10,6))
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        ax1 = sns.lineplot( x=first_month_data.index, y=y["true"][:filter])

        for quantile in quantiles:
            sns.lineplot(
                        x=first_month_data.index,
                        y=y[str(quantile)][:filter],
                        color='gray',
                        alpha=(1-abs(1-quantile)),
                        label=f'q{quantile}')

        #plt.xlim(x.index.min())
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
        model_filename = os.path.join(self.model_save_dir, f"xgboost_model.json")
        if os.path.exists(model_filename):
            self.booster = xgb.Booster()
            self.booster.load_model(model_filename)
            self.models_loaded = True
            print(f"Loaded pretrained XGBoost model from {model_filename}")
        else:
            print(f"No pretrained model found at {model_filename}, training a new model instead.")

    def train_xgboost_model(self, x_train, y_train, x_val, y_val, feature_names):
        evals_result = {}
        Xy_train = xgb.QuantileDMatrix(x_train, y_train, feature_names = feature_names)
        Xy_val = xgb.QuantileDMatrix(x_val, y_val, ref=Xy_train, feature_names = feature_names)

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



# Quantile Regressor Model Class
class QuantileRegressorModel(BaseModel):
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
        
        return predictions