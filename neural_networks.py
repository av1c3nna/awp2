from functools import partial
from itertools import chain
import torch
import torch.nn as nn
import numpy as np
import model_utils
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
import os, logging
import tensorflow as tf
from tensorflow.keras.saving import load_model


def save_pytorch_model(model, model_name):
    torch.save(model.state_dict(), f"{model_name}")

def load_pytorch_model(class_name, model_name, len_features, quantiles = np.arange(0.1, 1.0, 0.1).round(2)):
    model = class_name(quantiles = quantiles, len_features = len_features)
    model.load_state_dict(torch.load(f"{model_name}", weights_only=True))

class q_model(nn.Module):
    def __init__(self, 
                 quantiles, 
                 in_shape=50,  
                 dropout=0.5,
                 len_features=77):     
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        self.in_shape = in_shape
        self.len_features = len_features
        self.out_shape = len(quantiles)
        self.dropout = dropout
        self.build_model()
        self.init_weights()
        
    def build_model(self): 
        self.linear1 = nn.Linear(50, 150)
        self.linear2 = nn.Linear(150, 80)
        self.linear3 = nn.Linear(80, 40)
        self.linear4 = nn.Linear(40, 20)
        self.activation = nn.ReLU()

        
        # Final layers for quantiles
        final_layers = [
            nn.Linear(20, 1) for _ in range(len(self.quantiles))
        ]
        self.final_layers = nn.ModuleList(final_layers)

    def forward(self, x):

        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.activation(out)
        out = self.linear3(out)
        out = self.activation(out)
        out = self.linear4(out)
        out = self.activation(out)

        # Apply final layers to get quantile outputs
        quantile_outputs = [layer(out) for layer in self.final_layers]
        
        return torch.cat(quantile_outputs, dim=-1)
        
    def init_weights(self):
        for m in [self.linear1, self.linear2]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)  
        for m in chain(self.final_layers):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)        
        
    # def forward(self, x):
    #     tmp_ = self.base_model(x)
    #     return torch.cat([layer(tmp_) for layer in self.final_layers], dim=1)


class LSTM_model(q_model):
    def __init__(self, quantiles= np.arange(0.1, 1.0, 0.1).round(2), 
                 in_shape=50,  
                 dropout=0.5,
                 len_features=66):
        super().__init__(quantiles = quantiles, in_shape = in_shape, dropout = dropout, len_features = len_features)

    def build_model(self):
        self.lstm1 = nn.LSTM(self.len_features, 20, dropout=0.3,  num_layers = 2, batch_first=True)
        self.activation = nn.ReLU()
        # Final layers for quantiles
        final_layers = [
            nn.Linear(20, 1) for _ in range(len(self.quantiles))
        ]
        self.final_layers = nn.ModuleList(final_layers)

    def forward(self, x):
        # x is of shape (batch_size, seq_len, len_features)
        # LSTM layers - extracting the output (not hidden and cell states)
        out, _ = self.lstm1(x)
        # Apply final layers to get quantile outputs
        quantile_outputs = [layer(out) for layer in self.final_layers]
        
        return torch.cat(quantile_outputs, dim=-1)
    
    def init_weights(self): 
        for m in chain(self.final_layers):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)   
    

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
    
class QuantileLoss2(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            errors2 = preds[:, i] - target
            losses.append(torch.max((1 - q) * errors2, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
    
class Learner(model_utils.BaseModel):
    def __init__(self, model, optimizer_class, loss_func, lr):
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer_class(self.model.parameters())
        self.loss_func = loss_func.to(self.device)
        self.loss_history = []
        
    def fit(self, x, y, epochs, batch_size):
        self.model.train()
        for e in range(epochs):
            print(f"{e}. epoche startet: ")
            # shuffle_idx = np.arange(x.shape[0])
            # np.random.shuffle(shuffle_idx)
            # x = x[shuffle_idx]
            # y = y[shuffle_idx]
            epoch_losses = []
            for idx in range(0, x.shape[0], batch_size):
                self.optimizer.zero_grad()
                batch_x = torch.from_numpy(x[idx : min(idx + batch_size, x.shape[0]),:]).float().to(self.device).requires_grad_(False)
                batch_y = torch.from_numpy(y[idx : min(idx + batch_size, y.shape[0])]).float().to(self.device).requires_grad_(False)
                preds = self.model(batch_x)
                loss = self.loss_func(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.cpu().detach().numpy())                                
            epoch_loss =  np.mean(epoch_losses)
            self.loss_history.append(epoch_loss)
            if (e+1) % 10 == 0:
                print("Epoch {}: {}".format(e+1, epoch_loss))
                
    def predict(self, x, mc=False):
        if mc:
            self.model.train()
        else:
            self.model.eval()
        return self.model(torch.from_numpy(x).to(self.device).requires_grad_(False)).cpu().detach().numpy()

import pytorch_forecasting

class Trainer(model_utils.BaseModel):
    def __init__(self, feature_engineerer, model_class, quantiles, optimizer_class=torch.optim.Adam, in_shape=50, dropout=0.1, weight_decay=1e-6, lr = 0.01, batch_size = 96):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantiles = quantiles
        self.feature_engineerer = feature_engineerer
        self.loss_func = QuantileLoss2(self.quantiles)
        self.lr = lr
        self.batch_size = batch_size
        
        # Define model, loss function, and optimizer
        self.model = model_class(self.quantiles,in_shape=in_shape, dropout=dropout, len_features = len(self.feature_engineerer.features_after_fe)).to(self.device)
        self.learner = Learner(
            self.model, 
            partial(optimizer_class, weight_decay=weight_decay),
            self.loss_func,
            lr = self.lr
        )

    def fit(self, epochs=150):
        self.X_train = self.feature_engineerer.X_train
        self.y_train = self.feature_engineerer.y_train.values

        self.learner.fit(self.X_train, self.y_train, epochs, self.batch_size)

    def predict(self, prediction_set):
        self.y_pred = self.learner.predict(prediction_set.astype(np.float32))
        return self.y_pred
    
    def train_and_test(self, epochs=200):
        Trainer.fit(self, epochs=epochs)
        self.q_prediction_nn = {}
        self.predictionset = self.feature_engineerer.X_test.astype(np.float32)
        self.q_prediction_nn["true"] = self.feature_engineerer.y_test.values

        self.y_pred = self.learner.predict(self.predictionset)

        for i, quantile in enumerate(self.quantiles):

            self.q_prediction_nn[str(quantile)] = self.y_pred[:, i]

        self.q_prediction_nn_df = pd.DataFrame(self.q_prediction_nn)
        print(f"pinball score {model_utils.pinball_score(self.q_prediction_nn_df, quantiles=self.quantiles)}")


# https://github.com/sachinruk/KerasQuantileModel/blob/master/Keras%20Quantile%20Model.ipynb
@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context


class CNN_LSTM(model_utils.BaseModel):
    def __init__(self, feature_engineerer, sequence_length, forecast_length:int = 1,
                 quantiles = np.arange(0.1, 1.0, 0.1).round(2), 
                 cnn_filters:int = 128, activation:str="relu", lstm_layers:int = 200):
        self.feature_engineerer = feature_engineerer
        self.forecast_length = forecast_length
        self.sequence_length = sequence_length
        self.cnn_filters = cnn_filters
        self.activation = activation
        self.lstm_layers = lstm_layers
        self.quantiles = quantiles

    # Build the model
    def create_model(self, input_shape, quantile):
        inputs = Input(shape=input_shape)
        
        x = Conv1D(filters=self.cnn_filters, kernel_size=3, padding='same', activation=self.activation)(inputs)
        x = Dropout(0.3)(x)
        x = Conv1D(filters=self.cnn_filters, kernel_size=3, padding='same', activation=self.activation)(x)
        x = Dropout(0.3)(x)
        x = Conv1D(filters=self.cnn_filters, kernel_size=3, padding='same', activation=self.activation)(x)
        x = Dropout(0.3)(x)

        x = LSTM(self.lstm_layers, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(self.lstm_layers, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(self.lstm_layers, return_sequences=True)(x)
        x = Dropout(0.3)(x)

        attention = Attention()(x)
        outputs = Dense(self.forecast_length)(attention)

        
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss=lambda y,f: model_utils.pinball_loss(y, f, quantile)
                                    )
            
        return model

    def fit_models(self, model_name:str, model_save_dir:str = "CNN_LSTM", verbose:int = 0, lr:float = 0.001, epochs:int = 20, batch_size:int = 64):
        self.all_models = dict()
        for q in self.quantiles:
            model = self.create_model((self.sequence_length, self.feature_engineerer.X_train.shape[1]), quantile = round(q, 2))

            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=lr)

            # Train the model
            history = model.fit(self.feature_engineerer.X_train.reshape(-1, self.sequence_length, self.feature_engineerer.X_train.shape[1]), 
                                self.feature_engineerer.y_train.values.reshape(-1,self.sequence_length, 1), 
                                epochs=epochs, batch_size=batch_size, 
                                validation_data=(self.feature_engineerer.X_val.reshape(-1, self.sequence_length, 74), 
                                                self.feature_engineerer.y_val.values.reshape(-1,self.sequence_length, 1)),
                                callbacks=[early_stopping, reduce_lr],
                                verbose = verbose)
            
            self.all_models[dict] = model

            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            model.save(f'{model_save_dir}/{model_name}_quantile_{q}.h5')
            logging.info(f"Saved model at {model_save_dir}/{model_name}_quantile_{q}.h5")


def get_prediction_for_all_models(feature_engineerer, model_dir, model_name_pattern, quantiles = np.arange(0.1, 1.0, 0.1).round(2)):
    pred_and_true = pd.DataFrame(index = feature_engineerer.y_test.index)

    for q in quantiles:
        m = load_model(f"{model_dir}/{model_name_pattern}_quantile_{q}.h5", compile = False)
        pred = m.predict(feature_engineerer.X_test.reshape(-1, 1, feature_engineerer.X_test.shape[1]))[:, 0]
        if len(pred.shape) > 2:
            pred = pred[:, 0]
        pred_and_true[str(q)] = pred

    pred_and_true["true"] = feature_engineerer.y_test

    return pred_and_true