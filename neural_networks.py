from functools import partial
from itertools import chain
import torch
import torch.nn as nn
import numpy as np
import model_utils
import pandas as pd



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
        # LSTMs cannot be used inside nn.Sequential because they return tuples
        self.lstm1 = nn.LSTM(self.len_features, 50, dropout=0.3, batch_first=True)
        self.lstm2 = nn.LSTM(50, 25, dropout=0.3, batch_first=True)
        self.lstm3 = nn.LSTM(25, 10, dropout=0.5, batch_first=True)
        
        # Final layers for quantiles
        final_layers = [
            nn.Linear(10, 1) for _ in range(len(self.quantiles))
        ]
        self.final_layers = nn.ModuleList(final_layers)

    def forward(self, x):
        # x is of shape (batch_size, seq_len, len_features)
        
        # LSTM layers - extracting the output (not hidden and cell states)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        
        # Apply final layers to get quantile outputs
        quantile_outputs = [layer(out) for layer in self.final_layers]
        
        return torch.cat(quantile_outputs, dim=-1)
        
    def init_weights(self):
        for m in chain(self.final_layers):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)        
        
    # def forward(self, x):
    #     tmp_ = self.base_model(x)
    #     return torch.cat([layer(tmp_) for layer in self.final_layers], dim=1)
    

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
    
class Learner(model_utils.BaseModel):
    def __init__(self, model, optimizer_class, loss_func):
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
    

class Trainer(model_utils.BaseModel):
    def __init__(self, feature_engineerer, model_class, quantiles, optimizer_class=torch.optim.Adam, in_shape=50, dropout=0.1, weight_decay=1e-6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantiles = quantiles
        self.feature_engineerer = feature_engineerer
        self.loss_func = QuantileLoss(self.quantiles)
        
        # Define model, loss function, and optimizer
        self.model = model_class(self.quantiles,in_shape=in_shape, dropout=dropout, len_features = len(self.feature_engineerer.features_after_fe)).to(self.device)
        self.learner = Learner(
            self.model, 
            partial(optimizer_class, weight_decay=weight_decay),
            self.loss_func
        )

    def fit(self, epochs=150, batch_size=144):
        self.X_train = self.feature_engineerer.X_train
        self.y_train = self.feature_engineerer.y_train.values

        self.learner.fit(self.X_train, self.y_train, epochs, batch_size)

    def predict(self, prediction_set):
        self.y_pred = self.learner.predict(prediction_set.astype(np.float32))
        return self.y_pred
    
    def train_and_test(self, epochs=150, batch_size=144):
        Trainer.fit(self, epochs=epochs, batch_size=batch_size)
        self.q_prediction_nn = {}
        self.predictionset = self.feature_engineerer.X_test.astype(np.float32)
        self.q_prediction_nn["true"] = self.feature_engineerer.y_test.values

        self.y_pred = self.learner.predict(self.predictionset)

        for i, quantile in enumerate(self.quantiles):

            self.q_prediction_nn[str(quantile)] = self.y_pred[:, i]

        self.q_prediction_nn_df = pd.DataFrame(self.q_prediction_nn)
        print(f"pinball score {model_utils.pinball_score(self.q_prediction_nn_df, quantiles=self.quantiles)}")