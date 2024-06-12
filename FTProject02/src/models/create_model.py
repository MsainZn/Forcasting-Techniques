import torch
from torch import nn as nn
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# Define the MLP model
class MLP_LF(nn.Module):
    def __init__(self, input_size):
        super(MLP_LF, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Input layer to hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  # Hidden layer to output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the BiLSTM model
class BiLSTM_LF(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.3):
        super(BiLSTM_LF, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM Layer
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True, 
                            bidirectional=True, 
                            dropout=dropout)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        # Initial hidden state and cell state
        # The number of layers in the LSTM multiplied by 2 because it's a bidirectional LSTM (one direction for forward pass and one for backward).
        # x.size(0): The batch size, which is the number of sequences in the input batch.
        # self.hidden_size: The number of hidden units in each LSTM cell.
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size * 2)
        
        # Select the last time step output for each sequence in the batch
        out = out[:, -1, :]  # out: tensor of shape (batch_size, hidden_size * 2)
        
        # Pass through the fully connected layer
        out = self.fc(out)  # out: tensor of shape (batch_size, output_size)
        
        return out
    
# Define the SVM model
class SVM_LF:
    def __init__(
            self, 
            kernel='rbf', 
            C=1.0, 
            epsilon=0.1, 
            gamma='scale', 
            max_iter=1000
        ) -> None:
        
        self.model = SVR(kernel=kernel, 
                         C=C, 
                         epsilon=epsilon, 
                         verbose=True, 
                         gamma=gamma, 
                         max_iter=max_iter)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def calculate_rmse(self, true_values, predicted_values):
        return mean_squared_error(true_values, predicted_values, squared=False)

    def print_usage() -> None:
        print(
            '''
            Usage
                svm_forecaster = SVRForecaster()
                svm_forecaster.train(X_train_sequences, Y_train_sequences)

                svm_train_predictions = svm_forecaster.predict(X_train_sequences)
                svm_test_predictions = svm_forecaster.predict(X_test_sequences)

                train_rmse = svm_forecaster.calculate_rmse(Y_train_sequences, svm_train_predictions)
                test_rmse = svm_forecaster.calculate_rmse(Y_test_sequences, svm_test_predictions)
            '''
        )

# Define the kNN model
class KNN_LF:
    def __init__(
            self, 
            n_neighbors=5, 
            algorithm='auto', 
            weights='uniform', 
            metric='minkowski', 
            leaf_size=30, 
            p=2
        ) -> None:
        
        self.model = KNeighborsRegressor(
                            n_neighbors=n_neighbors, 
                            algorithm=algorithm, 
                            weights=weights, 
                            metric=metric,
                            leaf_size= leaf_size,
                            p=p)

    def train(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def calculate_rmse(self, true_values, predicted_values):
        return mean_squared_error(true_values, predicted_values, squared=False)

    def print_usage() -> None:
        print(
            '''
            Usage
                knn_forecaster = KNNForecaster()
                knn_forecaster.train(X_train_sequences, Y_train_sequences)

                knn_train_predictions = knn_forecaster.predict(X_train_sequences)
                knn_test_predictions = knn_forecaster.predict(X_test_sequences)

                train_rmse_knn = knn_forecaster.calculate_rmse(Y_train_sequences, knn_train_predictions)
                test_rmse_knn = knn_forecaster.calculate_rmse(Y_test_sequences, knn_test_predictions)

                print("Train RMSE (kNN):", train_rmse_knn)
                print("Test RMSE (kNN):", test_rmse_knn)
            '''
        )

# Define Random Forest model
class RandomForest_LF:
    def __init__(
            self, 
            n_estimators=100, 
            max_depth=None, 
            random_state=None
        )->None:
        self.model = RandomForestRegressor(
                        n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        random_state=random_state,
                        verbose=1
                    )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def calculate_rmse(self, true_values, predicted_values):
        return mean_squared_error(true_values, predicted_values, squared=False)

    def print_usage() -> None:
        print(
            '''
            # Usage
                rf_forecaster = RandomForestForecaster()
                rf_forecaster.train(X_train_sequences, Y_train_sequences)

                rf_train_predictions = rf_forecaster.predict(X_train_sequences)
                rf_test_predictions = rf_forecaster.predict(X_test_sequences)

                train_rmse_rf = rf_forecaster.calculate_rmse(Y_train_sequences, rf_train_predictions)
                test_rmse_rf = rf_forecaster.calculate_rmse(Y_test_sequences, rf_test_predictions)

                print("Train RMSE (Random Forest):", train_rmse_rf)
                print("Test RMSE (Random Forest):", test_rmse_rf)
            '''
        )

# Define Transformer model (Based on Attention is All You Need)
class Transformer_LF(nn.Module):
    def __init__(
            self, 
            input_size, 
            num_layers=6, 
            hidden_size=128, 
            num_heads=8, 
            dropout=0.1
        )->None:
        super(Transformer_LF, self).__init__()
        
        self.encoder_layer = nn.TransformerEncoderLayer(
                                        d_model=input_size, 
                                        nhead=num_heads, 
                                        dim_feedforward=hidden_size, 
                                        dropout=dropout
                            )
        self.transformer_encoder = nn.TransformerEncoder(
                                            self.encoder_layer, 
                                            num_layers=num_layers
                                        )
        
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x
    
# GEEKY VERSION FOR A GEEK LIKE ME!!!!
class BiLSTM_geeky(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.3):
        super(BiLSTM_geeky, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM Cell
        self.lstm_f = nn.LSTMCell(input_size, hidden_size)
        self.lstm_b = nn.LSTMCell(input_size, hidden_size)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        # Initial hidden state and cell state
        h0_f = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        c0_f = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        h0_b = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        c0_b = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        
        # Forward and backward hidden state and cell state lists
        h_f, c_f = [h0_f], [c0_f]
        h_b, c_b = [h0_b], [c0_b]
        
        # Forward pass through LSTM
        for i in range(x.size(1)):
            h_f_next, c_f_next = self.lstm_f(x[:, i, :], (h_f[-1], c_f[-1]))
            h_f.append(h_f_next)
            c_f.append(c_f_next)
            
        # Backward pass through LSTM
        for i in reversed(range(x.size(1))):
            h_b_next, c_b_next = self.lstm_b(x[:, i, :], (h_b[-1], c_b[-1]))
            h_b.append(h_b_next)
            c_b.append(c_b_next)
        
        # Concatenate forward and backward hidden states
        h_concat = torch.cat((h_f[-1], h_b[-1]), dim=1)
        
        # Pass through the fully connected layer
        out = self.fc(h_concat)
        
        return out
    