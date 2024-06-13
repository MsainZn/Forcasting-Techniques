import torch
from torch import nn
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import yaml
import os

def load_yaml_parameters(filepath, model_name):
    """ Load parameters from a YAML file. """
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config.get(model_name, {})

# Define the MLP model
class MLP_LF(nn.Module):
    def __init__(self, input_size:int, config_path:str='model_config.yaml'):
        super(MLP_LF, self).__init__()
        
        params = load_yaml_parameters(config_path, 'MLP_LF')
        hidden_dim = params.get('hidden_dim', 1)

        self.fc1 = nn.Linear(input_size, hidden_dim)  # Input layer to hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # Hidden layer to output layer

    def forward(self, x:torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the BiLSTM model
class BiLSTM_LF(nn.Module):
    def __init__(self, input_size:int, config_path:str='model_config.yaml'):
        super(BiLSTM_LF, self).__init__()

        params = load_yaml_parameters(config_path, 'BiLSTM_LF')
        hidden_size = params.get('hidden_size', 64)
        num_layers = params.get('num_layers', 2)
        dropout = params.get('dropout', 0.3)
        output_size = params.get('output_size', 1)

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
        
    def forward(self, x:torch.tensor):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Select the last time step output for each sequence
        out = self.fc(out)
        
        return out

# Define Transformer model (Based on Attention is All You Need)
class Transformer_LF(nn.Module):
    def __init__(self, input_size:int, config_path:str='model_config.yaml'):
        super(Transformer_LF, self).__init__()

        params = load_yaml_parameters(config_path, 'Transformer_LF')
        num_layers = params.get('num_layers', 6)
        hidden_size = params.get('hidden_size', 128)
        num_heads = params.get('num_heads', 8)
        dropout = params.get('dropout', 0.1)

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

    def forward(self, x:torch.tensor):
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

# Define the SVM model
class SVM_LF:
    def __init__(self, config_path:str='model_config.yaml'):
        params = load_yaml_parameters(config_path, 'SVM_LF')
        kernel = params.get('kernel', 'rbf')
        C = params.get('C', 1.0)
        epsilon = params.get('epsilon', 0.1)
        gamma = params.get('gamma', 'scale')
        max_iter = params.get('max_iter', 1000)
        
        self.model = SVR(
                         kernel=kernel, 
                         C=C, 
                         epsilon=epsilon, 
                         verbose=True, 
                         gamma=gamma, 
                         max_iter=max_iter)

    def calculate_rmse(self, true_values, predicted_values):
        return mean_squared_error(true_values, predicted_values, squared=False)

    def print_usage() -> None:
        print(
            '''
            Usage
                svm_forecaster = SVM_LF()
                svm_forecaster.train(X_train_sequences, Y_train_sequences)

                svm_train_predictions = svm_forecaster.predict(X_train_sequences)
                svm_test_predictions = svm_forecaster.predict(X_test_sequences)

                train_rmse = svm_forecaster.calculate_rmse(Y_train_sequences, svm_train_predictions)
                test_rmse = svm_forecaster.calculate_rmse(Y_test_sequences, svm_test_predictions)
            '''
        )

# Define the kNN model
class KNN_LF:
    def __init__(self, config_path:str='model_config.yaml'):
        params = load_yaml_parameters(config_path, 'KNN_LF')
        n_neighbors = params.get('n_neighbors', 5)
        algorithm = params.get('algorithm', 'auto')
        weights = params.get('weights', 'uniform')
        metric = params.get('metric', 'minkowski')
        leaf_size = params.get('leaf_size', 30)
        p = params.get('p', 2)
        
        self.model = KNeighborsRegressor(
                            n_neighbors=n_neighbors, 
                            algorithm=algorithm, 
                            weights=weights, 
                            metric=metric,
                            leaf_size= leaf_size,
                            p=p)

    def calculate_rmse(self, true_values, predicted_values):
        return mean_squared_error(true_values, predicted_values, squared=False)

    def print_usage() -> None:
        print(
            '''
            Usage
                knn_forecaster = KNN_LF()
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
    def __init__(self, config_path:str='model_config.yaml'):
        params = load_yaml_parameters(config_path, 'RandomForest_LF')
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth', None)
        random_state = params.get('random_state', None)
        
        self.model = RandomForestRegressor(
                        n_estimators=n_estimators, 
                        max_depth=max_depth, 
                        random_state=random_state,
                        verbose=1
                    )

    def calculate_rmse(self, true_values, predicted_values):
        return mean_squared_error(true_values, predicted_values, squared=False)

    def print_usage() -> None:
        print(
            '''
            # Usage
                rf_forecaster = RandomForest_LF()
                rf_forecaster.train(X_train_sequences, Y_train_sequences)

                rf_train_predictions = rf_forecaster.predict(X_train_sequences)
                rf_test_predictions = rf_forecaster.predict(X_test_sequences)

                train_rmse_rf = rf_forecaster.calculate_rmse(Y_train_sequences, rf_train_predictions)
                test_rmse_rf = rf_forecaster.calculate_rmse(Y_test_sequences, rf_test_predictions)

                print("Train RMSE (Random Forest):", train_rmse_rf)
                print("Test RMSE (Random Forest):", test_rmse_rf)
            '''
        )
