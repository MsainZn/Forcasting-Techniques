import runpy
import subprocess
import yaml
from torch.utils.data import DataLoader, TensorDataset

from features.build_features import Read_Pickle
from models.create_model import SVM_LF, KNN_LF, RandomForest_LF, BiLSTM_LF, MLP_LF, BiLSTM_LF, Transformer_LF


# Load the configuration
with open('init_config.yaml', 'r') as f:
    default_params = yaml.safe_load(f)

# Run Necessary Preprocessing
for file in default_params["exec_paths"]:
    subprocess.run(["python", file])


# Readv Compressed Data
# data_library = Read_Pickle(default_params["data_path"])

# tensor_library = {}
# for country in default_params['ctr_code']:
#     for period in ['Hourly', 'Daily', 'Weekly', 'Monthly']:
#         # Create TensorDatasets
#         train_dataset= TensorDataset(data_library[f'{country}_{period}']["trX"], data_library[f'{country}_{period}']["trY"])
#         test_dataset  = TensorDataset(data_library[f'{country}_{period}']["tsX"], data_library[f'{country}_{period}']["tsY"])
#         # DataLoader
#         tensor_library[f'{country}_{period}']['trn'] = DataLoader(train_dataset, batch_size=default_params["batch_size"], shuffle=True)
#         tensor_library[f'{country}_{period}']['tst'] = DataLoader(test_dataset,  batch_size=default_params["batch_size"],  shuffle=False)


# models_DL = {
#     "Transformer": Transformer_LF(input_size, num_layers, hidden_size, num_heads, dropout),
#     "BiLSTM": BiLSTM_LF(input_size, hidden_size, num_layers, output_size, dropout),
#     "MLP": MLP_LF(input_size)
# }

# # Define models
# models_ML = {
#     "SVM": SVM_LF(),
#     "KNN": KNN_LF(),
#     "RandomForest": RandomForest_LF()
# }


'''
# Define models


# Define criterion and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam

# Training loop
for name, model in models.items():
    print(f"Training {name}...")
    optimizer_instance = optimizer(model.parameters(), lr=0.001)
    train_model(model, criterion, optimizer_instance, train_loader)
    print(f"{name} trained successfully.")

****************************************************************************


# Training loop
for name, model in models.items():
    print(f"Training {name}...")
    train_model(model, X_train, y_train)
    print(f"{name} trained successfully.")
    
    print(f"Evaluating {name}...")
    rmse = evaluate_model(model, X_test, y_test)
    print(f"{name} RMSE: {rmse}")

'''