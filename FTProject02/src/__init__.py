import runpy
from features.build_features import Read_Pickle

# Configs
batch_size = 64
path_to_data"../../data/raw"

exec_files = ['data/make_dataset.py', 'features/build_features.py']

for file in exec_files:
    runpy.run_path(file)

Read_Pickle()

# # Create TensorDatasets
# train_dataset = TensorDataset(train_sequences_tensor, train_targets_tensor)
# test_dataset = TensorDataset(test_sequences_tensor, test_targets_tensor)

# # DataLoader
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


'''
# Define models
models = {
    "Transformer": Transformer_LF(input_size, num_layers, hidden_size, num_heads, dropout),
    "BiLSTM": BiLSTM_LF(input_size, hidden_size, num_layers, output_size, dropout),
    "MLP": MLP_LF(input_size)
}

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

# Define models
models = {
    "SVM": SVM_LF(),
    "KNN": KNN_LF(),
    "RandomForest": RandomForest_LF()
}

# Training loop
for name, model in models.items():
    print(f"Training {name}...")
    train_model(model, X_train, y_train)
    print(f"{name} trained successfully.")
    
    print(f"Evaluating {name}...")
    rmse = evaluate_model(model, X_test, y_test)
    print(f"{name} RMSE: {rmse}")

'''