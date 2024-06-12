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

'''