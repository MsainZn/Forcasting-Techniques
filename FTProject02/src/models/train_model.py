import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error

def train_DL_model(
        model, 
        criterion, 
        optimizer, 
        train_loader, 
        test_loader, 
        num_epochs=10
    )->tuple:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        mse_trn, r2_trn, avg_loss_trn = evaluate_DL_model(model, criterion, train_loader)
        mse_tst, r2_tst, avg_loss_tst = evaluate_DL_model(model, criterion, test_loader)

        print(f'MSE-Train = {mse_trn:5.4f} R2-Score = {r2_trn:5.4f} AVG-Loss = {avg_loss_trn:5.4f}')
        print(f'MSE-Test  = {mse_tst:5.4f} R2-Score = {r2_tst:5.4f} AVG-Loss = {avg_loss_tst:5.4f}')

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    mse_trn, r2_trn, avg_loss_trn = evaluate_DL_model(model, criterion, train_loader)
    mse_tst, r2_tst, avg_loss_tst = evaluate_DL_model(model, criterion, test_loader)

    print(f'MSE-Train-Final = {mse_trn:5.4f} R2-Score = {r2_trn:5.4f} AVG-Loss = {avg_loss_trn:5.4f}')
    print(f'MSE-Test-Final  = {mse_tst:5.4f} R2-Score = {r2_tst:5.4f} AVG-Loss = {avg_loss_tst:5.4f}')

    return model, mse_trn, r2_trn, avg_loss_trn, mse_tst, r2_tst, avg_loss_tst

def evaluate_DL_model(
            model, 
            criterion, 
            data_loader
    )->tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    mse = mean_squared_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)
    avg_loss = total_loss / len(data_loader.dataset)
    
    return mse, r2, avg_loss

def train_ML_model(
        model, 
        x_trn, 
        x_tst, 
        y_trn, 
        y_tst
    ):
    model.fit(x_trn, y_trn)

    rmse_trn, r2_trn, avg_loss_trn = evaluate_ML_model(model, x_trn, y_trn)
    rmse_tst, r2_tst, avg_loss_tst = evaluate_ML_model(model, x_tst, y_tst)

    print(f'Train: MSE = {rmse_trn:5.4f} R2-Score = {r2_trn:5.4f} AVG-Loss = {avg_loss_trn:5.4f}')
    print(f'Test:  MSE = {rmse_tst:5.4f} R2-Score = {r2_tst:5.4f} AVG-Loss = {avg_loss_tst:5.4f}')

    return model

def evaluate_ML_model(
        model, 
        x_tst, 
        y_tst
    ):
    predictions = model.predict(x_tst)
    rmse = model.calculate_rmse(y_tst, predictions)
    r2 = r2_score(y_tst, predictions)
    avg_loss = mean_squared_error(y_tst, predictions)

    return rmse, r2, avg_loss

