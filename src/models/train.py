import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import mlflow
import mlflow.pytorch

class SentimentModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SentimentModel, self).__init__()
        self.masking = nn.Sequential(nn.Dropout(p=0.0), nn.Identity())  # No masking in PyTorch
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, 1)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.masking(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    # Define hyperparameters
    input_size = 50
    hidden_size = 20
    learning_rate = 0.0008
    epochs = 100
    batch_size = 32

    # Create model
    model = SentimentModel(input_size, hidden_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert data to PyTorch tensors if they are not already
    
    mlflow.start_run()  # Start an MLflow run

    # Log hyperparameters
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            mlflow.log_metric("loss", loss, step=epoch)
            loss.backward()
            optimizer.step()

        # Validation
        with torch.no_grad():
            model.eval()
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)  # Calculate validation loss
            mlflow.log_metric("val_loss", val_loss, step=epoch)
    
    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()
