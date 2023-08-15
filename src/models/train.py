import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from ..data.twitter_dataset import TwitterDataset
from .sentiment_model import SentimentModel
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import logging
import argparse
import os

if __name__ == "__main__":
    logger = logging.getLogger("Train")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.info("Start training")

    parser = argparse.ArgumentParser("trainer")
    parser.add_argument("-lr", "--learning-rate", 
                        help="The learning rate for the optimizer", 
                        type=float, default=0.0008)
    parser.add_argument("-b", "--batch-size", 
                        help="The batch size used during training and validation", 
                        type=int, default=32)
    parser.add_argument("-p", "--patience", 
                        help="The patience value for the early stopping criterion", 
                        type=int, default=5)
    args = parser.parse_args()

    # Define hyperparameters
    input_size = 50
    hidden_size = 20
    learning_rate = args.learning_rate
    epochs = 100
    batch_size = args.batch_size

    # Early stopping params
    patience = args.patience
    min_val_loss = float("inf")
    best_epoch = 0

    # Create model
    model = SentimentModel(input_size, hidden_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dirname = os.path.dirname(__file__)
    
    # Define paths and parameters
    train_path = os.path.join(dirname, '../../data/train.csv')
    val_path = os.path.join(dirname, '../../data/val.csv')

    # Create an instance of the custom dataset
    train = TwitterDataset(train_path)
    val = TwitterDataset(val_path)

    def collate_fn(batch):
        filtered_batch = [item for item in batch if item is not None]
        if len(filtered_batch) == 0:
            return None
        texts, labels, lengths = zip(*filtered_batch)
        padded_texts = pad_sequence(texts, batch_first=True)
        return padded_texts, torch.tensor(labels), torch.tensor(lengths)
    
    logger.info("Load data")
    # Create a DataLoader for the dataset
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    logger.info("Start run")
    mlflow.start_run()  # Start an MLflow run

    # Log hyperparameters
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("patience", patience)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    # Training loop
    for epoch in range(epochs):
        logger.info("Epoch %s", epoch)
        model.train()
        loss_epoch = 0.0
        for batch_text, batch_label, lengths in train_loader:
            optimizer.zero_grad()
            lengths, sorted_indices = lengths.sort(descending=True)
            x = pack_padded_sequence(batch_text[sorted_indices], lengths, batch_first=True, enforce_sorted=False)
            outputs = model(x)
            loss = criterion(outputs, batch_label[sorted_indices].float())
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        
        mlflow.log_metric("loss", loss_epoch / len(train_loader), step=epoch)
        logger.info("Loss: %f", loss_epoch / len(train_loader))
        # Validation
        with torch.no_grad():
            model.eval()
            val_loss_epoch = 0.0
            for batch_X_val, batch_y_val, lengths in val_loader:
                lengths, sorted_indices = lengths.sort(descending=True)
                x = pack_padded_sequence(batch_X_val[sorted_indices], lengths, batch_first=True, enforce_sorted=False)
                val_outputs = model(x)
                val_loss = criterion(val_outputs, batch_y_val[sorted_indices].float())
                val_loss_epoch += val_loss.item()

        avg_epoch_val_loss = val_loss_epoch / len(val_loader)
        mlflow.log_metric("val_loss", avg_epoch_val_loss, step=epoch)
        logger.info("Val Loss: %f", avg_epoch_val_loss)

        # Early stopping criterion to avoid overfitting
        # if the score doesn't improve on the val set after a certain number of epochs, stop training
        if avg_epoch_val_loss < min_val_loss:
            min_val_loss = avg_epoch_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "models/best_model.pth")
        else:
            if epoch - best_epoch >= patience:
                print("Early stopping triggered.")
                break
    
    mlflow.log_artifact("models/best_model.pth", artifact_path="best_weights")
    mlflow.end_run()
