import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
import os
from .sentiment_model import SentimentModel
from ..data.twitter_dataset import TwitterDataset
from ..data.process import collate_fn
import logging

if __name__ == "__main__":
    logger = logging.getLogger("Eval")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.info("Start evaluation")

    model = SentimentModel(50, 20)
    dirname = os.path.dirname(__file__)
    model_path = "best_model.pth"
    model.load_state_dict(torch.load(model_path))

    test_path = os.path.join(dirname, '../../data/test.csv')

    # Create an instance of the custom dataset
    test = TwitterDataset(test_path)
    batch_size = 32
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        model.eval()
        loss = 0.0
        correct = 0
        for batch_X_test, batch_y_test, lengths in test_loader:
            lengths, sorted_indices = lengths.sort(descending=True)
            x = pack_padded_sequence(batch_X_test[sorted_indices], lengths, batch_first=True, enforce_sorted=False)
            outputs = model(x)
            test_loss = criterion(outputs, batch_y_test[sorted_indices].float())
            preds = (outputs>0.5).float()
            loss += test_loss.item()
            correct += torch.sum(preds == batch_y_test[sorted_indices].float())
        
        logger.info("Loss: %f", loss / len(test_loader))
        logger.info("Accuracy: %f", correct / (len(test_loader)*batch_size))

        