import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ..data.process import create_embedding
import gensim.downloader as api
from torch.nn.utils.rnn import pad_sequence

class TwitterDataset(Dataset):
    def __init__(self, csv_file):
        # I need to use a converter since the tokenized sentences saved as lists
        self.data = pd.read_csv(csv_file, 
                                converters={"tweet": lambda x: x.strip("[]").replace("'", "").split(", ")}
        )
        self.w2v = api.load("glove-twitter-50")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        text = sample['tweet']
        label = sample['target']
        text = create_embedding(text, self.w2v)
        if text.shape[0] == 0:
            return None
        return text, label, text.shape[0]


if __name__ == '__main__':
    def collate_fn(batch):
        filtered_batch = [item for item in batch if item is not None]
        if len(filtered_batch) == 0:
            return None
        texts, labels, lengths = zip(*filtered_batch)
        padded_texts = pad_sequence(texts, batch_first=True)
        return padded_texts, torch.tensor(labels), torch.tensor(lengths)
    train_path = "../../data/train.csv"
    batch_size = 32
    train = TwitterDataset(train_path)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    for batch in train_loader:
        if batch == None:
            break
        batch_text, batch_label, lengths = batch
        print(batch_text.shape)
        print(batch_label.shape)
        print(lengths)
        break