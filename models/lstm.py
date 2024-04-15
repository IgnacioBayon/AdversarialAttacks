import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
import nltk
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import functional as F
from torchtext.data.utils import get_tokenizer


def load_data():
    data = pd.read_csv('data/IMDBDataset.csv')
    data = data.sample(frac=1).reset_index(drop=True)
    data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    return data

def clean_text(review):
    # Remove HTML tags
    review = re.sub(r'<.*?>', '', review)
    # remove urls
    review = re.sub(r'http\S+', '', review)
    # remove non-alphabetic characters
    review = re.sub("[^a-zA-Z]"," ",review)
    # remove whitespaces
    review = ' '.join(review.split())
    # convert text to lowercase
    review = review.lower()
    return review

def preprocess_data(data):
    data['review'] = data['review'].apply(clean_text)
    return data

def remove_stopwords(text):
    words = [word for word in text.split() if word not in nltk.corpus.stopwords.words('english')]
    return ' '.join(words)

def remove_stopwords_from_data(data):
    data['review'] = data['review'].apply(remove_stopwords)
    return data

def tokenize_data(data):
    tokenizer = get_tokenizer('basic_english')
    tokenizer.fit_on_texts(data['review'])
    return tokenizer

def lemma(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    st = ''
    for word in text.split():
        st += lemmatizer.lemmatize(word) + ' '
    return st

def lemmatize_data(data):
    data['review'] = data['review'].apply(lemma)
    return data

def encode_data(data, tokenizer):
    data['review'] = tokenizer.texts_to_sequences(data['review'])
    return data

def pad_data(data):
    data['review'] = data['review'].apply(lambda x: x + [0] * (512 - len(x)))
    return data

def split_data(data):
    X = np.array(data['review'].tolist())
    y = np.array(data['sentiment'].tolist())
    return train_test_split(X, y, test_size=0.2, random_state=42)

class IMDBDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x
    
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y.unsqueeze(1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def predict(model, test_loader, device):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            output = model(X)
            y_pred.extend(output.squeeze().tolist())
    return y_pred

def main():
    data = load_data()
    data = preprocess_data(data)
    data = remove_stopwords_from_data(data)
    data = lemmatize_data(data)
    tokenizer = tokenize_data(data)
    data = encode_data(data, tokenizer)
    data = pad_data(data)
    X_train, X_val, y_train, y_val = split_data(data)
    
    train_dataset = IMDBDataset(X_train, y_train)
    val_dataset = IMDBDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(20000, 128, 128).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    y_pred = predict(model, val_loader, device)
    y_pred = [1 if x > 0 else 0 for x in y_pred]
    print(classification_report(y_val, y_pred))
    print(f'Accuracy: {accuracy_score(y_val, y_pred):.4f}')

if __name__ == '__main__':
    main()
