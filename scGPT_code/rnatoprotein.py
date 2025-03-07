import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import h5py
import tables 
import json
from scgpt.tasks import embed_data
import os
from scipy.stats import pearsonr

NUM_TRAIN = 1536 
NUM_TEST = 512 
TOTAL_SAMPLES = NUM_TRAIN + NUM_TEST 
NUM_GENES = 512  
EMBEDDING_CHECKPOINT = f"cell_embeddings_checkpoint_{TOTAL_SAMPLES}_{NUM_GENES}.npy"
MODEL_CHECKPOINT = "nn_model_checkpoint.pth"
EPOCHS = 50
BATCH_SIZE = 32  
LEARNING_RATE = 0.001


def load_and_embed_data(h5_file_path, model_dir):
    if os.path.exists(EMBEDDING_CHECKPOINT):
        embeddings = np.load(EMBEDDING_CHECKPOINT)
        if embeddings.shape[0] == TOTAL_SAMPLES and embeddings.shape[1] == NUM_GENES:
            print(f" Using saved embeddings. Shape: {embeddings.shape}")
            return embeddings
        else:
            print(f" Mismatch in saved embeddings. Expected: {TOTAL_SAMPLES}, Found: {embeddings.shape[0]}. Regenerating...")

    with h5py.File(h5_file_path, 'r') as file:
        data_values = file['train_cite_inputs/block0_values'][:TOTAL_SAMPLES, :NUM_GENES]
        items = file['train_cite_inputs/block0_items'][:NUM_GENES]
        axis0 = file['train_cite_inputs/axis0'][:TOTAL_SAMPLES]

    items = [item.decode('utf-8') if isinstance(item, bytes) else item for item in items]
    axis0 = [idx.decode('utf-8') if isinstance(idx, bytes) else idx for idx in axis0]

    print(f"Selected First {TOTAL_SAMPLES} Cells and {NUM_GENES} Genes. Shape: {data_values.shape}")

    df = pd.DataFrame(data=data_values, index=axis0, columns=items)
    adata = sc.AnnData(X=df.values, var=pd.DataFrame(index=df.columns), obs=pd.DataFrame(index=df.index))
    vocab_file = Path(model_dir) / "vocab.json"
    vocab = json.load(open(vocab_file))


    adata.var.loc[:, "gene_name"] = adata.var.index.str.split("_").str[-1]
    adata.var.loc[:, "id_in_vocab"] = [vocab.get(gene, -1) for gene in adata.var["gene_name"]]
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    valid_gene_count = np.sum(adata.var["id_in_vocab"] >= 0)
    print(f"Valid Genes After Mapping: {valid_gene_count} / {NUM_GENES}")

    batch_size = 5
    embeddings = []
    for i in range(0, adata.shape[0], batch_size):
        batch_adata = adata[i: i + batch_size]
        batch_embed_adata = embed_data(batch_adata, model_dir, gene_col="gene_name", batch_size=batch_size, device="cpu")
        embeddings.append(batch_embed_adata.obsm["X_scGPT"])
        np.save(EMBEDDING_CHECKPOINT, np.vstack(embeddings))
        print(f"Saved embeddings after {i + batch_size} samples")

    embeddings = np.vstack(embeddings)
    return embeddings


def load_protein_labels(h5_file_path):
    with h5py.File(h5_file_path, 'r') as file:
        Y_protein = file['train_cite_targets/block0_values'][:TOTAL_SAMPLES]
    print(f"Protein Labels Loaded. Shape: {Y_protein.shape}")
    return Y_protein

class ProteinPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProteinPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_nn(X_train, Y_train, X_test, Y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train, Y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_test, Y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(Y_test, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ProteinPredictor(input_dim=X_train.shape[1], output_dim=Y_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    if os.path.exists(MODEL_CHECKPOINT):
        checkpoint = torch.load(MODEL_CHECKPOINT)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        with torch.no_grad():
            Y_pred_test = model(X_test).cpu().numpy()
            Y_test_np = Y_test.cpu().numpy()
            pearson_corr = np.mean([pearsonr(Y_pred_test[:, i], Y_test_np[:, i])[0] for i in range(Y_test_np.shape[1])])
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Pearson Corr: {pearson_corr:.4f}")

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, MODEL_CHECKPOINT)
    return model

X_all = load_and_embed_data("../../Project/train_cite_inputs.h5", "../scGPT_human")
Y_all = load_protein_labels("../../Project/train_cite_targets.h5")
X_train, X_test = X_all[:NUM_TRAIN], X_all[NUM_TRAIN:]
Y_train, Y_test = Y_all[:NUM_TRAIN], Y_all[NUM_TRAIN:]
train_nn(X_train, Y_train, X_test, Y_test)