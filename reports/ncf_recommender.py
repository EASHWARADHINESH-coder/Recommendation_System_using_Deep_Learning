# IMPORT LIBRARIES
import copy
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_loader import load_users, load_movies, load_ratings

# PATHS

raw_data_path = Path(r"D:\DS PROJECTS SUBMISSION\Recommendation_System_Final\data\raw")
processed_data_path = Path(r"D:\DS PROJECTS SUBMISSION\Recommendation_System_Final\data\processed")
processed_data_path.mkdir(parents=True, exist_ok=True)

# LOAD DATA
def load_data():
    users_df = load_users()
    movies_df = load_movies()
    ratings_df = load_ratings()
    return users_df, movies_df, ratings_df

# ENCODE USER / ITEM IDS

def encode_ids(ratings_df: pd.DataFrame):
    ratings_copy = ratings_df.copy()

    unique_user_ids = sorted(ratings_copy["user_id"].unique())
    unique_movie_ids = sorted(ratings_copy["movie_id"].unique())

    user_to_index = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    item_to_index = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}

    index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}
    index_to_item = {idx: movie_id for movie_id, idx in item_to_index.items()}

    ratings_copy["user_idx"] = ratings_copy["user_id"].map(user_to_index)
    ratings_copy["item_idx"] = ratings_copy["movie_id"].map(item_to_index)

    with open(processed_data_path / "ncf_user_to_index.pkl", "wb") as f:
        pickle.dump(user_to_index, f)

    with open(processed_data_path / "ncf_item_to_index.pkl", "wb") as f:
        pickle.dump(item_to_index, f)

    with open(processed_data_path / "ncf_index_to_user.pkl", "wb") as f:
        pickle.dump(index_to_user, f)

    with open(processed_data_path / "ncf_index_to_item.pkl", "wb") as f:
        pickle.dump(index_to_item, f)

    return ratings_copy, user_to_index, item_to_index, index_to_user, index_to_item

# TRAIN/ VALID SPLIT

def train_valid_split(
    ratings_df: pd.DataFrame,
    valid_ratio: float = 0.2,
    random_state: int = 42
):
    shuffled_df = ratings_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    split_index = int(len(shuffled_df) * (1 - valid_ratio))

    train_df = shuffled_df.iloc[:split_index].copy()
    valid_df = shuffled_df.iloc[split_index:].copy()

    return train_df, valid_df

# DATASET CLASS

class RatingsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feedback_type: str = "explicit"):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)

        if feedback_type == "explicit":
            self.labels = torch.tensor(df["rating"].values, dtype=torch.float32)

        elif feedback_type == "implicit":
            if "implicit_feedback" in df.columns:
                labels = df["implicit_feedback"].values
            else:
                labels = (df["rating"].values >= 4).astype(np.float32)

            self.labels = torch.tensor(labels, dtype=torch.float32)

        else:
            raise ValueError("feedback_type must be 'explicit' or 'implicit'.")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

# NCF MODEL

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_dims: list = None,
        dropout: float = 0.25
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # USER EMBEDDINGS
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # ITEM EMBEDDINGS
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # FULLY CONNECTED LAYERS
        layers = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)

        x = torch.cat([user_vec, item_vec], dim=1)
        x = self.mlp(x)
        output = self.output_layer(x).squeeze(1)

        return output

# EARLY STOPPING

class EarlyStopping:
    def __init__(self, patience: int = 4, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, valid_loss: float, model: nn.Module):
        if valid_loss < self.best_loss - self.min_delta:
            self.best_loss = valid_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False

        self.counter += 1
        return self.counter >= self.patience

# TRAIN NCF MODEL

def train_ncf_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    feedback_type: str = "explicit",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 20,
    patience: int = 4,
    device: str = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    if feedback_type == "explicit":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    early_stopper = EarlyStopping(patience=patience)

    history = {
        "train_loss": [],
        "valid_loss": []
    }

    for epoch in range(1, epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        train_losses = []

        for users, items, labels in train_loader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(users, items)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # ---------------- VALID ----------------
        model.eval()
        valid_losses = []

        with torch.no_grad():
            for users, items, labels in valid_loader:
                users = users.to(device)
                items = items.to(device)
                labels = labels.to(device)

                outputs = model(users, items)
                loss = criterion(outputs, labels)

                valid_losses.append(loss.item())

        avg_valid_loss = float(np.mean(valid_losses))

        history["train_loss"].append(avg_train_loss)
        history["valid_loss"].append(avg_valid_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Valid Loss: {avg_valid_loss:.4f}"
        )

        should_stop = early_stopper.step(avg_valid_loss, model)
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)

    return model, history

# EVALUATION

def evaluate_explicit_model(model, data_loader, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    preds = []
    trues = []

    with torch.no_grad():
        for users, items, labels in data_loader:
            users = users.to(device)
            items = items.to(device)

            outputs = model(users, items).cpu().numpy()

            preds.extend(outputs.tolist())
            trues.extend(labels.numpy().tolist())

    preds = np.array(preds)
    trues = np.array(trues)

    mse = np.mean((preds - trues) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - trues))

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae)
    }


def evaluate_implicit_model(model, data_loader, threshold: float = 0.5, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    probs = []
    trues = []

    with torch.no_grad():
        for users, items, labels in data_loader:
            users = users.to(device)
            items = items.to(device)

            logits = model(users, items)
            pred_probs = torch.sigmoid(logits).cpu().numpy()

            probs.extend(pred_probs.tolist())
            trues.extend(labels.numpy().tolist())

    probs = np.array(probs)
    trues = np.array(trues)
    preds = (probs >= threshold).astype(int)

    accuracy = float((preds == trues).mean())

    return {
        "Accuracy": accuracy
    }

# RECOMMENDATION FUNCTION

def recommend_ncf(
    model: nn.Module,
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_to_index: dict,
    item_to_index: dict,
    top_n: int = 10,
    feedback_type: str = "explicit",
    device: str = None
) -> pd.DataFrame:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if user_id not in user_to_index:
        raise ValueError(f"user_id {user_id} not found in training mappings.")

    model.eval()
    model.to(device)

    user_idx = user_to_index[user_id]

    already_rated_movies = ratings_df.loc[
        ratings_df["user_id"] == user_id, "movie_id"
    ].unique().tolist()

    candidate_movie_ids = [
        movie_id for movie_id in item_to_index.keys()
        if movie_id not in already_rated_movies
    ]

    candidate_item_indices = torch.tensor(
        [item_to_index[movie_id] for movie_id in candidate_movie_ids],
        dtype=torch.long,
        device=device
    )

    user_tensor = torch.full(
        (len(candidate_movie_ids),),
        fill_value=user_idx,
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        scores = model(user_tensor, candidate_item_indices)

        if feedback_type == "implicit":
            scores = torch.sigmoid(scores)

        scores = scores.cpu().numpy()

    recommendation_df = pd.DataFrame({
        "movie_id": candidate_movie_ids,
        "predicted_score": scores
    })

    recommendation_df = recommendation_df.merge(
        movies_df[["movie_id", "title", "genre", "language", "imdb_rating"]],
        on="movie_id",
        how="left"
    )

    recommendation_df = recommendation_df.sort_values(
        by="predicted_score",
        ascending=False
    ).head(top_n)

    return recommendation_df

# SAVE ARTIFACTS

def save_ncf_artifacts(model, history, feedback_type: str):
    torch.save(model.state_dict(), processed_data_path / f"ncf_model_{feedback_type}.pt")

    with open(processed_data_path / f"ncf_history_{feedback_type}.pkl", "wb") as f:
        pickle.dump(history, f)

# MAIN

if __name__ == "__main__":
    # SETTINGS
    FEEDBACK_TYPE = "explicit"      # "explicit" or "implicit"
    EMBEDDING_DIM = 32
    HIDDEN_DIMS = [128, 64, 32]
    DROPOUT = 0.25
    BATCH_SIZE = 1024
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE = 4

    # LOAD DATA
    users_df, movies_df, ratings_df = load_data()

    # ENCODE IDS
    encoded_ratings_df, user_to_index, item_to_index, index_to_user, index_to_item = encode_ids(ratings_df)

    # SPLIT
    train_df, valid_df = train_valid_split(
        encoded_ratings_df,
        valid_ratio=0.2,
        random_state=42
    )

    # DATASET
    train_dataset = RatingsDataset(train_df, feedback_type=FEEDBACK_TYPE)
    valid_dataset = RatingsDataset(valid_df, feedback_type=FEEDBACK_TYPE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # MODEL
    model = NeuralCollaborativeFiltering(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT
    )

    # TRAIN
    model, history = train_ncf_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        feedback_type=FEEDBACK_TYPE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        patience=PATIENCE
    )

    # SAVE
    save_ncf_artifacts(model, history, feedback_type=FEEDBACK_TYPE)

    # EVALUATE
    if FEEDBACK_TYPE == "explicit":
        metrics = evaluate_explicit_model(model, valid_loader)
        print("\nNCF Explicit Feedback Metrics:")
        print(metrics)
    else:
        metrics = evaluate_implicit_model(model, valid_loader)
        print("\nNCF Implicit Feedback Metrics:")
        print(metrics)

    # SAMPLE RECOMMENDATION
    sample_user_id = ratings_df["user_id"].iloc[0]

    recommendations = recommend_ncf(
        model=model,
        user_id=sample_user_id,
        ratings_df=ratings_df,
        movies_df=movies_df,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        top_n=5,
        feedback_type=FEEDBACK_TYPE
    )

    print(f"\nTop NCF recommendations for user {sample_user_id}:")
    print(recommendations)