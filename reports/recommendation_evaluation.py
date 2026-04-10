# =========================================================
# IMPORT LIBRARIES
# =========================================================
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_loader import load_users, load_movies, load_ratings
from preprocessing import (
    create_user_item_matrix,
    create_normalized_user_item_matrix
)
from baseline_recommenders import (
    build_popularity_model,
    recommend_popular,
    create_user_similarity_matrix,
    recommend_user_based,
    create_item_similarity_matrix,
    recommend_item_based,
    create_svd_model,
    create_predicted_rating_matrix,
    recommend_svd
)
from content_based_nlp import (
    create_tfidf_features,
    recommend_for_existing_user,
    recommend_for_new_user_by_preference
)
from ncf_recommender import (
    encode_ids,
    train_valid_split,
    RatingsDataset,
    NeuralCollaborativeFiltering,
    train_ncf_model,
    recommend_ncf
)

# OPTIONAL: only if you saved hybrid_recommender.py
from hybrid_recommender import (
    build_popularity_lookup,
    recommend_hybrid
)


# =========================================================
# PATHS
# =========================================================
processed_data_path = Path(r"D:\DS PROJECTS SUBMISSION\Recommendation_System_Final\data\processed")
processed_data_path.mkdir(parents=True, exist_ok=True)


# =========================================================
# TIME-BASED TRAIN / TEST SPLIT
# =========================================================
def time_based_train_test_split(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2
):
    ratings_copy = ratings_df.copy()
    ratings_copy["timestamp"] = pd.to_datetime(ratings_copy["timestamp"])
    ratings_copy = ratings_copy.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(ratings_copy) * (1 - test_ratio))

    train_df = ratings_copy.iloc[:split_idx].copy()
    test_df = ratings_copy.iloc[split_idx:].copy()

    return train_df, test_df


# =========================================================
# FILTER TEST USERS TO AVOID LEAKAGE / INVALID USERS
# =========================================================
def build_eval_users(train_df: pd.DataFrame, test_df: pd.DataFrame, min_test_items: int = 1):
    train_users = set(train_df["user_id"].unique())

    test_user_counts = test_df.groupby("user_id")["movie_id"].nunique()
    valid_test_users = set(test_user_counts[test_user_counts >= min_test_items].index)

    eval_users = sorted(train_users.intersection(valid_test_users))
    return eval_users


# =========================================================
# GROUND TRUTH RELEVANT ITEMS
# =========================================================
def build_ground_truth(test_df: pd.DataFrame, threshold: int = 4):
    relevant_df = test_df.loc[test_df["rating"] >= threshold].copy()

    ground_truth = (
        relevant_df.groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    return ground_truth


# =========================================================
# METRICS
# =========================================================
def precision_at_k(recommended_items, relevant_items, k):
    recommended_items = recommended_items[:k]
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended_items if item in relevant_items)
    return hits / k


def recall_at_k(recommended_items, relevant_items, k):
    if len(relevant_items) == 0:
        return 0.0
    recommended_items = recommended_items[:k]
    hits = sum(1 for item in recommended_items if item in relevant_items)
    return hits / len(relevant_items)


def average_precision_at_k(recommended_items, relevant_items, k):
    recommended_items = recommended_items[:k]

    score = 0.0
    hits = 0

    for i, item in enumerate(recommended_items, start=1):
        if item in relevant_items:
            hits += 1
            score += hits / i

    if len(relevant_items) == 0:
        return 0.0

    return score / min(len(relevant_items), k)


def ndcg_at_k(recommended_items, relevant_items, k):
    recommended_items = recommended_items[:k]

    dcg = 0.0
    for i, item in enumerate(recommended_items, start=1):
        if item in relevant_items:
            dcg += 1 / math.log2(i + 1)

    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


# =========================================================
# EVALUATE A SINGLE MODEL
# =========================================================
def evaluate_model(eval_users, ground_truth, recommender_fn, k=10):
    precisions = []
    recalls = []
    maps = []
    ndcgs = []

    for user_id in eval_users:
        relevant_items = ground_truth.get(user_id, set())
        if len(relevant_items) == 0:
            continue

        try:
            recommended_items = recommender_fn(user_id)
        except Exception:
            continue

        if recommended_items is None or len(recommended_items) == 0:
            continue

        precisions.append(precision_at_k(recommended_items, relevant_items, k))
        recalls.append(recall_at_k(recommended_items, relevant_items, k))
        maps.append(average_precision_at_k(recommended_items, relevant_items, k))
        ndcgs.append(ndcg_at_k(recommended_items, relevant_items, k))

    return {
        "Precision@K": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        "Recall@K": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "MAP@K": round(float(np.mean(maps)) if maps else 0.0, 4),
        "NDCG@K": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
        "Evaluated_Users": len(precisions)
    }


# =========================================================
# TRAIN NCF ON TRAIN DATA ONLY
# =========================================================
def prepare_ncf_model(train_df: pd.DataFrame, feedback_type: str = "explicit"):
    encoded_train_df, user_to_index, item_to_index, index_to_user, index_to_item = encode_ids(train_df)

    train_part, valid_part = train_valid_split(
        encoded_train_df,
        valid_ratio=0.2,
        random_state=42
    )

    train_dataset = RatingsDataset(train_part, feedback_type=feedback_type)
    valid_dataset = RatingsDataset(valid_part, feedback_type=feedback_type)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=False)

    model = NeuralCollaborativeFiltering(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        embedding_dim=32,
        hidden_dims=[128, 64, 32],
        dropout=0.25
    )

    model, history = train_ncf_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        feedback_type=feedback_type,
        learning_rate=1e-3,
        weight_decay=1e-5,
        epochs=10,
        patience=3
    )

    return model, user_to_index, item_to_index


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    K = 10
    RELEVANCE_THRESHOLD = 4

    # LOAD DATA
    users_df = load_users()
    movies_df = load_movies()
    ratings_df = load_ratings()

    # TIME-BASED SPLIT
    train_df, test_df = time_based_train_test_split(ratings_df, test_ratio=0.2)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Latest train timestamp:", train_df["timestamp"].max())
    print("Earliest test timestamp:", test_df["timestamp"].min())

    # GROUND TRUTH
    eval_users = build_eval_users(train_df, test_df, min_test_items=1)
    ground_truth = build_ground_truth(test_df, threshold=RELEVANCE_THRESHOLD)

    print("Evaluation users:", len(eval_users))

    # =====================================================
    # PREPARE MATRICES USING TRAIN DATA ONLY
    # =====================================================
    train_user_item_matrix = create_user_item_matrix(train_df)
    train_normalized_user_item_matrix = create_normalized_user_item_matrix(train_df)

    # Popularity
    popularity_df = build_popularity_model(train_df)

    # User-based CF
    user_similarity_df = create_user_similarity_matrix(train_user_item_matrix)

    # Item-based CF
    item_similarity_df = create_item_similarity_matrix(train_user_item_matrix)

    # SVD
    svd_model, user_factors_df, item_factors_df = create_svd_model(
        train_normalized_user_item_matrix,
        n_components=50
    )
    predicted_ratings_df = create_predicted_rating_matrix(
        train_normalized_user_item_matrix,
        user_factors_df,
        item_factors_df
    )

    # Content-based
    movies_with_desc, tfidf, tfidf_matrix = create_tfidf_features(movies_df)

    # NCF
    ncf_model, user_to_index, item_to_index = prepare_ncf_model(
        train_df,
        feedback_type="explicit"
    )

    # Hybrid helpers
    movie_popularity_features = (
        train_df.groupby("movie_id")
        .agg(
            interaction_count=("rating", "count"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )
    max_count = movie_popularity_features["interaction_count"].max()
    movie_popularity_features["popularity_ratio"] = movie_popularity_features["interaction_count"] / max_count
    long_tail_threshold = movie_popularity_features["interaction_count"].quantile(0.80)
    movie_popularity_features["is_long_tail"] = (
        movie_popularity_features["interaction_count"] <= long_tail_threshold
    ).astype(int)

    popularity_lookup = build_popularity_lookup(movie_popularity_features)

    # =====================================================
    # RECOMMENDER WRAPPERS
    # =====================================================
    def popularity_wrapper(user_id):
        return recommend_popular(
            user_id=user_id,
            ratings_df=train_df,
            popularity_df=popularity_df,
            n_top=K
        )

    def user_cf_wrapper(user_id):
        return recommend_user_based(
            user_id=user_id,
            user_item_matrix=train_user_item_matrix,
            user_similarity_df=user_similarity_df,
            n_top=K
        )

    def item_cf_wrapper(user_id):
        return recommend_item_based(
            user_id=user_id,
            user_item_matrix=train_user_item_matrix,
            item_similarity_df=item_similarity_df,
            n_top=K
        )

    def svd_wrapper(user_id):
        return recommend_svd(
            user_id=user_id,
            user_item_matrix=train_user_item_matrix,
            predicted_ratings_df=predicted_ratings_df,
            n_top=K
        )

    def content_wrapper(user_id):
        try:
            rec_df = recommend_for_existing_user(
                user_id=user_id,
                ratings_df=train_df,
                movies_df=movies_with_desc,
                tfidf_matrix=tfidf_matrix,
                n_top=K
            )
            return rec_df["movie_id"].tolist()
        except ValueError:
            # fallback for users without liked items
            user_row = users_df.loc[users_df["user_id"] == user_id]
            if user_row.empty:
                return []
            preferred_genre = user_row.iloc[0]["preferred_category"]
            fallback_df = recommend_for_new_user_by_preference(
                preferred_genre=preferred_genre,
                movies_df=movies_with_desc,
                n_top=K
            )
            return fallback_df["movie_id"].tolist()

    def ncf_wrapper(user_id):
        if user_id not in user_to_index:
            return []
        rec_df = recommend_ncf(
            model=ncf_model,
            user_id=user_id,
            ratings_df=train_df,
            movies_df=movies_df,
            user_to_index=user_to_index,
            item_to_index=item_to_index,
            top_n=K,
            feedback_type="explicit"
        )
        return rec_df["movie_id"].tolist()

    def hybrid_wrapper(user_id):
        rec_df = recommend_hybrid(
            user_id=user_id,
            users_df=users_df,
            movies_df=movies_with_desc,
            ratings_df=train_df,
            user_item_matrix=train_user_item_matrix,
            predicted_ratings_df=predicted_ratings_df,
            tfidf_matrix=tfidf_matrix,
            ncf_model=ncf_model,
            user_to_index=user_to_index,
            item_to_index=item_to_index,
            popularity_lookup=popularity_lookup,
            n_top=K,
            feedback_type="explicit"
        )
        return rec_df["movie_id"].tolist()

    # =====================================================
    # RUN EVALUATION
    # =====================================================
    results = []

    results.append({
        "Model": "Popularity-Based",
        **evaluate_model(eval_users, ground_truth, popularity_wrapper, k=K)
    })

    results.append({
        "Model": "User-Based CF",
        **evaluate_model(eval_users, ground_truth, user_cf_wrapper, k=K)
    })

    results.append({
        "Model": "Item-Based CF",
        **evaluate_model(eval_users, ground_truth, item_cf_wrapper, k=K)
    })

    results.append({
        "Model": "SVD",
        **evaluate_model(eval_users, ground_truth, svd_wrapper, k=K)
    })

    results.append({
        "Model": "Content-Based NLP",
        **evaluate_model(eval_users, ground_truth, content_wrapper, k=K)
    })

    results.append({
        "Model": "NCF",
        **evaluate_model(eval_users, ground_truth, ncf_wrapper, k=K)
    })

    results.append({
        "Model": "Hybrid",
        **evaluate_model(eval_users, ground_truth, hybrid_wrapper, k=K)
    })

    results_df = pd.DataFrame(results)
    print("\n================ RECOMMENDATION EVALUATION RESULTS ================\n")
    print(results_df)

    output_file = processed_data_path / "recommendation_evaluation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nEvaluation results saved to: {output_file}")