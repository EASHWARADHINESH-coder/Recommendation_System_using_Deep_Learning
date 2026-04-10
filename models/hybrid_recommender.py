# =========================================================
# IMPORT LIBRARIES
# =========================================================
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import load_users, load_movies, load_ratings
from preprocessing import (
    create_user_item_matrix,
    create_normalized_user_item_matrix,
    create_implicit_feedback_matrix,
    create_movie_popularity_features,
    create_user_profile_features,
    save_pickle
)
from baseline_recommenders import (
    create_svd_model,
    create_predicted_rating_matrix
)
from content_based_nlp import (
    create_tfidf_features,
    create_content_similarity_matrix,
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
from torch.utils.data import DataLoader


# =========================================================
# PATHS
# =========================================================
processed_data_path = Path(r"D:\DS PROJECTS SUBMISSION\Recommendation_System_Final\data\processed")
processed_data_path.mkdir(parents=True, exist_ok=True)


# =========================================================
# SAVE / LOAD HELPERS
# =========================================================
def save_pickle_local(obj, filename: str):
    with open(processed_data_path / filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle_local(filename: str):
    with open(processed_data_path / filename, "rb") as f:
        return pickle.load(f)


# =========================================================
# POPULARITY LOOKUP
# =========================================================
def build_popularity_lookup(movie_popularity_features: pd.DataFrame) -> pd.DataFrame:
    df = movie_popularity_features.copy()

    df["popularity_penalty"] = 1 / (1 + np.log1p(df["interaction_count"]))
    df["long_tail_boost"] = np.where(df["is_long_tail"] == 1, 1.15, 1.00)

    return df.set_index("movie_id")


# =========================================================
# NORMALIZE SCORES TO 0-1
# =========================================================
def min_max_normalize(score_series: pd.Series) -> pd.Series:
    score_series = score_series.astype(float)

    if len(score_series) == 0:
        return score_series

    min_val = score_series.min()
    max_val = score_series.max()

    if max_val == min_val:
        return pd.Series(0.0, index=score_series.index)

    return (score_series - min_val) / (max_val - min_val)


# =========================================================
# COLLABORATIVE SCORE (SVD)
# =========================================================
def get_collaborative_scores(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    predicted_ratings_df: pd.DataFrame
) -> pd.Series:
    if user_id not in user_item_matrix.index:
        return pd.Series(dtype=float)

    already_rated = user_item_matrix.loc[user_id]
    already_rated_movies = already_rated[already_rated > 0].index

    scores = predicted_ratings_df.loc[user_id].drop(already_rated_movies, errors="ignore")
    scores = min_max_normalize(scores)

    return scores.sort_values(ascending=False)


# =========================================================
# CONTENT-BASED SCORE (TF-IDF USER PROFILE)
# =========================================================
def get_content_scores(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    tfidf_matrix
) -> pd.Series:
    try:
        content_recs = recommend_for_existing_user(
            user_id=user_id,
            ratings_df=ratings_df,
            movies_df=movies_df,
            tfidf_matrix=tfidf_matrix,
            n_top=len(movies_df)
        )
    except ValueError:
        return pd.Series(dtype=float)

    if content_recs.empty:
        return pd.Series(dtype=float)

    scores = pd.Series(
        data=content_recs["similarity_score"].values,
        index=content_recs["movie_id"].values
    )

    scores = min_max_normalize(scores)
    return scores.sort_values(ascending=False)


# =========================================================
# NCF SCORE
# =========================================================
def get_ncf_scores(
    user_id: int,
    model,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_to_index: dict,
    item_to_index: dict,
    feedback_type: str = "explicit"
) -> pd.Series:
    if user_id not in user_to_index:
        return pd.Series(dtype=float)

    ncf_df = recommend_ncf(
        model=model,
        user_id=user_id,
        ratings_df=ratings_df,
        movies_df=movies_df,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        top_n=len(movies_df),
        feedback_type=feedback_type
    )

    if ncf_df.empty:
        return pd.Series(dtype=float)

    scores = pd.Series(
        data=ncf_df["predicted_score"].values,
        index=ncf_df["movie_id"].values
    )

    scores = min_max_normalize(scores)
    return scores.sort_values(ascending=False)


# =========================================================
# APPLY POPULARITY RE-RANKING
# =========================================================
def apply_popularity_reranking(
    score_series: pd.Series,
    popularity_lookup: pd.DataFrame
) -> pd.DataFrame:
    recommendation_df = pd.DataFrame({"raw_score": score_series})

    recommendation_df = recommendation_df.join(
        popularity_lookup[
            ["interaction_count", "average_rating", "is_long_tail", "popularity_penalty", "long_tail_boost"]
        ],
        how="left"
    )

    recommendation_df["interaction_count"] = recommendation_df["interaction_count"].fillna(0)
    recommendation_df["average_rating"] = recommendation_df["average_rating"].fillna(0)
    recommendation_df["is_long_tail"] = recommendation_df["is_long_tail"].fillna(1)
    recommendation_df["popularity_penalty"] = recommendation_df["popularity_penalty"].fillna(1.0)
    recommendation_df["long_tail_boost"] = recommendation_df["long_tail_boost"].fillna(1.0)

    recommendation_df["adjusted_score"] = (
        recommendation_df["raw_score"]
        * recommendation_df["popularity_penalty"]
        * recommendation_df["long_tail_boost"]
    )

    recommendation_df = recommendation_df.sort_values("adjusted_score", ascending=False)
    return recommendation_df


# =========================================================
# SCORE FUSION STRATEGY
# =========================================================
def fuse_scores(
    collaborative_scores: pd.Series,
    content_scores: pd.Series,
    ncf_scores: pd.Series,
    popularity_lookup: pd.DataFrame,
    weights: dict = None
) -> pd.DataFrame:
    if weights is None:
        weights = {
            "collaborative": 0.35,
            "content": 0.25,
            "ncf": 0.40
        }

    all_movie_ids = set(collaborative_scores.index) | set(content_scores.index) | set(ncf_scores.index)

    fused = pd.DataFrame(index=list(all_movie_ids))
    fused["collaborative_score"] = collaborative_scores.reindex(fused.index).fillna(0.0)
    fused["content_score"] = content_scores.reindex(fused.index).fillna(0.0)
    fused["ncf_score"] = ncf_scores.reindex(fused.index).fillna(0.0)

    fused["hybrid_raw_score"] = (
        weights["collaborative"] * fused["collaborative_score"] +
        weights["content"] * fused["content_score"] +
        weights["ncf"] * fused["ncf_score"]
    )

    reranked_df = apply_popularity_reranking(
        fused["hybrid_raw_score"],
        popularity_lookup
    )

    fused = fused.join(reranked_df[["adjusted_score"]], how="left")
    fused = fused.sort_values("adjusted_score", ascending=False)

    return fused


# =========================================================
# COLD-START FALLBACK LOGIC
# =========================================================
def recommend_cold_start_user(
    user_id: int,
    users_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    popularity_lookup: pd.DataFrame,
    n_top: int = 10
) -> pd.DataFrame:
    user_row = users_df.loc[users_df["user_id"] == user_id]

    if user_row.empty:
        fallback = movies_df[["movie_id", "title", "genre", "language", "imdb_rating", "popularity_score"]].copy()
        fallback = fallback.sort_values(["imdb_rating", "popularity_score"], ascending=False).head(n_top)
        fallback["reason"] = "global_popularity_fallback"
        return fallback

    preferred_genre = user_row.iloc[0]["preferred_category"]

    cold_recs = recommend_for_new_user_by_preference(
        preferred_genre=preferred_genre,
        movies_df=movies_df,
        n_top=n_top
    ).copy()

    cold_recs["reason"] = "preferred_genre_fallback"
    return cold_recs


# =========================================================
# HYBRID RECOMMENDATION
# =========================================================
def recommend_hybrid(
    user_id: int,
    users_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    predicted_ratings_df: pd.DataFrame,
    tfidf_matrix,
    ncf_model,
    user_to_index: dict,
    item_to_index: dict,
    popularity_lookup: pd.DataFrame,
    n_top: int = 10,
    feedback_type: str = "explicit"
) -> pd.DataFrame:
    # ---------- Cold-start user ----------
    if user_id not in user_item_matrix.index or ratings_df.loc[ratings_df["user_id"] == user_id].empty:
        return recommend_cold_start_user(
            user_id=user_id,
            users_df=users_df,
            movies_df=movies_df,
            popularity_lookup=popularity_lookup,
            n_top=n_top
        )

    collaborative_scores = get_collaborative_scores(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        predicted_ratings_df=predicted_ratings_df
    )

    content_scores = get_content_scores(
        user_id=user_id,
        ratings_df=ratings_df,
        movies_df=movies_df,
        tfidf_matrix=tfidf_matrix
    )

    ncf_scores = get_ncf_scores(
        user_id=user_id,
        model=ncf_model,
        ratings_df=ratings_df,
        movies_df=movies_df,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        feedback_type=feedback_type
    )

    fused_scores = fuse_scores(
        collaborative_scores=collaborative_scores,
        content_scores=content_scores,
        ncf_scores=ncf_scores,
        popularity_lookup=popularity_lookup
    )

    result = fused_scores.head(n_top).reset_index().rename(columns={"index": "movie_id"})

    result = result.merge(
        movies_df[["movie_id", "title", "genre", "language", "imdb_rating", "popularity_score"]],
        on="movie_id",
        how="left"
    )

    return result[[
        "movie_id",
        "title",
        "genre",
        "language",
        "imdb_rating",
        "popularity_score",
        "collaborative_score",
        "content_score",
        "ncf_score",
        "adjusted_score"
    ]]


# =========================================================
# PREPARE NCF MODEL
# =========================================================
def train_ncf_for_hybrid(ratings_df: pd.DataFrame, feedback_type: str = "explicit"):
    encoded_ratings_df, user_to_index, item_to_index, index_to_user, index_to_item = encode_ids(ratings_df)

    train_df, valid_df = train_valid_split(
        encoded_ratings_df,
        valid_ratio=0.2,
        random_state=42
    )

    train_dataset = RatingsDataset(train_df, feedback_type=feedback_type)
    valid_dataset = RatingsDataset(valid_df, feedback_type=feedback_type)

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

    torch.save(model.state_dict(), processed_data_path / "hybrid_ncf_model.pt")
    save_pickle_local(user_to_index, "hybrid_user_to_index.pkl")
    save_pickle_local(item_to_index, "hybrid_item_to_index.pkl")

    return model, user_to_index, item_to_index


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    # LOAD DATA
    users_df = load_users()
    movies_df = load_movies()
    ratings_df = load_ratings()

    # PREPROCESSING
    user_item_matrix = create_user_item_matrix(ratings_df)
    normalized_user_item_matrix = create_normalized_user_item_matrix(ratings_df)
    implicit_feedback_matrix = create_implicit_feedback_matrix(ratings_df, threshold=4)
    movie_popularity_features = create_movie_popularity_features(ratings_df)
    user_profile_features = create_user_profile_features(users_df)

    save_pickle(user_item_matrix, "user_item_matrix.pkl")
    save_pickle(normalized_user_item_matrix, "normalized_user_item_matrix.pkl")
    save_pickle(implicit_feedback_matrix, "implicit_feedback_matrix.pkl")
    save_pickle(movie_popularity_features, "movie_popularity_features.pkl")
    save_pickle(user_profile_features, "user_profile_features.pkl")

    popularity_lookup = build_popularity_lookup(movie_popularity_features)

    # COLLABORATIVE FILTERING (SVD)
    svd_model, user_factors_df, item_factors_df = create_svd_model(
        normalized_user_item_matrix,
        n_components=50
    )
    predicted_ratings_df = create_predicted_rating_matrix(
        normalized_user_item_matrix,
        user_factors_df,
        item_factors_df
    )

    # CONTENT-BASED NLP
    movies_with_desc, tfidf, tfidf_matrix = create_tfidf_features(movies_df)
    content_similarity_df = create_content_similarity_matrix(movies_with_desc, tfidf_matrix)

    save_pickle_local(content_similarity_df, "content_similarity.pkl")

    # DEEP LEARNING (NCF)
    ncf_model, user_to_index, item_to_index = train_ncf_for_hybrid(
        ratings_df=ratings_df,
        feedback_type="explicit"
    )

    # SAMPLE EXISTING USER
    sample_user_id = ratings_df["user_id"].iloc[0]

    hybrid_recommendations = recommend_hybrid(
        user_id=sample_user_id,
        users_df=users_df,
        movies_df=movies_with_desc,
        ratings_df=ratings_df,
        user_item_matrix=user_item_matrix,
        predicted_ratings_df=predicted_ratings_df,
        tfidf_matrix=tfidf_matrix,
        ncf_model=ncf_model,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        popularity_lookup=popularity_lookup,
        n_top=10,
        feedback_type="explicit"
    )

    print(f"\nTop Hybrid Recommendations for existing user {sample_user_id}:")
    print(hybrid_recommendations)

    # SAMPLE COLD-START USER
    new_user_id = 999999

    cold_start_recommendations = recommend_hybrid(
        user_id=new_user_id,
        users_df=users_df,
        movies_df=movies_with_desc,
        ratings_df=ratings_df,
        user_item_matrix=user_item_matrix,
        predicted_ratings_df=predicted_ratings_df,
        tfidf_matrix=tfidf_matrix,
        ncf_model=ncf_model,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        popularity_lookup=popularity_lookup,
        n_top=10,
        feedback_type="explicit"
    )

    print(f"\nTop Hybrid Recommendations for cold-start user {new_user_id}:")
    print(cold_start_recommendations)
