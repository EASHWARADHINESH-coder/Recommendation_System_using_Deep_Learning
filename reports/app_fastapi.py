from pathlib import Path
from typing import List

import pickle
import numpy as np
import pandas as pd
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from data_loader import load_users, load_movies, load_ratings
from preprocessing import (
    create_user_item_matrix,
    create_normalized_user_item_matrix,
    create_movie_popularity_features
)
from baseline_recommenders import (
    create_svd_model,
    create_predicted_rating_matrix
)
from content_based_nlp import (
    create_tfidf_features,
    create_content_similarity_matrix,
    recommend_similar_movies,
    recommend_for_existing_user,
    recommend_for_new_user_by_preference
)
from ncf_recommender import (
    NeuralCollaborativeFiltering,
    recommend_ncf
)


# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "saved_models"

NCF_MODEL_PATH = ARTIFACTS_DIR / "ncf_model.pt"
USER_TO_INDEX_PATH = ARTIFACTS_DIR / "user_to_index.pkl"
ITEM_TO_INDEX_PATH = ARTIFACTS_DIR / "item_to_index.pkl"


# =========================================================
# GLOBAL OBJECTS
# =========================================================
users_df = None
movies_df = None
ratings_df = None

user_item_matrix = None
predicted_ratings_df = None

movies_with_desc = None
tfidf_matrix = None
content_similarity_df = None

ncf_model = None
user_to_index = None
item_to_index = None

popularity_lookup = None


# =========================================================
# RESPONSE MODELS
# =========================================================
class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    genre: str
    language: str
    imdb_rating: float
    popularity_score: float | None = None
    explanation: str


class RecommendationResponse(BaseModel):
    user_id: int
    strategy: str
    top_n: int
    recommendations: List[RecommendationItem]


class SimilarItemResponse(BaseModel):
    item_id: int
    top_n: int
    similar_items: List[RecommendationItem]


# =========================================================
# HELPERS
# =========================================================
def build_popularity_lookup(movie_popularity_features: pd.DataFrame) -> pd.DataFrame:
    df = movie_popularity_features.copy()
    df["popularity_penalty"] = 1 / (1 + np.log1p(df["interaction_count"]))
    df["long_tail_boost"] = np.where(df["is_long_tail"] == 1, 1.15, 1.00)
    return df.set_index("movie_id")


def min_max_normalize(score_series: pd.Series) -> pd.Series:
    if len(score_series) == 0:
        return score_series.astype(float)

    score_series = score_series.astype(float)
    min_val = score_series.min()
    max_val = score_series.max()

    if max_val == min_val:
        return pd.Series(0.0, index=score_series.index)

    return (score_series - min_val) / (max_val - min_val)


def load_pickle_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ncf_artifacts():
    if not NCF_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"NCF model file not found: {NCF_MODEL_PATH}\n"
            f"Train and save the model first."
        )
    if not USER_TO_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"user_to_index file not found: {USER_TO_INDEX_PATH}"
        )
    if not ITEM_TO_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"item_to_index file not found: {ITEM_TO_INDEX_PATH}"
        )

    local_user_to_index = load_pickle_file(USER_TO_INDEX_PATH)
    local_item_to_index = load_pickle_file(ITEM_TO_INDEX_PATH)

    model = torch.load(NCF_MODEL_PATH, map_location=torch.device("cpu"))

    if hasattr(model, "eval"):
        model.eval()

    return model, local_user_to_index, local_item_to_index


def get_collaborative_scores(user_id: int) -> pd.Series:
    if user_id not in user_item_matrix.index:
        return pd.Series(dtype=float)

    already_rated = user_item_matrix.loc[user_id]
    already_rated_movies = already_rated[already_rated > 0].index

    scores = predicted_ratings_df.loc[user_id].drop(already_rated_movies, errors="ignore")
    scores = min_max_normalize(scores)
    return scores.sort_values(ascending=False)


def get_content_scores(user_id: int) -> pd.Series:
    try:
        rec_df = recommend_for_existing_user(
            user_id=user_id,
            ratings_df=ratings_df,
            movies_df=movies_with_desc,
            tfidf_matrix=tfidf_matrix,
            n_top=len(movies_with_desc)
        )
    except Exception:
        return pd.Series(dtype=float)

    if rec_df.empty:
        return pd.Series(dtype=float)

    scores = pd.Series(
        data=rec_df["similarity_score"].values,
        index=rec_df["movie_id"].values
    )
    scores = min_max_normalize(scores)
    return scores.sort_values(ascending=False)


def get_ncf_scores(user_id: int) -> pd.Series:
    if ncf_model is None or user_to_index is None or item_to_index is None:
        return pd.Series(dtype=float)

    if user_id not in user_to_index:
        return pd.Series(dtype=float)

    rec_df = recommend_ncf(
        model=ncf_model,
        user_id=user_id,
        ratings_df=ratings_df,
        movies_df=movies_df,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        top_n=len(movies_df),
        feedback_type="explicit"
    )

    if rec_df.empty:
        return pd.Series(dtype=float)

    scores = pd.Series(
        data=rec_df["predicted_score"].values,
        index=rec_df["movie_id"].values
    )
    scores = min_max_normalize(scores)
    return scores.sort_values(ascending=False)


def apply_popularity_reranking(score_series: pd.Series) -> pd.DataFrame:
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

    return recommendation_df.sort_values("adjusted_score", ascending=False)


def fuse_scores(
    collaborative_scores: pd.Series,
    content_scores: pd.Series,
    ncf_scores: pd.Series
) -> pd.DataFrame:
    all_movie_ids = set(collaborative_scores.index) | set(content_scores.index) | set(ncf_scores.index)

    fused = pd.DataFrame(index=list(all_movie_ids))
    fused["collaborative_score"] = collaborative_scores.reindex(fused.index).fillna(0.0)
    fused["content_score"] = content_scores.reindex(fused.index).fillna(0.0)
    fused["ncf_score"] = ncf_scores.reindex(fused.index).fillna(0.0)

    fused["hybrid_raw_score"] = (
        0.35 * fused["collaborative_score"] +
        0.25 * fused["content_score"] +
        0.40 * fused["ncf_score"]
    )

    reranked = apply_popularity_reranking(fused["hybrid_raw_score"])
    fused = fused.join(reranked[["adjusted_score"]], how="left")
    return fused.sort_values("adjusted_score", ascending=False)


def explain_recommendation(user_id: int, movie_id: int, fused_row: pd.Series) -> str:
    user_row = users_df.loc[users_df["user_id"] == user_id]
    movie_row = movies_df.loc[movies_df["movie_id"] == movie_id]

    if user_row.empty or movie_row.empty:
        return "Recommended based on hybrid recommendation signals."

    preferred_category = user_row.iloc[0]["preferred_category"]
    genre = movie_row.iloc[0]["genre"]

    reasons = []

    if genre == preferred_category:
        reasons.append(f"matches your preferred category '{preferred_category}'")

    if fused_row.get("content_score", 0) > 0.5:
        reasons.append("has high content similarity to movies you liked")

    if fused_row.get("collaborative_score", 0) > 0.5:
        reasons.append("is supported by collaborative filtering patterns")

    if fused_row.get("ncf_score", 0) > 0.5:
        reasons.append("received a strong deep learning recommendation score")

    if movie_id in popularity_lookup.index:
        if popularity_lookup.loc[movie_id, "is_long_tail"] == 1:
            reasons.append("was boosted as a promising long-tail item")
        else:
            reasons.append("has strong overall popularity")

    if not reasons:
        return "Recommended using a fusion of collaborative, content-based, and deep learning signals."

    return "Recommended because it " + ", ".join(reasons) + "."


def cold_start_recommendations(user_id: int, top_n: int) -> pd.DataFrame:
    user_row = users_df.loc[users_df["user_id"] == user_id]

    if user_row.empty:
        fallback = movies_df[
            ["movie_id", "title", "genre", "language", "imdb_rating", "popularity_score"]
        ].copy()

        fallback = fallback.sort_values(
            by=["imdb_rating", "popularity_score"],
            ascending=False
        ).head(top_n)

        fallback["explanation"] = "Recommended using global popularity fallback for a new user."
        return fallback

    preferred_genre = user_row.iloc[0]["preferred_category"]

    fallback = recommend_for_new_user_by_preference(
        preferred_genre=preferred_genre,
        movies_df=movies_with_desc,
        n_top=top_n
    ).copy()

    fallback["explanation"] = (
        f"Recommended using cold-start fallback based on your preferred category '{preferred_genre}'."
    )
    return fallback


def validate_service_ready():
    if users_df is None or movies_df is None or ratings_df is None:
        raise HTTPException(status_code=500, detail="Service not initialized.")


# =========================================================
# LIFESPAN
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global users_df, movies_df, ratings_df
    global user_item_matrix, predicted_ratings_df
    global movies_with_desc, tfidf_matrix, content_similarity_df
    global ncf_model, user_to_index, item_to_index
    global popularity_lookup

    users_df = load_users()
    movies_df = load_movies()
    ratings_df = load_ratings()

    user_item_matrix_local = create_user_item_matrix(ratings_df)
    normalized_user_item_matrix = create_normalized_user_item_matrix(ratings_df)

    _, user_factors_df, item_factors_df = create_svd_model(
        normalized_user_item_matrix,
        n_components=50
    )

    predicted_ratings_df_local = create_predicted_rating_matrix(
        normalized_user_item_matrix,
        user_factors_df,
        item_factors_df
    )

    movie_popularity_features = create_movie_popularity_features(ratings_df)
    popularity_lookup_local = build_popularity_lookup(movie_popularity_features)

    movies_with_desc_local, _, tfidf_matrix_local = create_tfidf_features(movies_df)
    content_similarity_df_local = create_content_similarity_matrix(
        movies_with_desc_local,
        tfidf_matrix_local
    )

    # Load pre-trained NCF artifacts instead of training on startup
    try:
        ncf_model_local, user_to_index_local, item_to_index_local = load_ncf_artifacts()
    except Exception:
        ncf_model_local, user_to_index_local, item_to_index_local = None, {}, {}

    user_item_matrix = user_item_matrix_local
    predicted_ratings_df = predicted_ratings_df_local
    popularity_lookup = popularity_lookup_local
    movies_with_desc = movies_with_desc_local
    tfidf_matrix = tfidf_matrix_local
    content_similarity_df = content_similarity_df_local
    ncf_model = ncf_model_local
    user_to_index = user_to_index_local
    item_to_index = item_to_index_local

    yield


# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="Movie Recommendation Service",
    description="FastAPI Recommendation API with Hybrid + Similar Items endpoints",
    version="2.0.0",
    lifespan=lifespan
)


# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():
    return {
        "message": "Movie Recommendation FastAPI Service is running",
        "docs": "/docs",
        "endpoints": [
            "/recommend/{user_id}",
            "/similar-items/{item_id}"
        ]
    }


# =========================================================
# RECOMMEND ENDPOINT
# =========================================================
@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int, top_n: int = Query(10, ge=1, le=50)):
    validate_service_ready()

    user_known = user_id in set(users_df["user_id"].unique())
    user_has_history = (
        user_id in user_item_matrix.index
        and not ratings_df.loc[ratings_df["user_id"] == user_id].empty
    )

    if not user_has_history:
        fallback_df = cold_start_recommendations(user_id, top_n=top_n)

        recommendations = [
            RecommendationItem(
                movie_id=int(row["movie_id"]),
                title=str(row["title"]),
                genre=str(row["genre"]),
                language=str(row["language"]),
                imdb_rating=float(row["imdb_rating"]),
                popularity_score=float(row["popularity_score"]) if pd.notna(row["popularity_score"]) else None,
                explanation=str(row["explanation"])
            )
            for _, row in fallback_df.iterrows()
        ]

        strategy = "cold_start_fallback" if user_known else "global_fallback"

        return RecommendationResponse(
            user_id=user_id,
            strategy=strategy,
            top_n=top_n,
            recommendations=recommendations
        )

    collaborative_scores = get_collaborative_scores(user_id)
    content_scores = get_content_scores(user_id)
    ncf_scores = get_ncf_scores(user_id)

    fused_scores = fuse_scores(
        collaborative_scores=collaborative_scores,
        content_scores=content_scores,
        ncf_scores=ncf_scores
    )

    top_df = fused_scores.head(top_n).reset_index().rename(columns={"index": "movie_id"})
    top_df = top_df.merge(
        movies_df[["movie_id", "title", "genre", "language", "imdb_rating", "popularity_score"]],
        on="movie_id",
        how="left"
    )

    recommendations = []
    for _, row in top_df.iterrows():
        explanation = explain_recommendation(
            user_id=user_id,
            movie_id=int(row["movie_id"]),
            fused_row=row
        )

        recommendations.append(
            RecommendationItem(
                movie_id=int(row["movie_id"]),
                title=str(row["title"]),
                genre=str(row["genre"]),
                language=str(row["language"]),
                imdb_rating=float(row["imdb_rating"]),
                popularity_score=float(row["popularity_score"]) if pd.notna(row["popularity_score"]) else None,
                explanation=explanation
            )
        )

    return RecommendationResponse(
        user_id=user_id,
        strategy="hybrid_fusion",
        top_n=top_n,
        recommendations=recommendations
    )


# =========================================================
# SIMILAR ITEMS ENDPOINT
# =========================================================
@app.get("/similar-items/{item_id}", response_model=SimilarItemResponse)
def similar_items(item_id: int, top_n: int = Query(10, ge=1, le=50)):
    validate_service_ready()

    if item_id not in set(movies_df["movie_id"].unique()):
        raise HTTPException(status_code=404, detail=f"movie_id {item_id} not found.")

    rec_df = recommend_similar_movies(
        movie_id=item_id,
        movies_df=movies_with_desc,
        similarity_df=content_similarity_df,
        n_top=top_n
    )

    similar_items_list = []

    base_movie = movies_df.loc[movies_df["movie_id"] == item_id].iloc[0]

    for _, row in rec_df.iterrows():
        explanation = (
            f"Similar to '{base_movie['title']}' based on TF-IDF description similarity, "
            f"shared genre/language/production metadata, and cosine similarity score."
        )

        similar_items_list.append(
            RecommendationItem(
                movie_id=int(row["movie_id"]),
                title=str(row["title"]),
                genre=str(row["genre"]),
                language=str(row["language"]),
                imdb_rating=float(row["imdb_rating"]),
                popularity_score=float(row["popularity_score"]) if "popularity_score" in row and pd.notna(row["popularity_score"]) else None,
                explanation=explanation
            )
        )

    return SimilarItemResponse(
        item_id=item_id,
        top_n=top_n,
        similar_items=similar_items_list
    )