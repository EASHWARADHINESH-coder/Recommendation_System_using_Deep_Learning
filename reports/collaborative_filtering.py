# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# PROCESSED DATA PATH
data_path = Path(r"D:\DS PROJECTS SUBMISSION\Recommendation_System_Final\data\processed")
data_path.mkdir(parents=True, exist_ok=True)


# TRAIN SVD MODEL
def create_svd_model(user_item_matrix: pd.DataFrame, n_components: int = 50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_.T

    user_factors_df = pd.DataFrame(user_factors, index=user_item_matrix.index)
    item_factors_df = pd.DataFrame(item_factors, index=user_item_matrix.columns)

    with open(data_path / "svd_model.pkl", "wb") as f:
        pickle.dump(svd, f)

    with open(data_path / "user_factors.pkl", "wb") as f:
        pickle.dump(user_factors_df, f)

    with open(data_path / "item_factors.pkl", "wb") as f:
        pickle.dump(item_factors_df, f)

    return svd, user_factors_df, item_factors_df


# RECONSTRUCT PREDICTION MATRIX
def create_predicted_rating_matrix(
    user_item_matrix: pd.DataFrame,
    user_factors_df: pd.DataFrame,
    item_factors_df: pd.DataFrame
) -> pd.DataFrame:
    predicted_matrix = np.dot(user_factors_df.values, item_factors_df.values.T)

    predicted_df = pd.DataFrame(
        predicted_matrix,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )

    with open(data_path / "predicted_ratings.pkl", "wb") as f:
        pickle.dump(predicted_df, f)

    return predicted_df


# IMPLICIT ITEM-ITEM SIMILARITY
def create_item_similarity_from_implicit(implicit_feedback_matrix: pd.DataFrame) -> pd.DataFrame:
    item_similarity = cosine_similarity(implicit_feedback_matrix.T)

    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=implicit_feedback_matrix.columns,
        columns=implicit_feedback_matrix.columns
    )

    with open(data_path / "implicit_item_similarity.pkl", "wb") as f:
        pickle.dump(item_similarity_df, f)

    return item_similarity_df


# BUILD POPULARITY LOOKUP
def build_popularity_lookup(movie_popularity_features: pd.DataFrame) -> pd.DataFrame:
    df = movie_popularity_features.copy()

    df["popularity_penalty"] = 1 / (1 + np.log1p(df["interaction_count"]))
    df["long_tail_boost"] = np.where(df["is_long_tail"] == 1, 1.20, 1.00)

    return df.set_index("movie_id")


# APPLY POPULARITY BALANCE
def rerank_with_popularity_balance(
    score_series: pd.Series,
    popularity_lookup: pd.DataFrame
) -> pd.DataFrame:
    recommendation_df = pd.DataFrame({"raw_score": score_series})

    recommendation_df = recommendation_df.join(
        popularity_lookup[[
            "interaction_count",
            "average_rating",
            "popularity_ratio",
            "is_long_tail",
            "popularity_penalty",
            "long_tail_boost"
        ]],
        how="left"
    )

    recommendation_df["interaction_count"] = recommendation_df["interaction_count"].fillna(0)
    recommendation_df["average_rating"] = recommendation_df["average_rating"].fillna(0)
    recommendation_df["popularity_ratio"] = recommendation_df["popularity_ratio"].fillna(0)
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


# DETECT COLD-START ITEMS
def get_cold_start_items(
    movie_content_features: pd.DataFrame,
    popularity_lookup: pd.DataFrame,
    min_interactions: int = 5
) -> list:
    interaction_counts = popularity_lookup["interaction_count"] if "interaction_count" in popularity_lookup.columns else pd.Series(dtype=float)

    cold_items = []
    for movie_id in movie_content_features.index:
        count = interaction_counts.get(movie_id, 0)
        if count < min_interactions:
            cold_items.append(movie_id)

    return cold_items


# CONTENT SCORE FOR ITEMS USING USER PREFERRED CATEGORY
def score_items_for_new_user(
    user_id: int,
    user_profile_features: pd.DataFrame,
    movie_content_features: pd.DataFrame,
    popularity_lookup: pd.DataFrame
) -> pd.Series:
    if user_id not in user_profile_features.index:
        raise ValueError(f"user_id {user_id} not found in user profile features.")

    user_profile = user_profile_features.loc[user_id]

    genre_cols_user = [col for col in user_profile_features.columns if col.startswith("preferred_category_")]
    genre_cols_movie = [col for col in movie_content_features.columns if col.startswith("genre_")]

    scores = pd.Series(0.0, index=movie_content_features.index)

    # MATCH USER PREFERRED CATEGORY WITH MOVIE GENRE
    for user_col in genre_cols_user:
        if user_col.replace("preferred_category_", "genre_") in genre_cols_movie:
            movie_col = user_col.replace("preferred_category_", "genre_")
            scores += user_profile[user_col] * movie_content_features[movie_col]

    # ADD QUALITY / POPULARITY / RECENCY SIGNALS
    for numeric_col, weight in {
        "imdb_rating": 0.35,
        "popularity_score": 0.20,
        "release_year": 0.15
    }.items():
        if numeric_col in movie_content_features.columns:
            scores += movie_content_features[numeric_col] * weight

    reranked_df = rerank_with_popularity_balance(scores, popularity_lookup)
    return reranked_df["adjusted_score"]


# CONTENT-BASED RECOMMENDATION FOR EXISTING USER
def recommend_content_based_for_existing_user(
    user_id: int,
    original_user_item_matrix: pd.DataFrame,
    movie_content_features: pd.DataFrame,
    popularity_lookup: pd.DataFrame,
    n_top: int = 10
) -> list:
    if user_id not in original_user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found in user-item matrix.")

    user_ratings = original_user_item_matrix.loc[user_id]
    liked_movies = user_ratings[user_ratings >= 4].index.tolist()

    if len(liked_movies) == 0:
        cold_scores = score_items_for_new_user(
            user_id=user_id,
            user_profile_features=load_pickle("user_profile_features.pkl"),
            movie_content_features=movie_content_features,
            popularity_lookup=popularity_lookup
        )
        return cold_scores.head(n_top).index.tolist()

    liked_movie_features = movie_content_features.loc[
        movie_content_features.index.intersection(liked_movies)
    ]

    user_content_profile = liked_movie_features.mean(axis=0)

    raw_scores = movie_content_features.dot(user_content_profile)

    already_rated_movies = user_ratings[user_ratings > 0].index
    raw_scores = raw_scores.drop(already_rated_movies, errors="ignore")

    reranked_df = rerank_with_popularity_balance(raw_scores, popularity_lookup)
    return reranked_df.head(n_top).index.tolist()


# SVD RECOMMENDATION
def recommend_product_svd(
    user_id: int,
    original_user_item_matrix: pd.DataFrame,
    predicted_ratings_df: pd.DataFrame,
    popularity_lookup: pd.DataFrame,
    n_top: int = 10
) -> pd.Series:
    if user_id not in original_user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found in user-item matrix.")

    user_actual_ratings = original_user_item_matrix.loc[user_id]
    already_rated_movies = user_actual_ratings[user_actual_ratings > 0].index

    user_predictions = predicted_ratings_df.loc[user_id].drop(already_rated_movies, errors="ignore")
    reranked_df = rerank_with_popularity_balance(user_predictions, popularity_lookup)

    return reranked_df["adjusted_score"].head(n_top * 5)


# IMPLICIT RECOMMENDATION
def recommend_product_implicit(
    user_id: int,
    implicit_feedback_matrix: pd.DataFrame,
    item_similarity_df: pd.DataFrame,
    popularity_lookup: pd.DataFrame,
    n_top: int = 10
) -> pd.Series:
    if user_id not in implicit_feedback_matrix.index:
        raise ValueError(f"user_id {user_id} not found in implicit feedback matrix.")

    user_interactions = implicit_feedback_matrix.loc[user_id]
    interacted_movies = user_interactions[user_interactions > 0].index.tolist()

    scores = pd.Series(0.0, index=implicit_feedback_matrix.columns)

    for movie_id in interacted_movies:
        scores = scores.add(item_similarity_df[movie_id], fill_value=0)

    scores = scores.drop(interacted_movies, errors="ignore")
    reranked_df = rerank_with_popularity_balance(scores, popularity_lookup)

    return reranked_df["adjusted_score"].head(n_top * 5)


# HYBRID RECOMMENDATION
def recommend_hybrid(
    user_id: int,
    original_user_item_matrix: pd.DataFrame,
    implicit_feedback_matrix: pd.DataFrame,
    predicted_ratings_df: pd.DataFrame,
    item_similarity_df: pd.DataFrame,
    user_profile_features: pd.DataFrame,
    movie_content_features: pd.DataFrame,
    popularity_lookup: pd.DataFrame,
    n_top: int = 10
) -> list:
    # FULL COLD-START USER
    if user_id not in original_user_item_matrix.index:
        cold_scores = score_items_for_new_user(
            user_id=user_id,
            user_profile_features=user_profile_features,
            movie_content_features=movie_content_features,
            popularity_lookup=popularity_lookup
        )
        return cold_scores.head(n_top).index.tolist()

    svd_scores = recommend_product_svd(
        user_id=user_id,
        original_user_item_matrix=original_user_item_matrix,
        predicted_ratings_df=predicted_ratings_df,
        popularity_lookup=popularity_lookup,
        n_top=n_top
    )

    implicit_scores = recommend_product_implicit(
        user_id=user_id,
        implicit_feedback_matrix=implicit_feedback_matrix,
        item_similarity_df=item_similarity_df,
        popularity_lookup=popularity_lookup,
        n_top=n_top
    )

    content_recommendations = recommend_content_based_for_existing_user(
        user_id=user_id,
        original_user_item_matrix=original_user_item_matrix,
        movie_content_features=movie_content_features,
        popularity_lookup=popularity_lookup,
        n_top=n_top * 5
    )

    content_scores = pd.Series(
        np.linspace(1.0, 0.1, num=len(content_recommendations)),
        index=content_recommendations
    )

    all_items = set(svd_scores.index) | set(implicit_scores.index) | set(content_scores.index)

    hybrid_scores = {}
    for item in all_items:
        hybrid_scores[item] = (
            0.50 * svd_scores.get(item, 0) +
            0.25 * implicit_scores.get(item, 0) +
            0.25 * content_scores.get(item, 0)
        )

    hybrid_scores = pd.Series(hybrid_scores).sort_values(ascending=False)

    return hybrid_scores.head(n_top).index.tolist()


# HELPER TO LOAD PICKLE
def load_pickle(filename: str):
    with open(data_path / filename, "rb") as f:
        return pickle.load(f)


# TEST BLOCK
if __name__ == "__main__":
    user_item_matrix = load_pickle("user_item_matrix.pkl")
    normalized_user_item_matrix = load_pickle("normalized_user_item_matrix.pkl")
    implicit_feedback_matrix = load_pickle("implicit_feedback_matrix.pkl")
    movie_popularity_features = load_pickle("movie_popularity_features.pkl")
    movie_content_features = load_pickle("movie_content_features.pkl")
    user_profile_features = load_pickle("user_profile_features.pkl")

    svd_model, user_factors_df, item_factors_df = create_svd_model(
        normalized_user_item_matrix,
        n_components=50
    )

    predicted_ratings_df = create_predicted_rating_matrix(
        normalized_user_item_matrix,
        user_factors_df,
        item_factors_df
    )

    item_similarity_df = create_item_similarity_from_implicit(implicit_feedback_matrix)
    popularity_lookup = build_popularity_lookup(movie_popularity_features)

    sample_user_id = user_item_matrix.index[0]

    hybrid_recommendations = recommend_hybrid(
        user_id=sample_user_id,
        original_user_item_matrix=user_item_matrix,
        implicit_feedback_matrix=implicit_feedback_matrix,
        predicted_ratings_df=predicted_ratings_df,
        item_similarity_df=item_similarity_df,
        user_profile_features=user_profile_features,
        movie_content_features=movie_content_features,
        popularity_lookup=popularity_lookup,
        n_top=5
    )

    print(f"Top hybrid recommendations for existing user {sample_user_id}:")
    print(hybrid_recommendations)

    # EXAMPLE NEW USER
    new_user_id = user_profile_features.index[-1]

    cold_user_recommendations = recommend_hybrid(
        user_id=new_user_id,
        original_user_item_matrix=user_item_matrix.iloc[:-1],
        implicit_feedback_matrix=implicit_feedback_matrix.iloc[:-1],
        predicted_ratings_df=predicted_ratings_df.iloc[:-1],
        item_similarity_df=item_similarity_df,
        user_profile_features=user_profile_features,
        movie_content_features=movie_content_features,
        popularity_lookup=popularity_lookup,
        n_top=5
    )

    print(f"\nTop hybrid recommendations for cold-start user {new_user_id}:")
    print(cold_user_recommendations)

    cold_items = get_cold_start_items(
        movie_content_features=movie_content_features,
        popularity_lookup=popularity_lookup,
        min_interactions=5
    )

    print(f"\nNumber of cold-start items: {len(cold_items)}")