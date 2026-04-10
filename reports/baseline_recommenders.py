# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# PATHS
raw_data_path = Path(r"D:\DS PROJECTS SUBMISSION\Recommendation_System_Final\data\raw")
processed_data_path = Path(r"D:\DS PROJECTS SUBMISSION\Recommendation_System_Final\data\processed")
processed_data_path.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================
def load_data():
    users_df = pd.read_csv(raw_data_path / "users.csv")
    movies_df = pd.read_csv(raw_data_path / "movies.csv")
    ratings_df = pd.read_csv(raw_data_path / "ratings.csv")
    return users_df, movies_df, ratings_df


# =========================================================
# CREATE USER-ITEM MATRIX
# =========================================================
def create_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    user_item_matrix = ratings_df.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        aggfunc="mean",
        fill_value=0
    )
    return user_item_matrix


# =========================================================
# 1. POPULARITY-BASED RECOMMENDER
# =========================================================
def build_popularity_model(ratings_df: pd.DataFrame) -> pd.DataFrame:
    popularity_df = ratings_df.groupby("movie_id").agg(
        rating_count=("rating", "count"),
        avg_rating=("rating", "mean")
    ).reset_index()

    # weighted popularity score
    popularity_df["popularity_score"] = (
        0.7 * popularity_df["rating_count"] +
        0.3 * popularity_df["avg_rating"]
    )

    popularity_df = popularity_df.sort_values(
        by="popularity_score",
        ascending=False
    ).reset_index(drop=True)

    with open(processed_data_path / "popularity_model.pkl", "wb") as f:
        pickle.dump(popularity_df, f)

    return popularity_df


def recommend_popular(
    user_id: int,
    ratings_df: pd.DataFrame,
    popularity_df: pd.DataFrame,
    n_top: int = 10
) -> list:
    already_rated = ratings_df.loc[ratings_df["user_id"] == user_id, "movie_id"].tolist()

    recommendations = popularity_df.loc[
        ~popularity_df["movie_id"].isin(already_rated),
        "movie_id"
    ].head(n_top).tolist()

    return recommendations


# =========================================================
# 2. USER-BASED COLLABORATIVE FILTERING
# =========================================================
def create_user_similarity_matrix(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    similarity = cosine_similarity(user_item_matrix)

    user_similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    with open(processed_data_path / "user_similarity.pkl", "wb") as f:
        pickle.dump(user_similarity_df, f)

    return user_similarity_df


def recommend_user_based(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    n_top: int = 10
) -> list:
    if user_id not in user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found.")

    similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False).iloc[1:]

    weighted_scores = user_item_matrix.loc[similar_users.index].T.dot(similar_users)
    similarity_sum = similar_users.sum()

    if similarity_sum != 0:
        weighted_scores = weighted_scores / similarity_sum

    already_rated = user_item_matrix.loc[user_id]
    already_rated_movies = already_rated[already_rated > 0].index

    weighted_scores = weighted_scores.drop(already_rated_movies, errors="ignore")

    recommendations = weighted_scores.sort_values(ascending=False).head(n_top).index.tolist()
    return recommendations


# =========================================================
# 3. ITEM-BASED COLLABORATIVE FILTERING
# =========================================================
def create_item_similarity_matrix(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    item_similarity = cosine_similarity(user_item_matrix.T)

    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    with open(processed_data_path / "item_similarity.pkl", "wb") as f:
        pickle.dump(item_similarity_df, f)

    return item_similarity_df


def recommend_item_based(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    item_similarity_df: pd.DataFrame,
    n_top: int = 10
) -> list:
    if user_id not in user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found.")

    user_ratings = user_item_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0]

    scores = pd.Series(0.0, index=user_item_matrix.columns)

    for movie_id, rating in rated_movies.items():
        scores = scores.add(item_similarity_df[movie_id] * rating, fill_value=0)

    scores = scores.drop(rated_movies.index, errors="ignore")
    recommendations = scores.sort_values(ascending=False).head(n_top).index.tolist()

    return recommendations


# =========================================================
# 4. MATRIX FACTORIZATION (SVD)
# =========================================================
def create_svd_model(user_item_matrix: pd.DataFrame, n_components: int = 50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_.T

    user_factors_df = pd.DataFrame(user_factors, index=user_item_matrix.index)
    item_factors_df = pd.DataFrame(item_factors, index=user_item_matrix.columns)

    with open(processed_data_path / "svd_model.pkl", "wb") as f:
        pickle.dump(svd, f)

    with open(processed_data_path / "user_factors.pkl", "wb") as f:
        pickle.dump(user_factors_df, f)

    with open(processed_data_path / "item_factors.pkl", "wb") as f:
        pickle.dump(item_factors_df, f)

    return svd, user_factors_df, item_factors_df


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

    with open(processed_data_path / "predicted_ratings.pkl", "wb") as f:
        pickle.dump(predicted_df, f)

    return predicted_df


def recommend_svd(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    predicted_ratings_df: pd.DataFrame,
    n_top: int = 10
) -> list:
    if user_id not in user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found.")

    already_rated = user_item_matrix.loc[user_id]
    already_rated_movies = already_rated[already_rated > 0].index

    predicted_scores = predicted_ratings_df.loc[user_id].drop(already_rated_movies, errors="ignore")
    recommendations = predicted_scores.sort_values(ascending=False).head(n_top).index.tolist()

    return recommendations


# =========================================================
# MOVIE TITLE HELPER
# =========================================================
def map_movie_ids_to_titles(movie_ids: list, movies_df: pd.DataFrame) -> pd.DataFrame:
    result = movies_df.loc[movies_df["movie_id"].isin(movie_ids), ["movie_id", "title", "genre", "imdb_rating"]].copy()
    result["movie_id"] = pd.Categorical(result["movie_id"], categories=movie_ids, ordered=True)
    result = result.sort_values("movie_id")
    return result


# =========================================================
# TEST BLOCK
# =========================================================
if __name__ == "__main__":
    users_df, movies_df, ratings_df = load_data()

    user_item_matrix = create_user_item_matrix(ratings_df)

    # Popularity baseline
    popularity_df = build_popularity_model(ratings_df)

    # User-based CF
    user_similarity_df = create_user_similarity_matrix(user_item_matrix)

    # Item-based CF
    item_similarity_df = create_item_similarity_matrix(user_item_matrix)

    # SVD baseline
    svd_model, user_factors_df, item_factors_df = create_svd_model(user_item_matrix, n_components=50)
    predicted_ratings_df = create_predicted_rating_matrix(
        user_item_matrix,
        user_factors_df,
        item_factors_df
    )

    sample_user_id = user_item_matrix.index[0]

    popular_recs = recommend_popular(sample_user_id, ratings_df, popularity_df, n_top=5)
    user_based_recs = recommend_user_based(sample_user_id, user_item_matrix, user_similarity_df, n_top=5)
    item_based_recs = recommend_item_based(sample_user_id, user_item_matrix, item_similarity_df, n_top=5)
    svd_recs = recommend_svd(sample_user_id, user_item_matrix, predicted_ratings_df, n_top=5)

    print(f"\nSample user_id: {sample_user_id}")

    print("\nPopularity-based recommendations:")
    print(map_movie_ids_to_titles(popular_recs, movies_df))

    print("\nUser-based CF recommendations:")
    print(map_movie_ids_to_titles(user_based_recs, movies_df))

    print("\nItem-based CF recommendations:")
    print(map_movie_ids_to_titles(item_based_recs, movies_df))

    print("\nSVD recommendations:")
    print(map_movie_ids_to_titles(svd_recs, movies_df))