# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
# BUILD SYNTHETIC ITEM DESCRIPTION
# =========================================================
def build_movie_descriptions(movies_df: pd.DataFrame) -> pd.DataFrame:
    movies_copy = movies_df.copy()

    # If description already exists, keep it
    if "description" not in movies_copy.columns:
        movies_copy["description"] = (
            "Title " + movies_copy["title"].astype(str) + ". " +
            "Genre " + movies_copy["genre"].astype(str) + ". " +
            "Language " + movies_copy["language"].astype(str) + ". " +
            "Production house " + movies_copy["production_house"].astype(str) + ". " +
            "Released in " + movies_copy["release_year"].astype(str) + ". " +
            "IMDb rating " + movies_copy["imdb_rating"].astype(str) + ". " +
            "Popularity score " + movies_copy["popularity_score"].astype(str) + "."
        )

    return movies_copy


# =========================================================
# CREATE TF-IDF FEATURES
# =========================================================
def create_tfidf_features(movies_df: pd.DataFrame):
    movies_copy = build_movie_descriptions(movies_df)

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_copy["description"])

    with open(processed_data_path / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    with open(processed_data_path / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

    movies_copy[["movie_id", "title", "description"]].to_csv(
        processed_data_path / "movie_text_metadata.csv",
        index=False
    )

    return movies_copy, tfidf, tfidf_matrix


# =========================================================
# CREATE ITEM-ITEM COSINE SIMILARITY MATRIX
# =========================================================
def create_content_similarity_matrix(movies_df: pd.DataFrame, tfidf_matrix):
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=movies_df["movie_id"],
        columns=movies_df["movie_id"]
    )

    with open(processed_data_path / "content_similarity.pkl", "wb") as f:
        pickle.dump(similarity_df, f)

    return similarity_df


# =========================================================
# CONTENT-BASED RECOMMENDATION BY MOVIE
# =========================================================
def recommend_similar_movies(
    movie_id: int,
    movies_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    n_top: int = 10
) -> pd.DataFrame:
    if movie_id not in similarity_df.index:
        raise ValueError(f"movie_id {movie_id} not found in similarity matrix.")

    similarity_scores = similarity_df.loc[movie_id].sort_values(ascending=False)
    similarity_scores = similarity_scores.drop(movie_id, errors="ignore").head(n_top)

    recommendations = movies_df.loc[
        movies_df["movie_id"].isin(similarity_scores.index),
        ["movie_id", "title", "genre", "language", "imdb_rating"]
    ].copy()

    recommendations["movie_id"] = pd.Categorical(
        recommendations["movie_id"],
        categories=similarity_scores.index,
        ordered=True
    )
    recommendations = recommendations.sort_values("movie_id")
    recommendations["similarity_score"] = similarity_scores.values

    return recommendations


# =========================================================
# USER PROFILE FROM LIKED MOVIES
# =========================================================

def build_user_content_profile(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    tfidf_matrix,
    min_rating: int = 4
):
    liked_movie_ids = ratings_df.loc[
        (ratings_df["user_id"] == user_id) & (ratings_df["rating"] >= min_rating),
        "movie_id"
    ].tolist()

    if len(liked_movie_ids) == 0:
        return None, []

    movie_index_lookup = pd.Series(
        data=np.arange(len(movies_df)),
        index=movies_df["movie_id"].values
    )

    liked_indices = [
        movie_index_lookup[mid]
        for mid in liked_movie_ids
        if mid in movie_index_lookup.index
    ]

    if len(liked_indices) == 0:
        return None, []

    user_profile = tfidf_matrix[liked_indices].mean(axis=0)
    user_profile = np.asarray(user_profile)

    return user_profile, liked_movie_ids


# =========================================================
# CONTENT-BASED RECOMMENDATION FOR EXISTING USER
# =========================================================

def recommend_for_existing_user(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    tfidf_matrix,
    n_top: int = 10
) -> pd.DataFrame:
    user_profile, liked_movie_ids = build_user_content_profile(
        user_id=user_id,
        ratings_df=ratings_df,
        movies_df=movies_df,
        tfidf_matrix=tfidf_matrix,
        min_rating=4
    )

    if user_profile is None:
        raise ValueError(f"user_id {user_id} has no liked movies for content profile creation.")

    user_profile = np.asarray(user_profile)
    scores = cosine_similarity(user_profile, tfidf_matrix).flatten()

    recommendation_df = movies_df[["movie_id", "title", "genre", "language", "imdb_rating"]].copy()
    recommendation_df["similarity_score"] = scores

    recommendation_df = recommendation_df.loc[
        ~recommendation_df["movie_id"].isin(liked_movie_ids)
    ]

    recommendation_df = recommendation_df.sort_values(
        by="similarity_score",
        ascending=False
    ).head(n_top)

    return recommendation_df


# =========================================================
# COLD-START ITEM DETECTION
# =========================================================
def get_cold_start_items(ratings_df: pd.DataFrame, min_interactions: int = 5) -> pd.DataFrame:
    item_counts = ratings_df.groupby("movie_id").size().reset_index(name="interaction_count")
    cold_items = item_counts.loc[item_counts["interaction_count"] < min_interactions].copy()
    return cold_items


# =========================================================
# COLD-START ITEM HANDLING
# =========================================================
def recommend_for_cold_start_item(
    cold_movie_id: int,
    movies_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    n_top: int = 10
) -> pd.DataFrame:
    # Since cold-start item has little/no interaction data,
    # use metadata text similarity instead of collaborative signals.
    return recommend_similar_movies(
        movie_id=cold_movie_id,
        movies_df=movies_df,
        similarity_df=similarity_df,
        n_top=n_top
    )


# =========================================================
# COLD-START USER HANDLING USING PREFERRED CATEGORY
# =========================================================
def recommend_for_new_user_by_preference(
    preferred_genre: str,
    movies_df: pd.DataFrame,
    n_top: int = 10
) -> pd.DataFrame:
    filtered = movies_df.loc[
        movies_df["genre"].str.lower() == preferred_genre.lower(),
        ["movie_id", "title", "genre", "language", "imdb_rating", "popularity_score"]
    ].copy()

    filtered = filtered.sort_values(
        by=["imdb_rating", "popularity_score"],
        ascending=False
    ).head(n_top)

    return filtered


# =========================================================
# TEST BLOCK
# =========================================================
if __name__ == "__main__":
    users_df, movies_df, ratings_df = load_data()

    movies_with_desc, tfidf, tfidf_matrix = create_tfidf_features(movies_df)
    similarity_df = create_content_similarity_matrix(movies_with_desc, tfidf_matrix)

    print("TF-IDF feature matrix created successfully.")
    print("TF-IDF shape:", tfidf_matrix.shape)
    print("Content similarity shape:", similarity_df.shape)

    # Example 1: Similar movies for a movie
    sample_movie_id = movies_with_desc["movie_id"].iloc[0]
    print(f"\nContent-based recommendations for movie_id {sample_movie_id}:")
    print(recommend_similar_movies(sample_movie_id, movies_with_desc, similarity_df, n_top=5))

    # Example 2: Existing user recommendation
    sample_user_id = ratings_df["user_id"].iloc[0]
    print(f"\nContent-based recommendations for existing user {sample_user_id}:")
    try:
        print(recommend_for_existing_user(sample_user_id, ratings_df, movies_with_desc, tfidf_matrix, n_top=5))
    except ValueError as e:
        print(e)

    # Example 3: Cold-start item handling
    cold_items_df = get_cold_start_items(ratings_df, min_interactions=5)
    print(f"\nNumber of cold-start items: {len(cold_items_df)}")

    if len(cold_items_df) > 0:
        cold_movie_id = cold_items_df["movie_id"].iloc[0]
        print(f"\nRecommendations for cold-start item {cold_movie_id}:")
        print(recommend_for_cold_start_item(cold_movie_id, movies_with_desc, similarity_df, n_top=5))

    # Example 4: New user using preferred genre
    print("\nRecommendations for new user with preferred genre = Action:")
    print(recommend_for_new_user_by_preference("Action", movies_with_desc, n_top=5))