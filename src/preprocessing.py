# IMPORT LIBRARIES
import pandas as pd
import pickle
from pathlib import Path

# PROCESSED DATA PATH
data_path = Path(r"D:\DS PROJECTS SUBMISSION\Recommendation_System_Final\data\processed")
data_path.mkdir(parents=True, exist_ok=True)


# FUNCTION TO CREATE USER-ITEM MATRIX
def create_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    user_item_matrix = ratings_df.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        aggfunc="mean",
        fill_value=0
    )
    return user_item_matrix


# NORMALIZE RATINGS (USER MEAN CENTERING)
def create_normalized_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    ratings_copy = ratings_df.copy()

    user_mean = ratings_copy.groupby("user_id")["rating"].mean()

    ratings_copy["normalized_rating"] = ratings_copy.apply(
        lambda row: row["rating"] - user_mean.loc[row["user_id"]],
        axis=1
    )

    normalized_matrix = ratings_copy.pivot_table(
        index="user_id",
        columns="movie_id",
        values="normalized_rating",
        aggfunc="mean",
        fill_value=0
    )

    return normalized_matrix


# IMPLICIT FEEDBACK MATRIX
def create_implicit_feedback_matrix(ratings_df: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    ratings_copy = ratings_df.copy()

    ratings_copy["implicit_feedback"] = (ratings_copy["rating"] >= threshold).astype(int)

    implicit_matrix = ratings_copy.pivot_table(
        index="user_id",
        columns="movie_id",
        values="implicit_feedback",
        aggfunc="max",
        fill_value=0
    )

    return implicit_matrix


# MOVIE POPULARITY / LONG-TAIL FEATURES
def create_movie_popularity_features(ratings_df: pd.DataFrame) -> pd.DataFrame:
    movie_stats = ratings_df.groupby("movie_id").agg(
        interaction_count=("rating", "count"),
        average_rating=("rating", "mean")
    ).reset_index()

    max_count = movie_stats["interaction_count"].max()
    movie_stats["popularity_ratio"] = movie_stats["interaction_count"] / max_count

    long_tail_threshold = movie_stats["interaction_count"].quantile(0.80)
    movie_stats["is_long_tail"] = (
        movie_stats["interaction_count"] <= long_tail_threshold
    ).astype(int)

    return movie_stats


# MOVIE CONTENT FEATURE MATRIX
def create_movie_content_features(movies_df: pd.DataFrame) -> pd.DataFrame:
    movies_copy = movies_df.copy()

    numeric_cols = ["imdb_rating", "popularity_score", "release_year", "duration_mins", "budget"]
    categorical_cols = ["genre", "language", "production_house"]

    for col in numeric_cols:
        if col in movies_copy.columns:
            min_val = movies_copy[col].min()
            max_val = movies_copy[col].max()
            if max_val > min_val:
                movies_copy[col] = (movies_copy[col] - min_val) / (max_val - min_val)
            else:
                movies_copy[col] = 0.0

    cat_df = pd.get_dummies(movies_copy[categorical_cols], drop_first=False)
    num_df = movies_copy[numeric_cols]

    feature_df = pd.concat([movies_copy[["movie_id"]], num_df, cat_df], axis=1)

    return feature_df


# USER PROFILE FEATURES
def create_user_profile_features(users_df: pd.DataFrame) -> pd.DataFrame:
    users_copy = users_df.copy()

    # SCALE AGE
    if "age" in users_copy.columns:
        min_age = users_copy["age"].min()
        max_age = users_copy["age"].max()
        if max_age > min_age:
            users_copy["age"] = (users_copy["age"] - min_age) / (max_age - min_age)
        else:
            users_copy["age"] = 0.0

    user_profile_df = pd.get_dummies(
        users_copy[["user_id", "preferred_category", "age"]],
        columns=["preferred_category"],
        drop_first=False
    )

    user_profile_df = user_profile_df.set_index("user_id")
    return user_profile_df


# SAVE PICKLE FILE
def save_pickle(obj, filename: str) -> None:
    with open(data_path / filename, "wb") as f:
        pickle.dump(obj, f)


# TEST BLOCK
if __name__ == "__main__":
    raw_data_path = Path(r"D:\DS PROJECTS SUBMISSION\Recommendation_System_Final\data\raw")

    ratings_df = pd.read_csv(raw_data_path / "ratings.csv")
    users_df = pd.read_csv(raw_data_path / "users.csv")
    movies_df = pd.read_csv(raw_data_path / "movies.csv")

    user_item_matrix = create_user_item_matrix(ratings_df)
    save_pickle(user_item_matrix, "user_item_matrix.pkl")

    normalized_user_item_matrix = create_normalized_user_item_matrix(ratings_df)
    save_pickle(normalized_user_item_matrix, "normalized_user_item_matrix.pkl")

    implicit_feedback_matrix = create_implicit_feedback_matrix(ratings_df, threshold=4)
    save_pickle(implicit_feedback_matrix, "implicit_feedback_matrix.pkl")

    movie_popularity_features = create_movie_popularity_features(ratings_df)
    save_pickle(movie_popularity_features, "movie_popularity_features.pkl")

    movie_content_features = create_movie_content_features(movies_df)
    save_pickle(movie_content_features, "movie_content_features.pkl")

    user_profile_features = create_user_profile_features(users_df)
    save_pickle(user_profile_features, "user_profile_features.pkl")

    print("All matrices and cold-start features created successfully.")
    print("User-item matrix shape:", user_item_matrix.shape)
    print("Normalized matrix shape:", normalized_user_item_matrix.shape)
    print("Implicit matrix shape:", implicit_feedback_matrix.shape)
    print("Movie popularity features shape:", movie_popularity_features.shape)
    print("Movie content features shape:", movie_content_features.shape)
    print("User profile features shape:", user_profile_features.shape)