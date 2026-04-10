# IMPORT LIBRARIES
import pandas as pd
from pathlib import Path

# RAW DATA PATH
data_path = Path(r"D:\DS PROJECTS SUBMISSION\Recommendation_System_Final\data\raw")

# LOAD USERS
def load_users() -> pd.DataFrame:
    return pd.read_csv(data_path / "users.csv")

# LOAD MOVIES
def load_movies() -> pd.DataFrame:
    return pd.read_csv(data_path / "movies.csv")

# LOAD RATINGS
def load_ratings() -> pd.DataFrame:
    return pd.read_csv(data_path / "ratings.csv")

# TEST
if __name__ == "__main__":
    users_df = load_users()
    movies_df = load_movies()
    ratings_df = load_ratings()

    print("Users shape:", users_df.shape)
    print("Movies shape:", movies_df.shape)
    print("Ratings shape:", ratings_df.shape)

    print("\nUsers sample:")
    print(users_df.head())

    print("\nMovies sample:")
    print(movies_df.head())

    print("\nRatings sample:")
    print(ratings_df.head())