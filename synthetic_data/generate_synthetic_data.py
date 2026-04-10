# IMPORT LIBRARIES
import pandas as pd
import numpy as np
from faker import Faker
from pathlib import Path
import random

fake = Faker()
np.random.seed(42)
random.seed(42)

# CONFIG
num_users = 7500
num_movies = 3500
num_ratings = 125000

data_path = Path(r"D:\DS PROJECTS SUBMISSION\Recommendation_System_Final\data\raw")
data_path.mkdir(parents=True, exist_ok=True)

categories = [
    "Action", "Comedy", "Drama", "Thriller",
    "Romance", "Sci-Fi", "Horror", "Documentary"
]

production_houses = [
    "Universal Studios", "Walt Disney", "Legendary", "Warner Bros",
    "Marvel Studios", "DC", "Sony Pictures", "Paramount Pictures",
    "Lionsgate Films", "MGM Studios"
]

languages = ["English", "Tamil", "Hindi", "Korean", "Spanish", "French"]

# GENERATE USERS
users = []

for user_id in range(1, num_users + 1):
    users.append({
        "user_id": user_id,
        "name": fake.name(),
        "age": random.randint(18, 70),
        "location": fake.city(),
        "preferred_category": random.choice(categories)
    })

user_df = pd.DataFrame(users)
user_df.to_csv(data_path / "users.csv", index=False)

# GENERATE MOVIES
movies = []

for movie_id in range(1, num_movies + 1):
    genre = random.choice(categories)
    movies.append({
        "movie_id": movie_id,
        "title": f"{fake.word().title()} {fake.word().title()}",
        "genre": genre,
        "budget": random.randint(1_000_000, 300_000_000),
        "production_house": random.choice(production_houses),
        "release_year": random.randint(2000, 2025),
        "duration_mins": random.randint(80, 180),
        "language": random.choice(languages),
        "imdb_rating": round(random.uniform(4.5, 9.5), 1),
        "popularity_score": random.randint(1, 100)
    })

movie_df = pd.DataFrame(movies)
movie_df.to_csv(data_path / "movies.csv", index=False)

# GENERATE RATINGS
ratings_data = []
used_pairs = set()

while len(ratings_data) < num_ratings:
    user = user_df.sample(1).iloc[0]
    movie = movie_df.sample(1).iloc[0]

    pair = (user["user_id"], movie["movie_id"])
    if pair in used_pairs:
        continue

    used_pairs.add(pair)

    if user["preferred_category"] == movie["genre"]:
        rating_value = np.random.choice([4, 5], p=[0.4, 0.6])
        watch_count = np.random.randint(2, 8)
    else:
        rating_value = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
        watch_count = np.random.randint(1, 4)

    ratings_data.append({
        "user_id": int(user["user_id"]),
        "movie_id": int(movie["movie_id"]),
        "rating": int(rating_value),
        "watch_count": int(watch_count),
        "implicit_feedback": 1 if rating_value >= 4 else 0,
        "timestamp": fake.date_time_between(
            start_date="-1y",
            end_date="now"
        ).strftime("%Y-%m-%d %H:%M:%S")
    })

rating_df = pd.DataFrame(ratings_data)
rating_df.to_csv(data_path / "ratings.csv", index=False)

print("Synthetic movie recommendation data generation completed successfully.")
print("Users shape:", user_df.shape)
print("Movies shape:", movie_df.shape)
print("Ratings shape:", rating_df.shape)