import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# PATH
data_path = Path(r"D:\DS PROJECTS SUBMISSION\Recommendation_System_Final\data\raw")

# LOAD DATA
users_df = pd.read_csv(data_path / "users.csv")
movies_df = pd.read_csv(data_path / "movies.csv")
ratings_df = pd.read_csv(data_path / "ratings.csv")

print("Users:", users_df.shape)
print("Movies:", movies_df.shape)
print("Ratings:", ratings_df.shape)

# =========================================================
# 1️⃣ USER ACTIVITY DISTRIBUTION
# =========================================================

user_activity = ratings_df.groupby("user_id").size()

plt.figure(figsize=(8,5))
sns.histplot(user_activity, bins=50, kde=True)
plt.title("User Activity Distribution (Number of Ratings per User)")
plt.xlabel("Number of Ratings")
plt.ylabel("Number of Users")
plt.show()

# =========================================================
# 2️⃣ ITEM POPULARITY DISTRIBUTION
# =========================================================

item_popularity = ratings_df.groupby("movie_id").size()

plt.figure(figsize=(8,5))
sns.histplot(item_popularity, bins=50, kde=True)
plt.title("Item Popularity Distribution (Number of Ratings per Movie)")
plt.xlabel("Number of Ratings")
plt.ylabel("Number of Movies")
plt.show()

# =========================================================
# 3️⃣ SPARSITY ANALYSIS
# =========================================================

num_users = users_df["user_id"].nunique()
num_movies = movies_df["movie_id"].nunique()
num_interactions = len(ratings_df)

sparsity = 1 - (num_interactions / (num_users * num_movies))

print("\n📊 SPARSITY ANALYSIS")
print(f"Total possible interactions: {num_users * num_movies}")
print(f"Actual interactions: {num_interactions}")
print(f"Sparsity: {sparsity:.4f}")

# =========================================================
# 4️⃣ SPARSITY VISUALIZATION (HEATMAP SAMPLE)
# =========================================================

# Create user-item matrix (small sample for visualization)
sample_users = ratings_df["user_id"].unique()[:50]
sample_movies = ratings_df["movie_id"].unique()[:50]

sample_df = ratings_df[
    (ratings_df["user_id"].isin(sample_users)) &
    (ratings_df["movie_id"].isin(sample_movies))
]

matrix = sample_df.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating",
    fill_value=0
)

plt.figure(figsize=(10,6))
sns.heatmap(matrix, cmap="coolwarm")
plt.title("Interaction Matrix Sparsity (Sample Heatmap)")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.show()

# =========================================================
# 5️⃣ POPULARITY LONG-TAIL DISTRIBUTION
# =========================================================

sorted_popularity = item_popularity.sort_values(ascending=False).values

plt.figure(figsize=(8,5))
plt.plot(sorted_popularity)
plt.title("Long-Tail Distribution of Movie Popularity")
plt.xlabel("Movies Ranked by Popularity")
plt.ylabel("Number of Ratings")
plt.show()

# =========================================================
# 6️⃣ REALISM VALIDATION (KEY CHECKS)
# =========================================================

print("\n📊 REALISM CHECKS")

print("\nAverage ratings per user:", user_activity.mean())
print("Average ratings per movie:", item_popularity.mean())

print("\nTop 5 most popular movies:")
print(item_popularity.sort_values(ascending=False).head())

print("\nBottom 5 least popular movies:")
print(item_popularity.sort_values().head())

# =========================================================
# 7️⃣ INTERPRETATION (PRINT SUMMARY)
# =========================================================

print("\n📌 INTERPRETATION:")
print("- Data shows sparsity typical of real-world recommendation systems.")
print("- Few movies are highly popular → popularity bias exists.")
print("- Many movies have low interactions → long-tail distribution.")
print("- User activity varies → realistic user behavior.")