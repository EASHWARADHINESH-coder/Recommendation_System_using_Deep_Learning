\# Recommendation System Using Deep Learning



A complete end-to-end Movie Recommendation System built using Machine Learning, Deep Learning, NLP, and API deployment.  

This project covers synthetic data generation, preprocessing, baseline recommenders, collaborative filtering, Neural Collaborative Filtering (NCF), hybrid recommendation, evaluation, and FastAPI serving.



\---



\## Project Overview



This project recommends movies to users using multiple recommendation techniques.



It includes:



\- Synthetic movie ratings data generation

\- Data validation and sparsity analysis

\- Baseline recommendation models

\- Collaborative filtering models

\- Content-based recommendation using NLP

\- Deep learning recommendation using PyTorch NCF

\- Hybrid recommendation system

\- Model comparison and evaluation

\- FastAPI deployment



\---



\## 1. Synthetic Data Generation



The dataset is generated artificially for experimentation and training.



\### Concepts Used

\- \*\*Faker\*\* for synthetic names, movie titles, locations, and timestamps

\- \*\*Controlled distributions\*\* for ratings and watch behavior

\- \*\*Business rules\*\* such as matching user preferred category with movie genre



\### Generated Files

\- `users.csv`

\- `movies.csv`

\- `ratings.csv`



\### Important Fields

\- \*\*Users:\*\* user\_id, name, age, location, preferred\_category

\- \*\*Movies:\*\* movie\_id, title, genre, language, production\_house, imdb\_rating, popularity\_score

\- \*\*Ratings:\*\* user\_id, movie\_id, rating, watch\_count, implicit\_feedback, timestamp



\---



\## 2. EDA and Validation



Exploratory Data Analysis is used to validate whether the synthetic dataset behaves like a real recommendation dataset.



\### Key Checks

\- User activity distribution

\- Item popularity distribution

\- Sparsity analysis

\- Heatmap of user-item interactions

\- Long-tail distribution

\- Realism checks



\### Goal

To confirm that the dataset contains:

\- sparsity

\- popularity bias

\- long-tail items

\- variable user activity



\---



\## 3. Preprocessing



Preprocessing converts raw interaction data into recommender-ready matrices and features.



\### Outputs Created

\- \*\*User-item matrix\*\*

\- \*\*Normalized user-item matrix\*\*

\- \*\*Implicit feedback matrix\*\*

\- \*\*Movie popularity features\*\*

\- \*\*Movie content features\*\*

\- \*\*User profile features\*\*



\### Why It Is Needed

These processed matrices are required for:

\- collaborative filtering

\- SVD

\- implicit recommendation

\- cold-start handling

\- hybrid recommendation



\---



\## 4. Baseline Recommendation Models



Baseline recommenders are implemented to compare traditional methods.



\### Models Included



\#### Popularity-Based Recommender

Recommends globally popular items.



\#### User-Based Collaborative Filtering

Finds similar users and recommends items liked by them.



\#### Item-Based Collaborative Filtering

Finds similar items to those already liked by a user.



\#### Matrix Factorization (SVD)

Learns latent user-item factors from sparse interaction data.



\### Purpose

These models act as strong traditional baselines before deep learning.



\---



\## 5. Collaborative Filtering Improvements



The classical recommendation system is improved to handle practical real-world issues.



\### Improvements Included

\- \*\*Sparsity handling\*\*

&#x20; - SVD

&#x20; - implicit feedback

&#x20; - normalized ratings



\- \*\*Popularity bias correction\*\*

&#x20; - popularity penalty



\- \*\*Long-tail recommendation\*\*

&#x20; - boost for niche items



\- \*\*Cold-start handling\*\*

&#x20; - new user fallback

&#x20; - new item handling

&#x20; - profile-based recommendation



\---



\## 6. Content-Based Recommendation using NLP



Content-based recommendation is implemented using movie metadata converted into text descriptions.



\### NLP Concepts Used

\- TF-IDF Vectorization

\- Cosine Similarity

\- Item-to-item text similarity

\- User content profile from liked movies



\### Use Cases

\- Similar movie recommendation

\- Existing user content-based recommendation

\- Cold-start item recommendation

\- New user recommendation by preferred genre



\---



\## 7. Deep Learning Recommendation Model (NCF)



Neural Collaborative Filtering (NCF) is implemented using PyTorch.



\### NCF Architecture

\- User embeddings

\- Item embeddings

\- Fully connected hidden layers

\- ReLU activation

\- Dropout regularization

\- Output prediction layer



\### Supported Feedback Types

\- \*\*Explicit feedback\*\*

&#x20; - uses rating values

&#x20; - loss: `MSELoss`



\- \*\*Implicit feedback\*\*

&#x20; - uses binary interaction labels

&#x20; - loss: `BCEWithLogitsLoss`



\### Training Features

\- Train-validation split

\- Early stopping

\- Weight decay regularization

\- Batch training using DataLoader



\### Why NCF

NCF captures non-linear user-item relationships better than classical CF.



\---



\## 8. Hybrid Recommendation System



A hybrid recommender combines multiple signals into one final ranked list.



\### Signals Combined

\- Collaborative score (SVD)

\- Content-based score (TF-IDF)

\- NCF score



\### Additional Re-Ranking

\- popularity penalty

\- long-tail boost



\### Benefit

This improves:

\- recommendation quality

\- diversity

\- robustness

\- cold-start support



\---



\## 9. Evaluation



Recommendation models are evaluated using both predictive and ranking metrics.



\### Explicit Metrics

\- MSE

\- RMSE

\- MAE



\### Ranking Metrics

\- Precision@K

\- Recall@K

\- MAP@K

\- NDCG@K



\### Evaluation Strategy

\- Time-based train-test split

\- Relevant items defined using rating threshold

\- Multiple model comparison on the same test users



\---



\## 10. Classical CF vs NCF Comparison



This project compares traditional recommenders with deep learning models.



\### Classical CF

\- similarity-based

\- factorization-based

\- simpler and faster to train



\### NCF

\- embedding-based

\- deep neural network

\- captures non-linear interactions



\### Comparison Focus

\- performance

\- training time

\- inference time

\- scalability



\### General Observation

\- Classical CF is simpler and efficient

\- NCF is more flexible and powerful

\- Hybrid systems often perform best overall



\---



\## 11. FastAPI Deployment



The recommendation engine is exposed as an API using FastAPI.



\### Available Endpoints

\- `/`

\- `/recommend/{user\_id}`

\- `/similar-items/{item\_id}`



\### Features

\- hybrid recommendation serving

\- cold-start fallback

\- item similarity endpoint

\- API-ready recommendation response format



\---

