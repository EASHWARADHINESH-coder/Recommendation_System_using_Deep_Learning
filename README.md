# Recommendation System Using Deep Learning

An end-to-end **Movie Recommendation System** built using **Machine Learning, Deep Learning, NLP, and FastAPI**.

This project demonstrates how to design, build, evaluate, and deploy a modern recommendation engine using multiple recommendation strategies such as **Collaborative Filtering, Content-Based Filtering, Neural Collaborative Filtering (NCF), and Hybrid Recommendation**. It also includes **synthetic data generation**, **sparsity analysis**, **cold-start handling**, and **API deployment**.

---

## Project Overview

The main goal of this project is to recommend relevant movies to users based on their interaction history, movie content, and learned behavioral patterns.

This system is designed to simulate a real-world recommender pipeline by covering:

- Synthetic dataset generation
- Exploratory Data Analysis and validation
- Data preprocessing and feature engineering
- Baseline recommendation models
- Collaborative filtering techniques
- Content-based recommendation using NLP
- Deep learning recommendation using PyTorch
- Hybrid recommendation strategy
- Model evaluation and comparison
- FastAPI deployment for serving recommendations

---

## Problem Statement

In modern digital platforms, users are exposed to a massive number of choices. A recommendation system helps users discover relevant items quickly and improves user engagement.

This project solves that problem by building a movie recommendation engine that can:

- Recommend movies to existing users
- Suggest similar items
- Handle sparse user-item interactions
- Support cold-start users and items
- Combine traditional and deep learning approaches for better accuracy

---

## Objectives

- Build a complete recommendation pipeline from raw data to deployment
- Compare classical recommendation techniques with deep learning models
- Improve recommendation quality using hybrid ranking
- Handle practical recommender system challenges such as sparsity, popularity bias, long-tail items, and cold-start scenarios
- Serve recommendations through an API for real-world usability

---

## Dataset

Since public datasets may not always align with specific business scenarios, this project uses **synthetic movie interaction data** for experimentation and training.

### Generated Files

- `users.csv`
- `movies.csv`
- `ratings.csv`

### Key Fields

**Users**
- `user_id`
- `name`
- `age`
- `location`
- `preferred_category`

**Movies**
- `movie_id`
- `title`
- `genre`
- `language`
- `production_house`
- `imdb_rating`
- `popularity_score`

**Ratings**
- `user_id`
- `movie_id`
- `rating`
- `watch_count`
- `implicit_feedback`
- `timestamp`

### Data Generation Concepts

- Synthetic data generation using **Faker**
- Controlled distributions for realistic watch behavior
- Rating simulation with business rules
- Genre-preference alignment between users and movies

---

## Exploratory Data Analysis and Validation

EDA is performed to verify whether the synthetic dataset behaves like a realistic recommendation dataset.

### Key Validation Checks

- User activity distribution
- Item popularity distribution
- Sparsity analysis
- Heatmap of user-item interactions
- Long-tail item behavior
- Realism checks for rating and engagement patterns

### Why EDA Matters

Recommendation datasets are usually:
- Sparse
- Biased toward popular items
- Uneven across users and items
- Dominated by long-tail content

This analysis helps confirm that the generated data reflects these practical recommender system characteristics.

---

## Preprocessing

Preprocessing transforms raw interaction data into recommendation-ready structures and features.

### Outputs Created

- User-item matrix
- Normalized user-item matrix
- Implicit feedback matrix
- Movie popularity features
- Movie content features
- User profile features

### Importance of Preprocessing

These transformations are essential because recommendation models require structured representations of user-item behavior for:
- similarity computation
- matrix factorization
- implicit feedback learning
- hybrid ranking
- cold-start recommendations

---

## Recommendation Techniques Implemented

## 1. Baseline Recommendation Models

Baseline methods are implemented first to establish a performance benchmark.

### Popularity-Based Recommender
Recommends the most popular movies globally.

### User-Based Collaborative Filtering
Finds users with similar behavior and recommends movies liked by similar users.

### Item-Based Collaborative Filtering
Finds movies similar to the ones a user has already interacted with.

### Matrix Factorization (SVD)
Learns hidden latent factors from the sparse user-item matrix to improve recommendations.

### Why Baselines Are Important
These models provide a strong traditional foundation and help compare performance against more advanced deep learning methods.

---

## 2. Collaborative Filtering Improvements

Classical collaborative filtering is enhanced to address real-world recommendation challenges.

### Improvements Included

- Sparsity handling
- Normalized ratings
- Implicit feedback support
- Popularity bias correction
- Popularity penalty
- Long-tail recommendation boost
- Cold-start fallback for new users
- Profile-based handling for new items

### Practical Impact

These improvements make the system more robust by reducing over-dependence on popular content and improving recommendation diversity.

---

## 3. Content-Based Recommendation using NLP

Content-based recommendation is implemented using movie metadata transformed into textual representations.

### NLP Techniques Used

- TF-IDF Vectorization
- Text feature extraction
- Cosine similarity
- User profile generation from liked items

### Use Cases

- Recommend similar movies
- Recommend movies based on user taste profile
- Handle new items without interaction history
- Support cold-start recommendations for users with limited activity

---

## 4. Deep Learning Recommendation using NCF

The project uses **Neural Collaborative Filtering (NCF)** implemented in **PyTorch**.

### NCF Architecture

- User embedding layer
- Item embedding layer
- Fully connected hidden layers
- ReLU activation
- Dropout regularization
- Output prediction layer

### Feedback Types Supported

**Explicit Feedback**
- Uses rating values
- Loss function: `MSELoss`

**Implicit Feedback**
- Uses interaction labels
- Loss function: `BCEWithLogitsLoss`

### Training Features

- Train-validation split
- Batch training using DataLoader
- Early stopping
- Weight decay regularization

### Why NCF?

Unlike traditional collaborative filtering, NCF can learn **non-linear user-item interaction patterns**, which helps improve recommendation quality in more complex behavior settings.

---

## 5. Hybrid Recommendation System

The final system combines multiple recommendation signals into one ranked output.

### Signals Combined

- Collaborative Filtering score
- Content-Based score
- Neural Collaborative Filtering score

### Additional Re-Ranking

- Popularity penalty
- Long-tail boost

### Benefits of Hybrid Recommendation

- Better recommendation quality
- Improved diversity
- More robust performance
- Better cold-start support
- Balances accuracy and novelty

---

## Model Evaluation

The recommendation models are evaluated using both prediction-based and ranking-based metrics.

### Predictive Metrics

- MSE
- RMSE
- MAE

### Ranking Metrics

- Precision@K
- Recall@K
- MAP@K
- NDCG@K

### Evaluation Strategy

- Time-based train-test split
- Relevant items defined using rating threshold
- Fair comparison of multiple models on the same user set

### Why These Metrics?

Prediction metrics measure numerical accuracy, while ranking metrics evaluate how well the model recommends relevant items at the top positions.

---

## Classical CF vs Deep Learning Comparison

This project compares classical recommenders with Neural Collaborative Filtering.

### Classical Collaborative Filtering

- Simpler to implement
- Faster to train
- Works well with sufficient interaction overlap

### Neural Collaborative Filtering

- Embedding-based architecture
- Captures non-linear interactions
- More flexible and expressive

### Comparison Focus

- Performance
- Training time
- Inference time
- Scalability
- Practical usefulness

### General Observation

- Classical CF is efficient and interpretable
- NCF is more powerful for complex interactions
- Hybrid recommendation often gives the best overall performance

---

## FastAPI Deployment

The final recommendation system is deployed as an API using **FastAPI**.

### Endpoints

- `/`
- `/recommend/{user_id}`
- `/similar-items/{item_id}`

### API Features

- Recommendation serving
- Similar item lookup
- Hybrid recommendation output
- Cold-start fallback support
- Structured API response format

### Why FastAPI?

FastAPI provides:
- Fast API development
- Clean endpoint design
- Easy integration with frontend or external applications
- Production-ready deployment capability

---

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **PyTorch**
- **NLTK / NLP utilities**
- **FastAPI**
- **Uvicorn**
- **Faker**
- **Matplotlib / Seaborn** for analysis and visualization

---

## Project Workflow

1. Generate synthetic movie interaction data  
2. Validate the dataset using EDA  
3. Preprocess interactions and features  
4. Build baseline recommendation models  
5. Improve collaborative filtering methods  
6. Implement content-based recommendation using NLP  
7. Train deep learning model using NCF  
8. Combine models into a hybrid recommender  
9. Evaluate all models using ranking and error metrics  
10. Deploy the best recommendation pipeline using FastAPI  

---

## Key Learning Outcomes

This project demonstrates understanding of:

- Recommendation system fundamentals
- User-item interaction modeling
- Collaborative filtering
- Matrix factorization
- Content-based recommendation
- NLP in recommender systems
- Deep learning with embeddings
- Hybrid recommendation design
- Ranking evaluation metrics
- Cold-start handling
- API deployment using FastAPI

---

## Real-World Challenges Addressed

- Sparse user-item matrix
- Popularity bias
- Long-tail recommendation
- Cold-start users
- Cold-start items
- Model comparison and trade-offs
- Serving recommendations in production-style API format

---

## Future Improvements

- Add real-world datasets such as MovieLens
- Introduce transformer-based content embeddings
- Add user authentication and profile-based recommendation API
- Deploy on cloud platforms like Render or Railway
- Add frontend integration for live recommendation demo
- Use approximate nearest neighbor search for faster retrieval
- Track experiments using MLflow or Weights & Biases

---

## Conclusion

This project presents a complete recommendation system pipeline built using both traditional and modern approaches. It starts from synthetic data generation and extends all the way to deep learning-based recommendation and FastAPI deployment.

By combining **Collaborative Filtering**, **Content-Based Filtering**, **Neural Collaborative Filtering**, and **Hybrid Ranking**, the system becomes more accurate, flexible, and practical for real-world recommendation tasks.

---

## Interview Summary

This project can be explained in interviews as:

> Built an end-to-end movie recommendation system using synthetic interaction data, collaborative filtering, NLP-based content recommendation, PyTorch Neural Collaborative Filtering, and hybrid ranking. Evaluated the models using RMSE, Precision@K, Recall@K, MAP@K, and NDCG@K, and deployed the final system using FastAPI.

---

## Author

**Eashwaradhinesh K**

If you are using this project for portfolio or interview presentation, focus on:
- problem statement
- preprocessing
- recommendation techniques
- evaluation strategy
- hybrid model benefits
- deployment architecture
