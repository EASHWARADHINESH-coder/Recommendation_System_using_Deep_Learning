# =========================================================
# IMPORT LIBRARIES
# =========================================================
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_loader import load_users, load_movies, load_ratings
from preprocessing import (
    create_user_item_matrix,
    create_normalized_user_item_matrix,
    create_implicit_feedback_matrix,
    create_movie_popularity_features,
    save_pickle
)
from collaborative_filtering import (
    create_svd_model,
    create_predicted_rating_matrix,
    create_item_similarity_from_implicit,
    build_popularity_lookup,
    recommend_product_svd,
    recommend_product_implicit
)
from ncf_recommender import (
    encode_ids,
    train_valid_split,
    RatingsDataset,
    NeuralCollaborativeFiltering,
    train_ncf_model,
    evaluate_explicit_model,
    evaluate_implicit_model,
    recommend_ncf
)


# =========================================================
# PATHS
# =========================================================
processed_data_path = Path(r"D:\DS PROJECTS SUBMISSION\Recommendation_System_Final\data\processed")
processed_data_path.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPER: SAVE COMPARISON RESULTS
# =========================================================
def save_comparison_results(results_df: pd.DataFrame, filename: str = "classical_vs_ncf_comparison.csv"):
    output_path = processed_data_path / filename
    results_df.to_csv(output_path, index=False)
    print(f"\nComparison results saved to: {output_path}")


# =========================================================
# PREPARE CLASSICAL DATA
# =========================================================
def prepare_classical_inputs(ratings_df: pd.DataFrame):
    user_item_matrix = create_user_item_matrix(ratings_df)
    normalized_user_item_matrix = create_normalized_user_item_matrix(ratings_df)
    implicit_feedback_matrix = create_implicit_feedback_matrix(ratings_df, threshold=4)
    movie_popularity_features = create_movie_popularity_features(ratings_df)

    save_pickle(user_item_matrix, "user_item_matrix.pkl")
    save_pickle(normalized_user_item_matrix, "normalized_user_item_matrix.pkl")
    save_pickle(implicit_feedback_matrix, "implicit_feedback_matrix.pkl")
    save_pickle(movie_popularity_features, "movie_popularity_features.pkl")

    popularity_lookup = build_popularity_lookup(movie_popularity_features)

    return (
        user_item_matrix,
        normalized_user_item_matrix,
        implicit_feedback_matrix,
        movie_popularity_features,
        popularity_lookup
    )


# =========================================================
# CLASSICAL CF: SVD EVALUATION
# =========================================================
def evaluate_classical_svd(
    ratings_df: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    normalized_user_item_matrix: pd.DataFrame,
    popularity_lookup: pd.DataFrame
):
    start_train = time.time()

    svd_model, user_factors_df, item_factors_df = create_svd_model(
        normalized_user_item_matrix,
        n_components=50
    )

    predicted_ratings_df = create_predicted_rating_matrix(
        normalized_user_item_matrix,
        user_factors_df,
        item_factors_df
    )

    svd_train_time = time.time() - start_train

    # Explicit rating reconstruction evaluation
    actuals = []
    preds = []

    for _, row in ratings_df.iterrows():
        user_id = row["user_id"]
        movie_id = row["movie_id"]
        rating = row["rating"]

        if user_id in predicted_ratings_df.index and movie_id in predicted_ratings_df.columns:
            pred_score = predicted_ratings_df.loc[user_id, movie_id]
            actuals.append(rating)
            preds.append(pred_score)

    actuals = np.array(actuals, dtype=float)
    preds = np.array(preds, dtype=float)

    mse = np.mean((preds - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - actuals))

    # Recommendation inference time
    sample_users = list(user_item_matrix.index[:100])
    start_infer = time.time()

    for user_id in sample_users:
        _ = recommend_product_svd(
            user_id=user_id,
            original_user_item_matrix=user_item_matrix,
            predicted_ratings_df=predicted_ratings_df,
            popularity_lookup=popularity_lookup,
            n_top=10
        )

    svd_inference_time = time.time() - start_infer

    return {
        "Model": "Classical CF (SVD)",
        "Feedback_Type": "Explicit",
        "Train_Time_Seconds": round(svd_train_time, 4),
        "Inference_Time_100_Users_Seconds": round(svd_inference_time, 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "Accuracy": np.nan
    }


# =========================================================
# CLASSICAL CF: IMPLICIT EVALUATION
# =========================================================
def evaluate_classical_implicit(
    implicit_feedback_matrix: pd.DataFrame,
    popularity_lookup: pd.DataFrame
):
    start_train = time.time()

    item_similarity_df = create_item_similarity_from_implicit(implicit_feedback_matrix)

    implicit_train_time = time.time() - start_train

    sample_users = list(implicit_feedback_matrix.index[:100])
    start_infer = time.time()

    for user_id in sample_users:
        _ = recommend_product_implicit(
            user_id=user_id,
            implicit_feedback_matrix=implicit_feedback_matrix,
            item_similarity_df=item_similarity_df,
            popularity_lookup=popularity_lookup,
            n_top=10
        )

    implicit_inference_time = time.time() - start_infer

    return {
        "Model": "Classical CF (Implicit Item-Item)",
        "Feedback_Type": "Implicit",
        "Train_Time_Seconds": round(implicit_train_time, 4),
        "Inference_Time_100_Users_Seconds": round(implicit_inference_time, 4),
        "MSE": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "Accuracy": np.nan
    }


# =========================================================
# NCF EVALUATION
# =========================================================
def evaluate_ncf_pipeline(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    feedback_type: str = "explicit"
):
    encoded_ratings_df, user_to_index, item_to_index, index_to_user, index_to_item = encode_ids(ratings_df)

    train_df, valid_df = train_valid_split(
        encoded_ratings_df,
        valid_ratio=0.2,
        random_state=42
    )

    train_dataset = RatingsDataset(train_df, feedback_type=feedback_type)
    valid_dataset = RatingsDataset(valid_df, feedback_type=feedback_type)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=False)

    model = NeuralCollaborativeFiltering(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        embedding_dim=32,
        hidden_dims=[128, 64, 32],
        dropout=0.25
    )

    start_train = time.time()

    model, history = train_ncf_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        feedback_type=feedback_type,
        learning_rate=1e-3,
        weight_decay=1e-5,
        epochs=20,
        patience=4
    )

    ncf_train_time = time.time() - start_train

    if feedback_type == "explicit":
        metrics = evaluate_explicit_model(model, valid_loader)
        mse = metrics["MSE"]
        rmse = metrics["RMSE"]
        mae = metrics["MAE"]
        accuracy = np.nan
        model_name = "NCF"
    else:
        metrics = evaluate_implicit_model(model, valid_loader)
        mse = np.nan
        rmse = np.nan
        mae = np.nan
        accuracy = metrics["Accuracy"]
        model_name = "NCF"

    sample_users = list(ratings_df["user_id"].unique()[:100])

    start_infer = time.time()

    for user_id in sample_users:
        if user_id in user_to_index:
            _ = recommend_ncf(
                model=model,
                user_id=user_id,
                ratings_df=ratings_df,
                movies_df=movies_df,
                user_to_index=user_to_index,
                item_to_index=item_to_index,
                top_n=10,
                feedback_type=feedback_type
            )

    ncf_inference_time = time.time() - start_infer

    return {
        "Model": model_name,
        "Feedback_Type": feedback_type.capitalize(),
        "Train_Time_Seconds": round(ncf_train_time, 4),
        "Inference_Time_100_Users_Seconds": round(ncf_inference_time, 4),
        "MSE": round(float(mse), 4) if not pd.isna(mse) else np.nan,
        "RMSE": round(float(rmse), 4) if not pd.isna(rmse) else np.nan,
        "MAE": round(float(mae), 4) if not pd.isna(mae) else np.nan,
        "Accuracy": round(float(accuracy), 4) if not pd.isna(accuracy) else np.nan
    }


# =========================================================
# INTERPRETATION TABLE
# =========================================================
def build_scalability_interpretation(results_df: pd.DataFrame) -> pd.DataFrame:
    interpretation_rows = []

    for _, row in results_df.iterrows():
        model = row["Model"]

        if "Classical CF (SVD)" in model:
            interpretation = "Strong traditional model; usually faster to train than NCF and easier to scale than user-based similarity methods."
        elif "Implicit Item-Item" in model:
            interpretation = "Lightweight classical recommender; similarity-based and often simpler, but may become expensive as item count grows."
        elif "NCF" in model:
            interpretation = "Deep model; slower training but better ability to learn complex user-item patterns and scalable at inference after training."
        else:
            interpretation = "General recommender model."

        interpretation_rows.append({
            "Model": model,
            "Performance_vs_Scalability_Interpretation": interpretation
        })

    return pd.DataFrame(interpretation_rows)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    # LOAD DATA
    users_df = load_users()
    movies_df = load_movies()
    ratings_df = load_ratings()

    # PREPARE CLASSICAL INPUTS
    (
        user_item_matrix,
        normalized_user_item_matrix,
        implicit_feedback_matrix,
        movie_popularity_features,
        popularity_lookup
    ) = prepare_classical_inputs(ratings_df)

    # CLASSICAL CF RESULTS
    svd_result = evaluate_classical_svd(
        ratings_df=ratings_df,
        user_item_matrix=user_item_matrix,
        normalized_user_item_matrix=normalized_user_item_matrix,
        popularity_lookup=popularity_lookup
    )

    implicit_result = evaluate_classical_implicit(
        implicit_feedback_matrix=implicit_feedback_matrix,
        popularity_lookup=popularity_lookup
    )

    # NCF RESULTS
    ncf_explicit_result = evaluate_ncf_pipeline(
        ratings_df=ratings_df,
        movies_df=movies_df,
        feedback_type="explicit"
    )

    ncf_implicit_result = evaluate_ncf_pipeline(
        ratings_df=ratings_df,
        movies_df=movies_df,
        feedback_type="implicit"
    )

    # COMBINE RESULTS
    results_df = pd.DataFrame([
        svd_result,
        implicit_result,
        ncf_explicit_result,
        ncf_implicit_result
    ])

    print("\n================ COMPARISON: CLASSICAL CF vs NCF ================\n")
    print(results_df)

    interpretation_df = build_scalability_interpretation(results_df)

    print("\n================ PERFORMANCE vs SCALABILITY INTERPRETATION ================\n")
    print(interpretation_df)

    save_comparison_results(results_df)