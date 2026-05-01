# =========================================================
# STREAMLIT RECOMMENDATION DASHBOARD
# =========================================================

import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Movie Recommendation Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.sidebar.title("⚙️ Dashboard Settings")

API_BASE_URL = st.sidebar.text_input(
    "FastAPI Base URL",
    value="http://127.0.0.1:8000"
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10,
    step=5
)

st.sidebar.markdown("---")
st.sidebar.info(
    "First start FastAPI:\n\n"
    "`uvicorn app_fastapi:app --reload`\n\n"
    "Then run:\n\n"
    "`streamlit run streamlit_recommendation_dashboard.py`"
)


@st.cache_data(show_spinner=False)
def call_recommend_api(api_base_url: str, user_id: int, top_n: int):
    url = f"{api_base_url}/recommend/{user_id}"
    response = requests.get(url, params={"top_n": top_n}, timeout=60)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def call_similar_items_api(api_base_url: str, item_id: int, top_n: int):
    url = f"{api_base_url}/similar-items/{item_id}"
    response = requests.get(url, params={"top_n": top_n}, timeout=60)
    response.raise_for_status()
    return response.json()


def recommendations_to_df(response_json: dict) -> pd.DataFrame:
    data = response_json.get("recommendations", [])
    return pd.DataFrame(data)


def similar_items_to_df(response_json: dict) -> pd.DataFrame:
    data = response_json.get("similar_items", [])
    return pd.DataFrame(data)


def show_cards(df: pd.DataFrame):
    if df.empty:
        st.warning("No records found.")
        return

    for _, row in df.iterrows():
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 1.2, 1.2, 1.2])

            with col1:
                st.subheader(f"🎬 {row.get('title', 'Unknown Title')}")
                st.caption(f"Movie ID: {row.get('movie_id', '-')}")
                st.write(row.get("explanation", "No explanation available."))

            with col2:
                st.metric("Genre", row.get("genre", "-"))

            with col3:
                st.metric("Language", row.get("language", "-"))

            with col4:
                imdb = row.get("imdb_rating", "-")
                popularity = row.get("popularity_score", "-")
                st.metric("IMDb", imdb)
                st.caption(f"Popularity: {popularity}")


st.title("🎬 Streamlit Recommendation Dashboard")

st.markdown(
    """
This dashboard demonstrates:

- Recommended items
- Similar items
- Explanation panel
- Cold-start demo
- System design and scalability
"""
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "👤 Recommended Items",
        "🔎 Similar Items",
        "🧠 Explanation Panel",
        "🧊 Cold-Start Demo",
        "🏗️ System Design"
    ]
)


# =========================================================
# TAB 1: RECOMMENDED ITEMS
# =========================================================
with tab1:
    st.header("👤 Recommended Items")

    user_id = st.number_input(
        "Enter Existing User ID",
        min_value=1,
        value=1,
        step=1,
        key="user_id"
    )

    if st.button("Generate Recommendations", key="recommend_btn"):
        try:
            with st.spinner("Fetching recommendations..."):
                response_json = call_recommend_api(API_BASE_URL, int(user_id), top_n)

            rec_df = recommendations_to_df(response_json)

            st.success(f"Strategy Used: {response_json.get('strategy', '-')}")
            st.dataframe(rec_df, use_container_width=True)

            st.subheader("Recommendation Cards")
            show_cards(rec_df)

        except requests.exceptions.ConnectionError:
            st.error("FastAPI is not running. Start it using: uvicorn app_fastapi:app --reload")
        except Exception as e:
            st.error(f"Error: {e}")


# =========================================================
# TAB 2: SIMILAR ITEMS
# =========================================================
with tab2:
    st.header("🔎 Similar Items")

    item_id = st.number_input(
        "Enter Movie ID",
        min_value=1,
        value=1,
        step=1,
        key="item_id"
    )

    if st.button("Find Similar Items", key="similar_btn"):
        try:
            with st.spinner("Finding similar movies..."):
                response_json = call_similar_items_api(API_BASE_URL, int(item_id), top_n)

            similar_df = similar_items_to_df(response_json)

            st.success(f"Similar items for Movie ID: {response_json.get('item_id', item_id)}")
            st.dataframe(similar_df, use_container_width=True)

            st.subheader("Similar Item Cards")
            show_cards(similar_df)

        except requests.exceptions.ConnectionError:
            st.error("FastAPI is not running. Start it using: uvicorn app_fastapi:app --reload")
        except Exception as e:
            st.error(f"Error: {e}")


# =========================================================
# TAB 3: EXPLANATION PANEL
# =========================================================
with tab3:
    st.header("🧠 Explanation Panel")

    explanation_user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        value=1,
        step=1,
        key="explanation_user_id"
    )

    if st.button("Show Explanations", key="explanation_btn"):
        try:
            response_json = call_recommend_api(API_BASE_URL, int(explanation_user_id), top_n)
            rec_df = recommendations_to_df(response_json)

            st.info(f"Strategy Used: {response_json.get('strategy', '-')}")

            if not rec_df.empty:
                selected_title = st.selectbox(
                    "Select Movie",
                    rec_df["title"].tolist()
                )

                selected_row = rec_df[rec_df["title"] == selected_title].iloc[0]

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric("Movie ID", selected_row.get("movie_id", "-"))
                    st.metric("Genre", selected_row.get("genre", "-"))
                    st.metric("IMDb Rating", selected_row.get("imdb_rating", "-"))

                with col2:
                    st.subheader("Why recommended?")
                    st.success(selected_row.get("explanation", "No explanation available."))

                    st.markdown(
                        """
### Explanation is based on:

- User preferred category
- Similarity to liked movies
- Collaborative filtering score
- Neural Collaborative Filtering score
- Popularity or long-tail boost
"""
                    )
            else:
                st.warning("No recommendations available for explanation.")

        except requests.exceptions.ConnectionError:
            st.error("FastAPI is not running. Start it using: uvicorn app_fastapi:app --reload")
        except Exception as e:
            st.error(f"Error: {e}")


# =========================================================
# TAB 4: COLD-START DEMO
# =========================================================
with tab4:
    st.header("🧊 Cold-Start Demo")

    st.markdown(
        """
Cold-start happens when the system does not have enough user or item history.

This dashboard checks:

- Existing user → Hybrid recommendation
- Existing user without history → Category fallback
- Unknown user → Global popularity fallback
"""
    )

    cold_user_id = st.number_input(
        "Enter New / Unknown User ID",
        min_value=1,
        value=999999,
        step=1,
        key="cold_user_id"
    )

    if st.button("Run Cold-Start Demo", key="cold_btn"):
        try:
            with st.spinner("Running cold-start fallback..."):
                response_json = call_recommend_api(API_BASE_URL, int(cold_user_id), top_n)

            cold_df = recommendations_to_df(response_json)
            strategy = response_json.get("strategy", "-")

            st.success(f"Strategy Used: {strategy}")

            if strategy == "global_fallback":
                st.info("Unknown user detected. Showing globally popular and highly rated movies.")
            elif strategy == "cold_start_fallback":
                st.info("User exists but has no rating history. Showing preference/category-based recommendations.")
            else:
                st.info("User has enough history. Hybrid recommendation is used.")

            st.dataframe(cold_df, use_container_width=True)
            show_cards(cold_df)

        except requests.exceptions.ConnectionError:
            st.error("FastAPI is not running. Start it using: uvicorn app_fastapi:app --reload")
        except Exception as e:
            st.error(f"Error: {e}")


# =========================================================
# TAB 5: SYSTEM DESIGN
# =========================================================
with tab5:
    st.header("🏗️ System Design & Scalability")

    st.markdown(
        """
## 1. Batch Recommendation Flow

1. Collect user, movie, and rating data.
2. Create user-item matrix.
3. Create normalized user-item matrix.
4. Create implicit feedback matrix.
5. Train SVD model.
6. Create TF-IDF content similarity.
7. Train NCF deep learning model.
8. Save model artifacts.
9. FastAPI loads the saved models.
10. Streamlit displays the recommendations.

---

## 2. Real-Time Recommendation Flow

1. User enters user ID or movie ID in Streamlit.
2. Streamlit sends request to FastAPI.
3. FastAPI checks whether user/item exists.
4. If user has history, hybrid recommendation is used.
5. If user has no history, cold-start fallback is used.
6. FastAPI returns results with explanation.
7. Streamlit displays recommendations.

---

## 3. Retraining Strategy

- Retrain popularity statistics daily.
- Retrain SVD and similarity models weekly.
- Retrain NCF model weekly or when new data increases.
- Compare old and new models before deployment.
- Deploy only if new model improves Precision@K, Recall@K, MAP@K, or NDCG@K.

---

## 4. Data Growth Handling

- Store raw data separately from processed data.
- Save processed features as pickle or parquet files.
- Use sparse matrices for large user-item data.
- Use caching for frequent users.
- Use approximate nearest neighbor search for large movie catalogs.
- Use cloud storage for large model artifacts.

---

## 5. Monitoring KPIs

### Model KPIs

- Precision@K
- Recall@K
- MAP@K
- NDCG@K
- RMSE
- MAE

### Business KPIs

- Click-through rate
- Watch rate
- Completion rate
- User retention
- Conversion rate

### System KPIs

- API latency
- Error rate
- Request throughput
- Cold-start coverage
- Cache hit rate

### Fairness KPIs

- Diversity
- Novelty
- Long-tail exposure
- Popularity bias
"""
    )