import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import streamlit as st
import ast

# -----------------------------
# User input
# -----------------------------
fav_movie = st.text_input("What is your favourite Movie?")

# -----------------------------
# Load datasets
# -----------------------------
movies = pd.read_csv("movies_metadata.csv", low_memory=False)
ratings = pd.read_csv("ratings_small.csv")

movies = movies.head(10000)
movies = movies.rename(columns={'id': 'movieId'})
movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')
movies = movies.dropna(subset=['movieId'])
movies['movieId'] = movies['movieId'].astype(int)

# -----------------------------
# Content-Based Filtering (CBF)
# -----------------------------
movies['genres'] = movies['genres'].fillna('')
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

def get_cbf_score(liked_movie_id, candidate_ids):
    cosine_sim = (tfidf_matrix * tfidf_matrix[liked_movie_id].T).toarray().ravel()
    return pd.Series(cosine_sim[candidate_ids], index=candidate_ids)

# -----------------------------
# Collaborative Filtering (CF) with TruncatedSVD
# -----------------------------
user_item_matrix = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
svd = TruncatedSVD(n_components=20, random_state=42)
latent_matrix = svd.fit_transform(user_item_matrix)
pred_matrix = np.dot(latent_matrix, svd.components_)
pred_df = pd.DataFrame(pred_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)

def get_cf_score(user_id, candidate_ids):
    if user_id not in pred_df.index:
        return pd.Series(0, index=candidate_ids)
    return pred_df.loc[user_id, candidate_ids]

# -----------------------------
# Hybrid Recommender
# -----------------------------
def hybrid_recommend(user_id, liked_title, top_n=10, alpha=0.5):
    try:
        liked_movie_id = movies[movies['title'].str.contains(liked_title, case=False, na=False)].iloc[0]['movieId']
    except IndexError:
        return pd.DataFrame()

    candidate_ids = movies["movieId"].tolist()
    if liked_movie_id in candidate_ids:
        candidate_ids.remove(liked_movie_id)

    cbf = get_cbf_score(liked_movie_id, candidate_ids)
    cf = get_cf_score(user_id, candidate_ids)

    # normalize
    cbf_norm = (cbf - cbf.min()) / (cbf.max() - cbf.min()) if cbf.max() > cbf.min() else cbf
    cf_norm = (cf - cf.min()) / (cf.max() - cf.min()) if cf.max() > cf.min() else cf

    final_score = alpha * cbf_norm + (1 - alpha) * cf_norm
    top_movies = final_score.sort_values(ascending=False).head(top_n)

    recommendations = movies[movies["movieId"].isin(top_movies.index)][["movieId", "title", "genres"]]
    recommendations = recommendations.set_index("movieId").loc[top_movies.index]
    recommendations = recommendations.copy()
    recommendations["Rank"] = range(1, len(recommendations) + 1)

    return recommendations[["Rank", "title", "genres"]].reset_index(drop=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Hybrid Movie Recommender")

if fav_movie:
    results = hybrid_recommend(user_id=1, liked_title=fav_movie, top_n=10, alpha=0.6)

    if results.empty:
        st.warning(f"Sorry, no match found for '{fav_movie}'. Try another movie.")
    else:
        st.subheader("🎬 Top 10 Recommended Movies")
        for i, row in results.iterrows():
            rank = row["Rank"]
            title = row["title"]
            genres = row["genres"]
            try:
                genre_list = ast.literal_eval(genres) if isinstance(genres, str) else genres
                genre_names = [g["name"] for g in genre_list if isinstance(g, dict)]
                genre_str = ", ".join(genre_names)
            except Exception:
                genre_str = "N/A"

            st.markdown(f"**{rank}. {title}**  \n*Genres:* {genre_str}")
            st.write("---")
