import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD , accuracy
from surprise.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import streamlit as st
import ast
fav_movie = st.text_input("What is your favourite Movie?")


# -----------------------------
# Load datasets
# -----------------------------
movies = pd.read_csv("movies_metadata.csv", low_memory=False)  # id, title, genres
ratings = pd.read_csv("ratings_small.csv")  # userId, movieId, rating

# Keep only first 10000 for memory reasons
movies = movies.head(10000)

# Convert TMDB 'id' to numeric movieId (for matching with ratings)
movies = movies.rename(columns={'id': 'movieId'})
movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')
movies = movies.dropna(subset=['movieId'])
movies['movieId'] = movies['movieId'].astype(int)

# Fill missing genres

def parse_genres(x):
    try:
        genre_list = ast.literal_eval(x) if isinstance(x, str) else []
        return [g["name"] for g in genre_list if isinstance(g, dict)]
    except Exception:
        return []

movies["genres_list"] = movies["genres"].apply(parse_genres)

# -----------------------------
# Content-Based Filtering (CBF)
# -----------------------------
movies['genres'] = movies['genres'].fillna('')  # avoid NaN
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

def get_cbf_score(liked_movie_id, candidate_ids):
    try:
        idx = movies[movies["movieId"] == liked_movie_id].index[0]
    except IndexError:
        return pd.Series(0, index=candidate_ids)  # movie not found
    cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    scores = pd.Series(cosine_sim, index=movies["movieId"])
    return scores.loc[candidate_ids]


# -----------------------------
# Collaborative Filtering (CF)
# -----------------------------
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2)

svd = SVD()
svd.fit(trainset)
predictions = svd.test(testset)

    def get_cf_score(user_id, candidate_ids):
        scores = {}
        for mid in candidate_ids:
            scores[mid] = svd.predict(user_id, mid).est
        return pd.Series(scores)


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

    if cbf.max() > cbf.min():
        cbf_norm = (cbf - cbf.min()) / (cbf.max() - cbf.min())
    else:
        cbf_norm = cbf
    if cf.max() > cf.min():
        cf_norm = (cf - cf.min()) / (cf.max() - cf.min())
    else:
        cf_norm = cf

    final_score = alpha * cbf_norm + (1 - alpha) * cf_norm
    top_movies = final_score.sort_values(ascending=False).head(top_n)

    recommendations = movies[movies["movieId"].isin(top_movies.index)][["movieId", "title", "genres"]]
    recommendations = recommendations.set_index("movieId").loc[top_movies.index]

    recommendations = recommendations.copy()
    recommendations["Rank"] = range(1, len(recommendations) + 1)

    return recommendations[["Rank", "title", "genres"]].reset_index(drop=True)



def precision_recall_at_k(predictions, k=5, threshold=4.0):
    """Compute precision and recall at top-k recommendations per user."""
    from collections import defaultdict
    import numpy as np

    # Map predictions to each user
    user_pred = defaultdict(list)
    user_true = defaultdict(list)
    
    for pred in predictions:
        user_pred[pred.uid].append((pred.iid, pred.est))
        if pred.r_ui >= threshold:
            user_true[pred.uid].append(pred.iid)

    precisions = []
    recalls = []

    for uid, user_ratings in user_pred.items():
        # Sort by predicted rating
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_k = [iid for iid, _ in user_ratings[:k]]

        # True positives
        hits = len(set(top_k) & set(user_true[uid]))
        precisions.append(hits / k if k > 0 else 0)
        recalls.append(hits / len(user_true[uid]) if user_true[uid] else 0)

    return np.mean(precisions), np.mean(recalls)

precision, recall = precision_recall_at_k(predictions, k=5)
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"Precision@5: {precision:.4f}, Recall@5: {recall:.4f}, F1@5: {f1:.4f}")

results = hybrid_recommend(user_id=1, liked_title=fav_movie, top_n=10, alpha=0.6)


rmse = accuracy.rmse(predictions)
mse = accuracy.mse(predictions)



print(f"RMSE: {rmse:.4f}, MSE: {mse:.4f}")
precision, recall = precision_recall_at_k(predictions, k=5)
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"Precision@10: {precision:.4f}, Recall@5: {recall:.4f}, F1@5: {f1:.4f}")

# print(results.to_string(index=False))

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
            st.write("---")  # Optional: add a separator between movies


    # Evaluation metrics
    rmse = accuracy.rmse(predictions, verbose=False)
    mse = accuracy.mse(predictions, verbose=False)
    precision, recall = precision_recall_at_k(predictions, k=5)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    st.subheader("📊 Evaluation Metrics")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**MSE:** {mse:.4f}")
    st.write(f"**Precision@5:** {precision:.4f}")
    st.write(f"**Recall@5:** {recall:.4f}")
    st.write(f"**F1@5:** {f1:.4f}")



# Evaluation Metrics

