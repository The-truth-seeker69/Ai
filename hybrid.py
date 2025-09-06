def hybrid():
    import pandas as pd
    import numpy as np
    import ast
    import requests
    import streamlit as st
    from collections import defaultdict
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    # ast convert [{'id': 28, 'name': 'Action'}]" to python lists

    from surprise import Dataset, Reader, SVD, accuracy
    from surprise.model_selection import train_test_split, GridSearchCV


    current_year = 2025

    # -----------------------------
    # Streamlit Input
    # -----------------------------
    st.title("🎬 Movie Recommender System")


    user_id = st.text_input("Login with user ID")

    fav_movie = st.text_input("Movie Title")


    # -----------------------------
    # Load Datasets
    # -----------------------------
    movies = pd.read_csv("dataset/movies_metadata.csv", low_memory=False)
    # Keep only first 10000 for memory reasons

    movies = movies.head(400000)

    # Rename 'id' column to 'movieId' and convert to int
    movies = movies.rename(columns={'id': 'movieId'})
    # Convert id to int

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

    # Finds the index of the liked movie.
    # Computes cosine similarity between that movie’s TF-IDF vector and all others.
    # Returns similarity scores for only the candidate movies.
    # Example: If user likes "Inception", it will compute similarity with all movies.
    # candidates actually means the candidate movies that user might like cause it finds similiarity on other candidates such as genres



    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    movies["genres_str2"] = movies["genres_list"].apply(lambda g: " ".join(g))

    movies['features'] = (
        movies['overview'].fillna('') + " " +
        movies["genres_str2"].fillna('') + " " +
        movies['production_companies'].fillna('')+ " " +
        movies['production_countries'].fillna('')+ " " +
        movies['spoken_languages'].fillna('')
    )


    tfidf_matrix = tfidf.fit_transform(movies['features'])

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



    # Load ratings dataset
    ratings = pd.read_csv("dataset/ratings_small.csv")  # userId, movieId, rating
    # tell that the rating is 0 -5

    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

    # n_factors : number of hidden features
    # n_epoches : how many times algorithm loop over the data
    # lr_all : learning rate (how fast the models learn)
    # reg_all : regularization (how strongly to avoid overfitting).
    param_grid = {
        'n_factors': [50, 150],
        'n_epochs': [20, 30],
        'lr_all': [0.002, 0.005],
        'reg_all': [0.02, 0.05]
    }
    grid = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
    grid.fit(data)
    # at the end it test every possible combination of these ingredients (those paramgrid) to see which model performs the best

    best_params = grid.best_params['rmse']

    # 80 train and 20 test
    trainset, testset = train_test_split(data, test_size=0.20)


    # Train an SVD (Singular Value Decomposition) recommender.
    # Learns hidden factors (e.g. taste dimensions like “likes sci-fi”, “likes romance”).
    # Produces predicted ratings for testset.
    svd = SVD(**best_params)
    svd.fit(trainset)
    predictions = svd.test(testset)

    # For each candidate movie, predict how much user_id would rate it.
    # Returns scores (predicted ratings).
    # output should be something like {10 : 4.3 and more} which predict how much will use love the movie

    def get_cf_score(user_id, candidate_ids):
        scores = {}
        for mid in candidate_ids:
            scores[mid] = svd.predict(user_id, mid).est
        return pd.Series(scores)


    # -----------------------------
    # Hybrid Recommender
    # -----------------------------'
    # if alpha 0.6 meaning that 60% cbf and 40% cf

    def hybrid_recommend(user_id, liked_title, movies ,top_n=10, alpha=0.6):

        try:
            # get liked_movie_id from the title , from the input defined in front
            liked_movie_id = movies[movies['title'].str.contains(liked_title, case=False, na=False)].iloc[0]['movieId']
        except IndexError:
            return pd.DataFrame()
        # Define candidate movie IDs (exclude the movie user already liked one).
        
        candidate_ids = movies["movieId"].tolist()
        if liked_movie_id in candidate_ids:
            candidate_ids.remove(liked_movie_id)

    # Filter candidates: keep only those with >=2 matching genres
        liked_genres = movies.loc[movies["movieId"] == liked_movie_id, "genres_list"].values[0]
        def count_shared_genres(genres_a, genres_b):
                return len(set(genres_a) & set(genres_b))
        candidate_ids = [
            mid for mid in candidate_ids
        if count_shared_genres(liked_genres, movies.loc[movies["movieId"] == mid, "genres_list"].values[0]) >= 2
            ]
        
        cbf = get_cbf_score(liked_movie_id, candidate_ids)
        # if user id is not the rating fall back to content based only
        if int(user_id) not in ratings['userId'].unique():
        # Cold-start → no CF info
            cf = pd.Series(0, index=candidate_ids)
        else:
            cf = get_cf_score(int(user_id), candidate_ids)
            
        # Ensures scores are between 0–1 so they’re comparable.
        cbf_norm = (cbf - cbf.min()) / (cbf.max() - cbf.min()) if cbf.max() > cbf.min() else cbf
        cf_norm = (cf - cf.min()) / (cf.max() - cf.min()) if cf.max() > cf.min() else cf
        # rewarding the latest year 
        current_year = 2020
        movies['release_year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
        movies['recency_score'] = 1 / (1 + (current_year - movies['release_year'].fillna(current_year)))
        #calculate popularitry based on the the vote avergae and vote count 

        movies['popularity_score'] = movies['vote_average'] * np.log1p(movies['vote_count'])

        # loc : DataFrame.loc[row_labels, column_labels]
        # for all movies whose movieId is in candidate_ids, return their recency_score column".
        movies = movies.set_index("movieId", drop=False)

        recency_scores    = movies.loc[candidate_ids, "recency_score"]
        popularity_scores = movies.loc[candidate_ids, "popularity_score"].rank(pct=True)



            # 0.5 (alpha) * cbf + (1-0.5) & cf

        final_score = alpha * cbf_norm + (1 - alpha) * cf_norm + 0.1 * popularity_scores + 0.1 * recency_scores


        top_movies = final_score.sort_values(ascending=False).head(top_n)
        recommendations = movies[movies["movieId"].isin(top_movies.index)][[
            "movieId", "title", "genres", "imdb_id", "release_date",
            "runtime", "overview", "poster_path", "vote_average", "vote_count"
        ]]
        recommendations = recommendations.set_index("movieId").loc[top_movies.index]
        recommendations = recommendations.copy()
        recommendations["Rank"] = range(1, len(recommendations) + 1)

        return recommendations[[
            "Rank", "title", "genres", "imdb_id", "release_date",
            "runtime", "overview", "poster_path", "vote_average", "vote_count"
        ]].reset_index(drop=True)

    # -----------------------------
    # Evaluation Metrics
    # -----------------------------



    def evaluate_recommendation_quality(predictions, top_n=30, threshold=3.0):
        """Evaluate recommendation quality using multiple metrics"""
        
        # 1. Traditional accuracy metrics
        rmse = accuracy.rmse(predictions, verbose=False)
        mse = accuracy.mse(predictions, verbose=False)  
        
        # 2. Precision, Recall, and F1 at K
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))
        
        precisions = dict()
        recalls = dict()
        f1_scores = dict()
        
        for uid, user_ratings in user_est_true.items():
            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            
            # Number of relevant items (true rating >= threshold)
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            
            # Number of recommended items in top n
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:top_n])
            
            # Number of relevant and recommended items in top n
            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:top_n]
            )
            
            # Precision@K: Proportion of recommended items that are relevant
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
            
            # Recall@K: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
            
            # F1-score: Harmonic mean of precision and recall
            if precisions[uid] + recalls[uid] > 0:
                f1_scores[uid] = 2 * (precisions[uid] * recalls[uid]) / (precisions[uid] + recalls[uid])
            else:
                f1_scores[uid] = 0
        
        # Calculate averages
        avg_precision = np.mean(list(precisions.values()))
        avg_recall = np.mean(list(recalls.values()))
        avg_f1 = np.mean(list(f1_scores.values()))
        
        # 3. Coverage: Percentage of items that can be recommended
        all_items = set(ratings['movieId'].unique())
        recommended_items = set()
        for uid, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            recommended_items.update([movie_id for (_, movie_id) in user_ratings[:top_n]])
        
        coverage = len(recommended_items) / len(all_items) if all_items else 0
        
        return {
            'rmse': rmse,
            'mse': mse,  
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'coverage': coverage
        }

    # Calculate evaluation metrics

    # -----------------------------
    # Streamlit UI
    # -----------------------------

    API_KEY = "7706a7"
    poster_cache = {}
    # fetching movie poster
    def fetch_movie_poster(imdb_id):
        if imdb_id in poster_cache:
            return poster_cache[imdb_id]

        url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        poster_url = "https://placehold.co/300x450?text=No+Poster"
        if data.get("Response") == "True" and data.get("Poster"):
            poster_url = data.get("Poster")
        poster_cache[imdb_id] = poster_url
        return poster_url

    if fav_movie:
        results = hybrid_recommend(user_id=user_id, liked_title=fav_movie,movies=movies,top_n=10, alpha=0.6)
        if results.empty:
            st.warning(f"Sorry, no match found for '{fav_movie}'. Try another movie.")
        else:
            metrics = evaluate_recommendation_quality(predictions)

            st.subheader("🎬 Top 10 Recommended Movies")
            for i, row in results.iterrows():
                genres = row["genres"]
                try:
                    genre_list = ast.literal_eval(genres) if isinstance(genres, str) else genres
                    genre_names = [g["name"] for g in genre_list if isinstance(g, dict)]
                    genre_str = ", ".join(genre_names)
                except Exception:
                    genre_str = "N/A"

                st.markdown(f"### {row['title']} ({row['release_date'][:4] if pd.notna(row['release_date']) else 'N/A'})")
                st.markdown(f"**Genres:** {genre_str}")
                st.write(f"**Runtime:** {row['runtime']:.0f} mins")
                st.write(f"**Rating:** ⭐ {row['vote_average']:.0f}/10 (based on {row['vote_count']:.0f} votes)")
                poster_url = fetch_movie_poster(row.get("imdb_id"))
                if poster_url:
                    st.image(poster_url, width=200)
                st.markdown("---")

            show_metrics = st.checkbox("Show Model Evaluation Metrics", value=False)
            if show_metrics:
                st.subheader("📊 Evaluation Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                    st.metric("MSE", f"{metrics['mse']:.4f}")
                with col2:
                    
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                    # “Of all the good movies that exist for you, how many did I actually recommend?
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                    st.metric("F1 Score", f"{metrics['f1']:.4f}")




