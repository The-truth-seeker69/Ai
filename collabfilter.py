def collab():
    import pandas as pd
    from surprise import Reader, Dataset, SVDpp
    from surprise.model_selection import train_test_split, cross_validate
    from surprise import accuracy
    from collections import defaultdict
    import streamlit as st
    import warnings
    import ast
    import requests
    import itertools

    # TMDB API Key - not required for core model but kept for poster fetching
    API_KEY = "637c7ce4e3bf68f21a26e57b5226f8e3"
    BASE_URL = "https://api.themoviedb.org/3"

    def get_poster_url(title):
        """Fetch movie poster from TMDb API"""
        search_url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={title}"
        try:
            response = requests.get(search_url).json()
            if "results" in response and response["results"]:
                poster_path = response["results"][0].get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w200{poster_path}"
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching poster: {e}")
        return None # fallback if no poster found

    warnings.simplefilter('ignore')

    # =========================
    # Load & Clean Datasets
    # =========================
    @st.cache_data
    def load_and_preprocess_data():
        """Load, clean, and filter datasets."""
        ratings = pd.read_csv('dataset/ratings_small.csv')
        movies = pd.read_csv('dataset/movies_metadata.csv', low_memory=False)
        credits = pd.read_csv('dataset/credits.csv')

        movies = movies.drop([19730, 29503, 35587], errors="ignore")
        movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
        movies = movies.dropna(subset=['id'])
        movies['id'] = movies['id'].astype(int)

        credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
        credits = credits.dropna(subset=['id'])
        credits['id'] = credits['id'].astype(int)
        movies = movies.merge(credits, on='id')

        movies_filtered = movies[['id', 'title', 'release_date', 'genres', 'overview', 'poster_path', 'cast', 'crew']]

        ratings = ratings[ratings['movieId'].isin(movies_filtered['id'])]
        
        user_counts = ratings['userId'].value_counts()
        ratings = ratings[ratings['userId'].isin(user_counts[user_counts >= 5].index)]

        movie_counts = ratings['movieId'].value_counts()
        ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= 5].index)]
        
        return ratings, movies_filtered

    # Use Streamlit's caching to avoid reloading data every time
    ratings, movies_filtered = load_and_preprocess_data()

    # =========================
    # Precision & Recall
    # =========================
    def precision_recall_at_k(predictions, k=10, threshold=3):
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions, recalls, f1_scores = defaultdict(float), defaultdict(float), defaultdict(float)
        for uid, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                for (est, true_r) in user_ratings[:k])
            
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
            
            if (precisions[uid] + recalls[uid]) != 0:
                f1_scores[uid] = 2 * (precisions[uid] * recalls[uid]) / (precisions[uid] + recalls[uid])

        avg_precision = sum(p for p in precisions.values()) / len(precisions)
        avg_recall = sum(r for r in recalls.values()) / len(recalls)
        avg_f1 = sum(f for f in f1_scores.values()) / len(f1_scores) if f1_scores else 0
        return avg_precision, avg_recall, avg_f1

    # =========================
    # Train & Tune Collaborative Filtering Model (SVD++)
    # =========================
    @st.cache_resource
    def train_model(ratings_df):
        """Trains the SVDpp model with hyperparameter tuning."""
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
        trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
        
        param_grid = {
            'n_factors': [50, 100],
            'n_epochs': [20, 30],
            'lr_all': [0.005, 0.01],
            'reg_all': [0.02, 0.04]
        }

        # Manual Grid Search for progress feedback
        best_rmse = float('inf')
        best_params = {}
        
        total_combinations = len(list(itertools.product(*param_grid.values())))
        progress_bar = st.progress(0, text=f"Tuning model hyperparameters...")
        status_text = st.info("Starting grid search...")
        
        for i, (n_factors, n_epochs, lr_all, reg_all) in enumerate(itertools.product(*param_grid.values())):
            progress = (i + 1) / total_combinations
            progress_bar.progress(progress)
            status_text.info(f"Testing combination {i+1}/{total_combinations}: n_factors={n_factors}, n_epochs={n_epochs}, lr_all={lr_all}, reg_all={reg_all}")
            
            algo = SVDpp(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, verbose=False)
            
            try:
                # Added a random_state for reproducibility
                results = cross_validate(algo, data.build_full_trainset(), measures=['rmse'], cv=3, verbose=False, random_state=42)
                current_rmse = results['test_rmse'].mean()
                if current_rmse < best_rmse:
                    best_rmse = current_rmse
                    best_params = {
                        'n_factors': n_factors,
                        'n_epochs': n_epochs,
                        'lr_all': lr_all,
                        'reg_all': reg_all
                    }
            except Exception as e:
                status_text.warning(f"Failed to train with parameters {best_params}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Use the best parameters to train the final model on the full training data
        algo = SVDpp(**best_params)
        algo.fit(trainset)
        
        # Evaluate on the holdout test set
        predictions = algo.test(testset)

        rmse_score = accuracy.rmse(predictions, verbose=False)
        mae_score = accuracy.mae(predictions, verbose=False)
        precision, recall, f1_score = precision_recall_at_k(predictions)

        return algo, predictions, best_params, rmse_score, mae_score, precision, recall, f1_score

    # Train the model and get predictions
    algo, predictions, best_params, rmse_score, mae_score, precision, recall, f1_score = train_model(ratings)

    user_id = st.number_input("Login with User ID:", min_value=1, max_value=ratings['userId'].max(), value=1)

    # --- Genre Filter
    all_genres = sorted({g['name'] for movie_list in movies_filtered['genres'].dropna()
                        for g in ast.literal_eval(movie_list)})
    selected_genres = st.multiselect("Filter by Genres:", options=all_genres)

    # --- Sort Option
    sort_option = st.selectbox("Sort recommendations by:", options=["Predicted Rating (from highest)", "Predicted Rating (from lowest)"])

    st.write("---")

    if st.button("Get Recommendations"):
        st.subheader(f"Top 10 Recommendations for User {user_id}")

        seen_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()
        all_movie_ids = ratings['movieId'].unique()
        unseen_movies_pred = [algo.predict(user_id, iid) for iid in all_movie_ids if iid not in seen_movies]

        # Filter by genres
        if selected_genres:
            filtered_recs = []
            for rec in unseen_movies_pred:
                movie_info = movies_filtered[movies_filtered['id'] == rec.iid].iloc[0]
                try:
                    genres = [g['name'] for g in ast.literal_eval(movie_info['genres'])]
                except:
                    genres = []
                if any(g in genres for g in selected_genres):
                    filtered_recs.append(rec)
            unseen_movies_pred = filtered_recs

        # Sort recommendations
        reverse = True if sort_option == "Predicted Rating (from highest)" else False
        unseen_movies_pred.sort(key=lambda x: x.est, reverse=reverse)

        # Show top 10
        top_10_recs = unseen_movies_pred[:10]

        for rec in top_10_recs:
            movie_info = movies_filtered[movies_filtered['id'] == rec.iid].iloc[0]

            # --- Extract director
            try:
                crew_data = ast.literal_eval(movie_info['crew'])
                director = [c['name'] for c in crew_data if c['job'] == 'Director']
                director = director[0] if director else "Unknown"
            except:
                director = "Unknown"

            # --- Extract cast
            try:
                cast_data = ast.literal_eval(movie_info['cast'])
                cast_names = [c['name'] for c in cast_data[:3]]
            except:
                cast_names = []

            # --- Extract genres
            try:
                genres = [g['name'] for g in ast.literal_eval(movie_info['genres'])]
            except:
                genres = []

            # --- Extract release year
            try:
                year = pd.to_datetime(movie_info.get('release_date', ''), errors='coerce').year
            except:
                year = None

            # --- Title with Year instead of Director
            if year:
                st.markdown(f"### {movie_info['title']} ({year})")
            else:
                st.markdown(f"### {movie_info['title']}")

            # --- Poster
            poster_url = get_poster_url(movie_info['title'])
            if poster_url:
                st.image(poster_url, width=120)
            else:
                st.image("https://via.placeholder.com/120x180?text=No+Image", width=120)

            # --- Metadata
            st.markdown(f"**Genres:** {', '.join(genres)}")
            st.markdown(f"**Director:** {director}")
            st.markdown(f"**Predicted Rating:** {rec.est:.2f}")
            st.markdown(f"**Cast:** {', '.join(cast_names)}")
            st.markdown(f"{movie_info['overview'] if pd.notna(movie_info['overview']) else 'No overview available.'}")
            st.markdown("---")


    st.subheader("📊 Final Model Evaluation")
    st.write(f"RMSE: {rmse_score:.4f}")
    st.write(f"MAE: {mae_score:.4f}")
    st.write(f"Precision@10: {precision:.4f}")
    st.write(f"Recall@10: {recall:.4f}")
    st.write(f"F1 Score@10: {f1_score:.4f}")
