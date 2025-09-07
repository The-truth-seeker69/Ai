def contentbased():
    import pandas as pd
    import numpy as np
    from ast import literal_eval
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
    from nltk.stem.snowball import SnowballStemmer
    import streamlit as st
    import requests
    from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
    import warnings

    warnings.simplefilter('ignore')

    # ===============================
    # Load and process data (cached)
    # ===============================
    @st.cache_data
    def load_and_process():
        # Load datasets
        md = pd.read_csv('dataset/movies_metadata.csv')
        links_small = pd.read_csv('dataset/links_small.csv')
        credits = pd.read_csv('dataset/credits.csv')
        keywords = pd.read_csv('dataset/keywords.csv')

        # Clean links_small
        links_small = links_small[links_small['tmdbId'].notnull()]
        links_small['tmdbId'] = links_small['tmdbId'].astype(int)
        links_small['imdbId'] = links_small['imdbId'].astype(str).str.zfill(7)

        md = md.drop([19730, 29503, 35587])  # drop bad rows

        # Process genres
        md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else []
        )

        # Merge credits and keywords
        md['id'] = md['id'].astype(int)
        credits['id'] = credits['id'].astype(int)
        keywords['id'] = keywords['id'].astype(int)
        md = md.merge(credits, on='id')
        md = md.merge(keywords, on='id')

        # Subset and merge imdbId
        smd = md[md['id'].isin(links_small['tmdbId'])]
        smd = smd.merge(links_small[['tmdbId', 'imdbId']], left_on='id', right_on='tmdbId', how='left')

        smd['tagline'] = smd['tagline'].fillna('')
        smd['description'] = smd['overview'] + ' ' + smd['tagline']
        smd['description'] = smd['description'].fillna('')

        # TF-IDF on description
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
        tfidf_matrix = tf.fit_transform(smd['description'])
        cosine_sim_desc = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Process cast, crew, keywords
        smd['cast'] = smd['cast'].apply(literal_eval)
        smd['crew'] = smd['crew'].apply(literal_eval)
        smd['keywords'] = smd['keywords'].apply(literal_eval)

        def get_director(x):
            for i in x:
                if i['job'] == 'Director':
                    return i['name']
            return np.nan

        smd['director_display'] = smd['crew'].apply(get_director)
        smd['cast_display'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        smd['cast_display'] = smd['cast_display'].apply(lambda x: x[:3] if len(x) >= 3 else x)

        smd['cast'] = smd['cast_display'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
        smd['director'] = smd['director_display'].astype(str).apply(lambda x: str.lower(x.replace(" ", "")))
        smd['director'] = smd['director'].apply(lambda x: [x])  # boost importance

        smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

        # s = smd.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
        # s.name = 'keyword'
        # s = s.value_counts()
        # s = s[s > 1]

        # stemmer = SnowballStemmer('english')

        # def filter_keywords(x):
        #     return [stemmer.stem(i) for i in x if i in s]

        # smd['keywords'] = smd['keywords'].apply(filter_keywords)
        smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

        # Build the "soup"
        smd['soup_keywords'] = smd['keywords'].apply(lambda x: ' '.join(x))
        smd['soup_castdir'] = smd['cast'].apply(lambda x: ' '.join(x)) + ' ' + smd['director'].apply(lambda x: ' '.join(x))
        smd['soup_genres'] = smd['genres'].apply(lambda x: ' '.join([g.lower().replace(" ", "") for g in x]))

        count_kw = CountVectorizer(stop_words='english')
        count_cd = CountVectorizer(stop_words='english')
        count_gn = CountVectorizer(stop_words='english')

        cosine_sim_keywords = cosine_similarity(count_kw.fit_transform(smd['soup_keywords']))
        cosine_sim_castdir = cosine_similarity(count_cd.fit_transform(smd['soup_castdir']))
        cosine_sim_genres = cosine_similarity(count_gn.fit_transform(smd['soup_genres']))

        # --------------------------
        # Recency & Popularity Scores
        # --------------------------
        current_year = 2025
        smd['release_year'] = pd.to_datetime(smd['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
        smd['recency_score'] = 1 / (1 + (current_year - smd['release_year']))
        smd['popularity_score'] = smd['vote_average'] * np.log1p(smd['vote_count'])

        # normalize
        smd['recency_score_norm'] = (smd['recency_score'] - smd['recency_score'].min()) / (smd['recency_score'].max() - smd['recency_score'].min())
        smd['popularity_score_norm'] = (smd['popularity_score'] - smd['popularity_score'].min()) / (smd['popularity_score'].max() - smd['popularity_score'].min())

        # Weighted rating (IMDb-style)
        smd['vote_count'] = smd['vote_count'].fillna(0).astype(int)
        smd['vote_average'] = smd['vote_average'].fillna(0).astype(float)
        m = smd['vote_count'].quantile(0.6)
        C = smd['vote_average'].mean()
        smd['wr'] = smd.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(x['vote_count']+m)*C), axis=1)
        smd['wr_norm'] = (smd['wr'] - smd['wr'].min()) / (smd['wr'].max() - smd['wr'].min())

        # --------------------------
        # Optimized similarity weights
        # --------------------------
        cosine_sim_hybrid = (
            0.20 * cosine_sim_keywords +
            0.20 * cosine_sim_castdir +
            0.20 * cosine_sim_genres +
            0.40 * cosine_sim_desc
        )

        # Combine similarity + weighted rating + recency + popularity
        final_score = (
            0.50 * cosine_sim_hybrid +
            0.20 * smd['wr_norm'].values[:, None] +
            0.20 * smd['recency_score_norm'].values[:, None] +
            0.10 * smd['popularity_score_norm'].values[:, None]
        )

        smd = smd.reset_index(drop=True)
        indices = pd.Series(smd.index, index=smd['title'])

        return smd, final_score, indices

    # ===============================
    # Recommendation function
    # ===============================
    def get_recommendations(title, smd, cosine_sim_hybrid, indices, top_n=10):
        if title not in indices:
            matches = indices.index[indices.index.str.lower() == title.lower()]
            if len(matches) == 0:
                raise KeyError(f"Movie '{title}' not found.")
            title = matches[0]

        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim_hybrid[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # sim_scores = sim_scores[1:26]  # exclude the movie itself
        sim_scores = sim_scores[1:top_n+1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        # movies = smd.iloc[movie_indices].copy()

        # Compute weighted rating (IMDb style)
        # vote_counts = movies['vote_count'].fillna(0).astype(int)
        # vote_averages = movies['vote_average'].fillna(0).astype(float)
        # C = vote_averages.mean()
        # m = vote_counts.quantile(0.60)

        # qualified = movies[(movies['vote_count'] >= m) & (movies['vote_average'] > 0)].copy()
        # qualified['vote_count'] = qualified['vote_count'].astype(int)
        # qualified['vote_average'] = qualified['vote_average'].astype(float)

        # def weighted_rating(x):
        #     v = x['vote_count']
        #     R = x['vote_average']
        #     return (v / (v + m) * R) + (m / (v + m) * C)

        # qualified['wr'] = qualified.apply(weighted_rating, axis=1)

        # Normalize weighted rating to 0-1
        # wr_min, wr_max = qualified['wr'].min(), qualified['wr'].max()
        # qualified['wr_norm'] = (qualified['wr'] - wr_min) / (wr_max - wr_min)

        # Map cosine similarity to qualified movies
        # qualified['score'] = [scores[movie_indices.index(i)] for i in qualified.index]

        # Optional: combine scores for final ranking
        # qualified['final_score'] = (qualified['score'] * 0.7) + (qualified['wr_norm'] * 0.3)

        # Sort by final score
        # qualified = qualified.sort_values('final_score', ascending=False).head(10)

        # def format4(x):
        #     return f"{x:.4f}"

        # return pd.DataFrame({
        #     "title": qualified['title'],
        #     "genres": qualified['genres'],
        #     "description": qualified['description'],
        #     "keywords": qualified['keywords'],
        #     "cast_display": qualified['cast_display'],
        #     "director_display": qualified['director_display'],
        #     "imdb_id": qualified['imdbId'],
        #     "poster_path": qualified['poster_path'],
        #     "score": qualified['score'].apply(format4),
        #     "weighted_rating": qualified['wr'].apply(format4),
        #     "final_score": qualified['final_score'].apply(format4)
        # })

        return pd.DataFrame({ 
            "title": smd['title'].iloc[movie_indices],
            "genres": smd['genres'].iloc[movie_indices],
            "description": smd['description'].iloc[movie_indices],
            "keywords": smd['keywords'].iloc[movie_indices],
            "cast_display": smd['cast_display'].iloc[movie_indices],
            "director_display": smd['director_display'].iloc[movie_indices],
            "imdb_id": smd['imdbId'].iloc[movie_indices],
            "poster_path": smd['poster_path'].iloc[movie_indices],
            "score": [f"{s:.4f}" for s in scores],
            "weighted_rating": smd['wr'].iloc[movie_indices].apply(lambda x: f"{x:.2f}")
        })

    # ===============================
    # Evaluation function
    # ===============================
    def evaluate_recommendations(title, recs, smd, indices):
        idx = indices[title]

        # Reference movie features
        true_genres = set(smd.loc[idx, 'genres'])
        true_director = smd.loc[idx, 'director_display']
        true_cast = set(smd.loc[idx, 'cast_display'])
        true_keywords = set(smd.loc[idx, 'keywords'])

        # Compute similarity scores for all movies
        sim_scores = list(enumerate(cosine_sim_hybrid[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Exclude the input movie itself
        sim_scores = [s for s in sim_scores if s[0] != idx]

        y_true_all = []   # for recall@all
        y_pred_all = []   # for recall@all
        y_true_top10 = [] # for precision@10
        y_pred_top10 = [] # for precision@10

        for rank, (movie_idx, sim_score) in enumerate(sim_scores):
            row = smd.iloc[movie_idx]

            rec_genres = set(row['genres'])
            rec_director = row['director_display']
            rec_cast = set(row['cast_display']) if isinstance(row['cast_display'], list) else set()
            rec_keywords = set(row['keywords'])

            # Weighted relevance: count 2+ features overlap
            overlap_count = 0
            if len(true_keywords & rec_keywords) > 1:
                overlap_count += 1
            if rec_director == true_director or len(true_cast & rec_cast) > 0:
                overlap_count += 1
            if len(true_genres & rec_genres) > 1:
                overlap_count += 1

            is_relevant = int(overlap_count >= 2)

            # For recall@all
            y_true_all.append(is_relevant)
            y_pred_all.append(int(rank < 10))  # only top 10 are recommended

            # For precision@10
            if rank < 10:
                y_true_top10.append(is_relevant)
                y_pred_top10.append(1)  # all in top 10 are recommended

        # Metrics
        precision_top10 = precision_score(y_true_top10, y_pred_top10, zero_division=0)
        recall_all = recall_score(y_true_all, y_pred_all, zero_division=0)
        f1_score_all = (2 * precision_top10 * recall_all) / (precision_top10 + recall_all + 1e-8)

        return precision_top10, recall_all, f1_score_all

    # ===============================
    # OMDb poster fetcher
    # ===============================
    API_KEY = "98035cce"
    poster_cache = {}  # Dictionary to store fetched posters

    def fetch_movie_poster(imdb_id):
        if imdb_id in poster_cache:
            return poster_cache[imdb_id]  # Return cached poster

        url = f"http://www.omdbapi.com/?i=tt{imdb_id}&apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        poster_url = "https://placehold.co/300x450?text=No+Poster"
        
        if data.get("Response") == "True" and data.get("Poster"):
            poster_url = data.get("Poster")

        poster_cache[imdb_id] = poster_url  # Cache it
        return poster_url

    # ===============================
    # Streamlit app
    # ===============================
    st.title("🎬 Movie Recommender System")
    st.write("Enter a movie title below to get recommendations!")

    # Load data (cached)
    smd, cosine_sim_hybrid, indices = load_and_process()

    # User input
    movie_name = st.text_input("Movie title:")

    if movie_name:
        try:
            # Get the input movie itself as a "recommendation"
            input_idx = indices[movie_name] if movie_name in indices else indices[indices.index.str.lower() == movie_name.lower()][0]
            input_movie = smd.iloc[[input_idx]]
            input_poster = fetch_movie_poster(input_movie['imdbId'].values[0]) if pd.notna(input_movie['imdbId'].values[0]) else "https://placehold.co/300x450?text=No+Poster"
            
            # Display the input movie first
            year_display = ""
            if 'release_date' in smd.columns and pd.notna(input_movie['release_date'].values[0]):
                year_display = input_movie['release_date'].values[0][:4]

            title_display_input = f"{input_movie['title'].values[0]} ({year_display})" if year_display else input_movie['title'].values[0]

            st.subheader(f"Input Movie: {title_display_input}")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(input_poster, width='stretch')
            with col2:
                st.write(f"**Genres:** {', '.join(input_movie['genres'].values[0])}")
                st.write(f"**Director:** {input_movie['director_display'].values[0]}")
                st.write(f"**Cast:** {', '.join(input_movie['cast_display'].values[0])}")
                st.caption(input_movie['description'].values[0])
            st.markdown("---")

            # Now get top 10 recommendations
            results = get_recommendations(movie_name, smd, cosine_sim_hybrid, indices).head(10)
            st.success(f"Top 10 recommendations for **{movie_name}**:")

            # Display recommendations as before
            for _, row in results.iterrows():
                poster_url = fetch_movie_poster(row['imdb_id']) if pd.notna(row['imdb_id']) else "https://placehold.co/300x450?text=No+Poster"
                year_display = ""
                if 'release_date' in smd.columns and pd.notna(smd.loc[smd['title'] == row['title'], 'release_date'].values[0]):
                    year_display = smd.loc[smd['title'] == row['title'], 'release_date'].values[0][:4]
                title_display = f"{row['title']} ({year_display})" if year_display else row['title']
                genres_display = ', '.join(row['genres'])
                director_display = row['director_display'] if pd.notna(row['director_display']) else "Unknown"
                cast_display = ', '.join(row['cast_display']) if isinstance(row['cast_display'], list) else ""
                plot_display = row['description']

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(poster_url, width='stretch')
                with col2:
                    st.subheader(title_display)
                    st.write(f"**Genres:** {genres_display}")
                    st.write(f"**Director:** {director_display}")
                    st.write(f"**Cast:** {cast_display}")
                    st.write(f"**Score:** {row['score']}")
                    st.write(f"**Weighted Rating:** {row['weighted_rating']}/10")
                    # st.write(f"**Final Score:** {row['final_score']}")
                    st.caption(plot_display)
                st.markdown("---")

            # Evaluate recommendations
            precision, recall, f1 = evaluate_recommendations(movie_name, results, smd, indices)
            st.write(f"**Evaluation Metrics:**")
            st.write(f"- Precision@10: {precision:.4f}")
            st.write(f"- Recall@all: {recall:.4f}")
            st.write(f"- F1 Score (precision@10, recall@all): {f1:.4f}")
            #i can't do precision@all, im doing for top 10, i can't do recall@10, false negative is 0?
        except KeyError:
            st.error("❌ Movie not found in database. Please try another.")
