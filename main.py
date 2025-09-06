import streamlit as st
from collabfilter import collab

# === Main Menu ===
st.title("🎬 Movie Recommender System")

algorithm_choice = st.radio(
    "Select which recommendation algorithm you want to run:",
    ["SVD++ (Collaborative Filtering)", "SVD (Matrix Factorization)", "KNN (User-Based)"],
    horizontal=True
)

# === Algorithm Loader ===
def get_algorithm(choice):
    if choice == "Collaborative Filtering":
        return collab()
    elif choice == "SVD (Matrix Factorization)":
        return SVD(n_factors=50, n_epochs=20, verbose=False)
    elif choice == "KNN (User-Based)":
        return KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    else:
        return collab()  # default fallback

# === Page Display ===
st.write(f"### Currently Using: **{algorithm_choice}**")
st.write("---")

# Train model with the selected algorithm
algo = get_algorithm(algorithm_choice)

# (Here you’d plug it into your train/test split pipeline and display recommendations)
