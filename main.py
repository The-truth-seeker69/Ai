import streamlit as st
from collabfilter import collab
from hybrid import hybrid
# === Main Menu ===
st.title("🎬 Movie Recommender System")

algorithm_choice = st.radio(
    "Select which recommendation algorithm you want to run:",
    ["Collaborative Filtering", "Content-Based Filtering", "Hybrid"],
    horizontal=True
)

# === Algorithm Loader ===
def get_algorithm(choice):
    if choice == "Collaborative Filtering":
        return collab()
    elif choice == "Content-Based Filtering":
        return SVD(n_factors=50, n_epochs=20, verbose=False)
    elif choice == "Hybrid":
        return hybrid()
    else:
        return collab()  # default fallback

# === Page Display ===
st.write(f"### Currently Using: **{algorithm_choice}**")
st.write("---")

# Train model with the selected algorithm
algo = get_algorithm(algorithm_choice)

