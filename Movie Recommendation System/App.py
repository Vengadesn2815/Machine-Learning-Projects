import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load dataset
movies_data = pd.read_csv('movies.csv')

# Preprocessing
def preprocess_data(df):
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        df[feature] = df[feature].fillna('')
    df['combined_features'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']
    return df

movies_data = preprocess_data(movies_data)

# Vectorization & Similarity Calculation
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])
similarity = cosine_similarity(feature_vectors)

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1521747116042-5c6bc5e2f8bf'); 
        background-size: cover;
        background-position: center;
        background-attachment: fixed; /* Ensures the background is fixed during scrolling */
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #333;
        color: white;
    }
    .stButton>button {
        background: linear-gradient(to right, #ff416c, #ff4b2b);
        color: white;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.1);
    }
    .movie-card {
        background: rgba(51, 51, 51, 0.8);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        transition: 0.3s;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }
    .movie-card:hover {
        transform: scale(1.05);
        background: rgba(68, 68, 68, 0.9);
        box-shadow: 0px 6px 15px rgba(255, 255, 255, 0.4);
    }
    h1 {
        white-space: nowrap;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
    }
    .footer a {
        color: #ff416c;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 Movie Recommendation System")
movie_name = st.text_input("Enter a movie name")

if st.button("Recommend"):
    if movie_name:
        list_of_all_titles = movies_data['title'].tolist()
        close_matches = difflib.get_close_matches(movie_name, list_of_all_titles)
        if close_matches:
            close_match = close_matches[0]
            index_of_movie = movies_data[movies_data['title'] == close_match].index[0]
            similarity_scores = list(enumerate(similarity[index_of_movie]))
            sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:15]
            recommended_movies = [movies_data.iloc[i[0]]['title'] for i in sorted_movies[:10]]

            st.success(f"Movies similar to {close_match}:")
            for movie in recommended_movies:
                st.markdown(f"<div class='movie-card'>🎥 {movie}</div>", unsafe_allow_html=True)
        else:
            st.error("Movie not found!")

# Footer with contact details
st.markdown(
    """
    <div class="footer">
        <p>Contact: <a href="https://www.linkedin.com/in/subash-vengadesan152815/" target="_blank">LinkedIn</a> | 
        Mobile: +91 6383404505 | City: Puducherry, India | 
        Email: <a href="mailto:svengadesan433@gmail.com">svengadesan433@gmail.com</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
