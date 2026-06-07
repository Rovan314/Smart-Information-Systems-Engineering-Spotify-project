import streamlit as st
import pandas as pd

from recommender import SpotifyRecommender


st.set_page_config(
    page_title="Spotify Metadata Recommender",
    page_icon="🎧",
    layout="wide"
)


st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(180deg, #0b0b0b 0%, #121212 100%);
        color: #FFFFFF;
    }

    h1, h2, h3 {
        color: #1DB954;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #191414 0%, #0f0f0f 100%);
        border-right: 1px solid #1DB954;
        box-shadow: 0px 0px 20px rgba(29, 185, 84, 0.08);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1DB954;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #FFFFFF !important;
    }

    * {
        accent-color: #1DB954 !important;
    }

    .stSlider > div > div > div > div {
        background-color: #1DB954;
    }

    div[data-testid="stSlider"] span {
        background-color: #1DB954 !important;
        border-color: #1DB954 !important;
    }

    span[data-baseweb="tag"] {
        background-color: #1DB954 !important;
        color: black !important;
        border-radius: 20px !important;
        border: none !important;
    }

    span[data-baseweb="tag"] span {
        color: black !important;
    }

    div[data-baseweb="select"] {
        background-color: #191414;
        border-radius: 12px;
    }

    div[data-baseweb="select"] > div {
        background-color: #191414 !important;
        border-color: #1DB954 !important;
    }

    li[aria-selected="true"] {
        background-color: #1DB954 !important;
        color: black !important;
    }

    li:hover {
        background-color: rgba(29, 185, 84, 0.2) !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: black;
        border-radius: 25px;
        border: none;
        padding: 0.6rem 1.4rem;
        font-weight: bold;
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        transform: scale(1.03);
        background: #1ed760;
        color: black;
    }

    div[data-testid="stMetric"] {
        background: rgba(25, 20, 20, 0.85);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #1DB954;
        box-shadow: 0px 0px 10px rgba(29, 185, 84, 0.15);
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border: 1px solid rgba(29, 185, 84, 0.3);
    }

    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_recommender():
    recommender = SpotifyRecommender(
        modern_path="data/spotify_data clean.csv",
        classic_path="data/track_data_final.csv"
    )

    recommender.load_and_prepare_data()
    recommender.build_feature_matrix()

    return recommender


recommender = load_recommender()
df = recommender.df


st.title("🎧 Spotify Metadata Recommender")

st.caption(
    "A metadata-based recommendation system using genre analysis, "
    "Jaccard similarity, Truncated SVD, and cosine similarity."
)

st.write(
    """
    This project recommends songs using Spotify metadata such as genres,
    popularity, artist reach, release year, album type, and track duration.
    The system is designed to work without user listening history,
    helping address the cold-start recommendation problem.
    """
)

st.markdown("---")


st.sidebar.header("Recommendation Settings")

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

source_filter = st.sidebar.multiselect(
    "Dataset source",
    options=sorted(df["dataset_source"].unique()),
    default=sorted(df["dataset_source"].unique())
)

filtered_df = df[df["dataset_source"].isin(source_filter)]


selected_song = st.selectbox(
    "Choose a song:",
    options=sorted(filtered_df["song_label"].unique())
)

selected_row = df[df["song_label"] == selected_song].iloc[0]


st.subheader("Selected Track")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Track Popularity", int(selected_row["track_popularity"]))

with col2:
    st.metric("Artist Popularity", int(selected_row["artist_popularity"]))

with col3:
    st.metric("Release Year", int(selected_row["release_year"]))

st.write(f"**Artist:** {selected_row['artist_name']}")
st.write(f"**Genres:** {selected_row['artist_genres']}")
st.write(f"**Album:** {selected_row['album_name']}")
st.write(f"**Dataset:** {selected_row['dataset_source']}")

st.markdown("---")


if st.button("Recommend Similar Songs"):

    recommendations = recommender.recommend(
        selected_label=selected_song,
        top_n=top_n
    )

    st.subheader("Recommended Songs")

    display_df = recommendations.copy()

    display_df["similarity_score"] = display_df["similarity_score"].round(4)
    display_df["jaccard_score"] = display_df["jaccard_score"].round(4)
    display_df["cosine_svd_score"] = display_df["cosine_svd_score"].round(4)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("Method Explanation")

    st.write(
        """
        The recommender system combines multiple similarity techniques
        to generate recommendations:

        - Genres are transformed into binary vectors using MultiLabelBinarizer.
        - Jaccard similarity measures direct overlap between genre sets.
        - Numerical metadata such as popularity, followers, release year,
          and duration are normalized using MinMax scaling.
        - Categorical metadata such as album type and dataset source
          are one-hot encoded.
        - Truncated SVD reduces the high-dimensional metadata space into
          compact latent representations.
        - Cosine similarity is then applied in the SVD latent space.
        - Final recommendations combine both Jaccard similarity and
          cosine similarity scores.
        """
    )