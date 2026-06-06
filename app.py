import streamlit as st
import pandas as pd

from recommender import SpotifyRecommender


st.set_page_config(
    page_title="Spotify Metadata Recommender",
    page_icon="🎧",
    layout="wide"
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


st.title("🎧 Spotify Metadata-Based Recommender System")

st.write(
    """
    This project recommends songs using Spotify metadata such as artist genres,
    track popularity, artist popularity, followers, album type, release year,
    and duration. The system applies content-based filtering with cosine similarity.
    """
)

recommender = load_recommender()
df = recommender.df

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

st.write("**Artist:**", selected_row["artist_name"])
st.write("**Genres:**", selected_row["artist_genres"])
st.write("**Album:**", selected_row["album_name"])
st.write("**Dataset:**", selected_row["dataset_source"])

if st.button("Recommend Similar Songs"):
    recommendations = recommender.recommend(
        selected_label=selected_song,
        top_n=top_n
    )

    st.subheader("Recommended Songs")

    display_df = recommendations.copy()
    display_df["similarity_score"] = display_df["similarity_score"].round(4)
    display_df["jaccard_score"]    = display_df["jaccard_score"].round(4)
    display_df["cosine_svd_score"] = display_df["cosine_svd_score"].round(4)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Method Explanation")

    st.write(
        """
        The recommender represents each song as a metadata vector. Genres are transformed
        using TF-IDF, while numerical features such as popularity, followers, release year,
        and duration are normalized. Categorical features such as album type and dataset
        source are one-hot encoded. Cosine similarity is then used to find songs with the
        most similar metadata profiles.
        """
    )