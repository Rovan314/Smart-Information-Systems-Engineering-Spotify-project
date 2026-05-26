import ast
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class SpotifyRecommender:
    def __init__(self, modern_path, classic_path):
        self.modern_path = modern_path
        self.classic_path = classic_path
        self.df = None
        self.feature_matrix = None

    def load_and_prepare_data(self):
        modern = pd.read_csv(self.modern_path)
        classic = pd.read_csv(self.classic_path)

        modern["dataset_source"] = "Modern 2025 Tracks"
        classic["dataset_source"] = "Classic 2009-2023 Tracks"

        # Standardize duration column
        modern["track_duration_ms"] = modern["track_duration_min"] * 60 * 1000
        modern = modern.drop(columns=["track_duration_min"], errors="ignore")

        df = pd.concat([modern, classic], ignore_index=True)

        # Basic cleaning
        df = df.drop_duplicates(subset=["track_id"])
        df = df.dropna(subset=["track_name", "artist_name"])

        # Fill missing values
        df["artist_genres"] = df["artist_genres"].fillna("unknown")
        df["album_type"] = df["album_type"].fillna("unknown")

        numeric_cols = [
            "track_popularity",
            "artist_popularity",
            "artist_followers",
            "album_total_tracks",
            "track_duration_ms"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

        df["explicit"] = df["explicit"].fillna(False).astype(int)

        # Clean genres into usable text
        df["clean_genres"] = df["artist_genres"].apply(self.clean_genres)

        # Add release year
        df["album_release_date"] = pd.to_datetime(
            df["album_release_date"],
            errors="coerce"
        )

        df["release_year"] = df["album_release_date"].dt.year
        df["release_year"] = df["release_year"].fillna(df["release_year"].median())

        # Display label for dropdown
        df["song_label"] = (
            df["track_name"].astype(str)
            + " - "
            + df["artist_name"].astype(str)
        )

        df = df.reset_index(drop=True)

        self.df = df
        return df

    @staticmethod
    def clean_genres(value):
        """
        Handles both:
        "country hip hop, southern hip hop"
        and
        "['pop', 'dance pop']"
        """
        if pd.isna(value):
            return "unknown"

        value = str(value).strip()

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return " ".join(parsed)
        except Exception:
            pass

        return value.replace(",", " ")

    def build_feature_matrix(self):
        if self.df is None:
            raise ValueError("Data must be loaded before building features.")

        text_features = self.df["clean_genres"]

        numeric_features = self.df[
            [
                "track_popularity",
                "artist_popularity",
                "artist_followers",
                "album_total_tracks",
                "track_duration_ms",
                "release_year",
                "explicit"
            ]
        ]

        categorical_features = self.df[["album_type", "dataset_source"]]

        genre_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=500
        )

        genre_matrix = genre_vectorizer.fit_transform(text_features)

        numeric_scaler = MinMaxScaler()
        numeric_matrix = numeric_scaler.fit_transform(numeric_features)

        categorical_encoder = OneHotEncoder(handle_unknown="ignore")
        categorical_matrix = categorical_encoder.fit_transform(categorical_features)

        self.feature_matrix = hstack([
            genre_matrix,
            numeric_matrix,
            categorical_matrix
        ]).tocsr()

        return self.feature_matrix

    def recommend(self, selected_label, top_n=10):
        if self.df is None or self.feature_matrix is None:
            raise ValueError("Data and feature matrix must be prepared first.")

        matches = self.df[self.df["song_label"] == selected_label]

        if matches.empty:
            return pd.DataFrame()

        selected_index = matches.index[0]

        selected_vector = self.feature_matrix[selected_index]
        similarity_scores = cosine_similarity(
            selected_vector,
            self.feature_matrix
        ).flatten()

        # Exclude selected song itself
        similar_indices = similarity_scores.argsort()[::-1]
        similar_indices = [
            i for i in similar_indices
            if i != selected_index
        ][:top_n]

        results = self.df.iloc[similar_indices].copy()
        results["similarity_score"] = similarity_scores[similar_indices]

        return results[
            [
                "track_name",
                "artist_name",
                "artist_genres",
                "track_popularity",
                "artist_popularity",
                "album_name",
                "album_type",
                "release_year",
                "dataset_source",
                "similarity_score"
            ]
        ]