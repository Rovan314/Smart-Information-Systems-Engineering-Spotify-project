import ast
import numpy as np
import pandas as pd
 
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, MultiLabelBinarizer
 
 
class SpotifyRecommender:
    """
    Content-based recommender that combines three similarity signals:
 
    1. Jaccard similarity on binary genre vectors
       For two tracks A and B with genre sets G_A and G_B:
           J(A, B) = |G_A ∩ G_B| / |G_A ∪ G_B|
       Computed at query time against all tracks — avoids storing
       the full n×n Jaccard matrix in memory.
 
    2. TruncatedSVD (latent factor decomposition)
       Applied to the combined feature matrix (binary genres +
       scaled numerics + one-hot categoricals) to find the
       dominant latent dimensions that explain the most variance.
 
    3. Cosine similarity on SVD-reduced embeddings
       Once each track is represented as a dense SVD latent
       vector, cosine similarity measures their angular closeness
       in that compressed space.
 
    Final recommendation score:
        score = JACCARD_WEIGHT * jaccard + COSINE_WEIGHT * cosine_svd
    """
 
    JACCARD_WEIGHT = 0.4
    COSINE_WEIGHT  = 0.6
    SVD_COMPONENTS = 110
 
    def __init__(self, modern_path, classic_path):
        self.modern_path   = modern_path
        self.classic_path  = classic_path
 
        self.df             = None
        self.feature_matrix = None   # SVD-reduced dense matrix  (n_tracks × k)
        self.genre_binary   = None   # Binary genre matrix        (n_tracks × n_genres)
        self.mlb            = None   # Fitted MultiLabelBinarizer (kept for inspection)
        self.svd            = None   # Fitted TruncatedSVD        (kept for inspection)
 
    # ──────────────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────────────
 
    def load_and_prepare_data(self):
        modern  = pd.read_csv(self.modern_path)
        classic = pd.read_csv(self.classic_path)
 
        modern["dataset_source"]  = "Modern 2025 Tracks"
        classic["dataset_source"] = "Classic 2009-2023 Tracks"
 
        # Normalise duration to milliseconds
        modern["track_duration_ms"] = modern["track_duration_min"] * 60 * 1000
        modern = modern.drop(columns=["track_duration_min"], errors="ignore")
 
        df = pd.concat([modern, classic], ignore_index=True)
 
        df = df.drop_duplicates(subset=["track_name", "artist_name"])
        df = df.dropna(subset=["track_name", "artist_name"])
 
        df["artist_genres"] = df["artist_genres"].fillna("unknown")
        df["album_type"]    = df["album_type"].fillna("unknown")
 
        numeric_cols = [
            "track_popularity",
            "artist_popularity",
            "artist_followers",
            "album_total_tracks",
            "track_duration_ms",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
 
        df["explicit"] = df["explicit"].fillna(False).astype(int)
 
        # Parse genres into Python lists (needed for MLB and Jaccard)
        df["genre_list"] = df["artist_genres"].apply(self._parse_genre_list)

        # Count the number of genres associated with each track (could indicate genre diversity or niche-ness)
        df["num_genres"] = df["genre_list"].apply(len)
 
        # Release year
        df["album_release_date"] = pd.to_datetime(
            df["album_release_date"], errors="coerce"
        )
        df["release_year"] = df["album_release_date"].dt.year
        df["release_year"] = df["release_year"].fillna(df["release_year"].median())
 
        # Dropdown label
        df["song_label"] = (
            df["track_name"].astype(str) + " - " + df["artist_name"].astype(str)
        )
 
        # Keep clean_genres for app.py compatibility (not used in pipeline now)
        df["clean_genres"] = df["genre_list"].apply(lambda lst: " ".join(lst))
 
        df = df.reset_index(drop=True)
        self.df = df
        return df
 
    # ──────────────────────────────────────────────────────────────────────────
    # Feature matrix:  binary genres + scaled numerics + one-hot categoricals
    #                  → TruncatedSVD → self.feature_matrix
    # ──────────────────────────────────────────────────────────────────────────
 
    def build_feature_matrix(self):
        if self.df is None:
            raise ValueError("Call load_and_prepare_data() first.")
 
        # ── 1. Binary genre matrix  (Jaccard representation) ──────────────────
        # MultiLabelBinarizer converts each track's genre list into a binary
        # row vector: 1 if the track belongs to that genre, 0 otherwise.
        # This binary encoding is exactly what Jaccard similarity operates on:
        #   J(A, B) = dot(A, B) / (|A| + |B| - dot(A, B))
        self.mlb = MultiLabelBinarizer()
        genre_binary = self.mlb.fit_transform(self.df["genre_list"]).astype(np.float32)
        self.genre_binary = genre_binary                 # stored for query-time Jaccard
 
        genre_sparse = csr_matrix(genre_binary)
 
        # ── 2. Scaled numeric features ────────────────────────────────────────
        numeric_cols = [
            "track_popularity", 
            "artist_popularity", 
            "artist_followers", 
            "track_duration_ms",
            "release_year",    # <Eval
            "num_genres",      # <Eval
            "explicit"     # <Eval
        ]
        
        # Fill missing numeric values with the median (to be safe)
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
            
        scaler = MinMaxScaler()
        numeric_matrix = csr_matrix(scaler.fit_transform(self.df[numeric_cols]))
        numeric_sparse = csr_matrix(numeric_matrix)
 
        # ── 3. One-hot categorical features ───────────────────────────────────
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        cat_sparse  = cat_encoder.fit_transform(
            self.df[["album_type"]] #removed "dataset_source" to reduce dimensionality and potential data leakage
        )
 
        # ── 4. Stack all feature blocks ───────────────────────────────────────
        combined = hstack([genre_sparse, numeric_sparse, cat_sparse]).tocsr()
 
        # ── 5. TruncatedSVD: compress to k latent dimensions ─────────────────
        # SVD finds the directions of maximum variance in the combined feature
        # space. Each track is projected onto the top-k singular vectors,
        # giving a dense embedding that captures the dominant patterns across
        # genre, popularity, release era, and metadata features jointly.
        n_components = min(self.SVD_COMPONENTS, combined.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.feature_matrix = self.svd.fit_transform(combined).astype(np.float32)
 
        explained = self.svd.explained_variance_ratio_.sum()
        print(
            f"SVD: {n_components} components explain "
            f"{explained:.1%} of total variance."
        )
 
        return self.feature_matrix
 
    # ──────────────────────────────────────────────────────────────────────────
    # Recommendation: Jaccard (genres) + Cosine (SVD embeddings) → blended rank
    # ──────────────────────────────────────────────────────────────────────────
 
    def recommend(self, selected_label, top_n=10):
        if self.df is None or self.feature_matrix is None:
            raise ValueError("Call load_and_prepare_data() and build_feature_matrix() first.")
 
        matches = self.df[self.df["song_label"] == selected_label]
        if matches.empty:
            return pd.DataFrame()
 
        idx = matches.index[0]
 
        # ── Step A: Jaccard similarity on binary genre vectors ─────────────────
        # For the query track's genre vector q and every other track's vector v:
        #   J(q, v) = dot(q, v) / (||q||^2 + ||v||^2 - dot(q, v))
        # Avoids building the full n×n matrix — computed for one row at a time.
        q_genre = self.genre_binary[idx]                         # (n_genres,)
        dot_products  = self.genre_binary.dot(q_genre)           # (n_tracks,)
        sum_all       = self.genre_binary.sum(axis=1)            # (n_tracks,)
        q_sum         = q_genre.sum()
        union         = sum_all + q_sum - dot_products
        jaccard_scores = np.where(union > 0, dot_products / union, 0.0)
 
        # ── Step B: Cosine similarity on SVD-reduced latent embeddings ─────────
        # Each track is now a dense k-dimensional vector in latent space.
        # Cosine similarity measures the angle between them, capturing
        # combined similarity across genres, popularity, era, and metadata.
        q_svd          = self.feature_matrix[idx : idx + 1]      # (1, k)
        cosine_scores  = cosine_similarity(q_svd, self.feature_matrix).flatten()
 
        # ── Step C: Blend both signals ─────────────────────────────────────────
        # Jaccard emphasises genre overlap (interpretable, sparse).
        # Cosine-SVD captures richer latent patterns across all features.
        final_scores = (
            self.JACCARD_WEIGHT * jaccard_scores
            + self.COSINE_WEIGHT * cosine_scores
        )
 
        # Rank, excluding the query track itself
        ranked = np.argsort(final_scores)[::-1]
        ranked = [i for i in ranked if i != idx][:top_n]
 
        results = self.df.iloc[ranked].copy()
        results["similarity_score"]  = final_scores[ranked].round(4)
        results["jaccard_score"]     = jaccard_scores[ranked].round(4)
        results["cosine_svd_score"]  = cosine_scores[ranked].round(4)
 
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
                "jaccard_score",
                "cosine_svd_score",
                "similarity_score",
            ]
        ]
 
    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_genre_list(value):

        if pd.isna(value):
            return ["unknown"]

        value = str(value).strip()

        if value == "" or value.lower() in ["nan", "none", "[]"]:
            return ["unknown"]

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):

                genres = [
                    str(g).strip().lower()
                    for g in parsed
                    if str(g).strip()
                ]

                return sorted(set(genres)) or ["unknown"]

        except Exception:
            pass

        if "," in value:

            genres = [
                genre.strip().lower()
                for genre in value.split(",")
                if genre.strip()
            ]

        else:
            genres = [value.lower()]

        return sorted(set(genres)) or ["unknown"]

    @staticmethod
    def clean_genres(value):

        genres = SpotifyRecommender._parse_genre_list(value)

        return ", ".join(genres)