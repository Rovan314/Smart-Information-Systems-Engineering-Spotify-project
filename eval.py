import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from recommender import SpotifyRecommender

def run_evaluation():
    print("Loading data and building feature matrix...")
    # 1. Initialize the recommender
    recommender = SpotifyRecommender(
        modern_path="data/spotify_data clean.csv",
        classic_path="data/track_data_final.csv"
    )
    recommender.load_and_prepare_data()
    recommender.build_feature_matrix()

    # =====================================================================
    # 2. DEFINE YOUR TEST CASE (Use 'Track Name - Artist Name' format)
    # =====================================================================
    test_song_label = "3 - Britney Spears" 
    
    # Define ~5 to 10 songs that are objectively similar to the test song.
    # Make sure they are typed exactly as they appear in the dataset!
    ground_truth_positives = [
        "Toxic - Britney Spears",
        "Oops!...I Did It Again - Britney Spears",
        "Womanizer - Britney Spears",
        "Circus - Britney Spears",
        "Gimme More - Britney Spears"
        "Espresso - Sabrina Carpenter", 
        "Love Me Harder - Ariana Grande", 
        "Training Season - Dua Lipa"
    ]

    print(f"Testing recommendations for track: {test_song_label}")
    
    # 3. Get Top 20 recommendations
    recs = recommender.recommend(selected_label=test_song_label, top_n=20)
    
    if recs.empty:
        print(f"Error: Could not find '{test_song_label}' in the dataset. Check spelling!")
        return

    # Reconstruct the song_labels for the recommended tracks to compare them
    recs["song_label"] = recs["track_name"].astype(str) + " - " + recs["artist_name"].astype(str)
    recommended_labels = recs["song_label"].tolist()

    print("\n--- ACTUAL TOP 20 RECOMMENDATIONS ---")
    for i, song in enumerate(recommended_labels):
        print(f"{i+1}. {song}")
    print("-------------------------------------\n")
    print("-------------------------------------\n")
    print("\n--- ARTIST DIVERSITY SCORE ---")
    unique_artists = recs["artist_name"].nunique()
    total_recs = len(recs)
    diversity_ratio = unique_artists / total_recs
    
    print(f"Unique Artists: {unique_artists} out of {total_recs} tracks")
    print(f"Diversity Ratio: {diversity_ratio:.2f} (1.0 is perfect diversity, near 0 is highly repetitive)")
    print("-------------------------------------\n")

    print("\n--- METADATA COMPARISON (The Vibe Check) ---")
    for index, row in recs.head(10).iterrows():
        print(f"🎵 {row['track_name']} by {row['artist_name']}")
        print(f"   Year: {row['release_year']} | Popularity: {row['track_popularity']}/100")
        print(f"   Genres: {row['artist_genres']}")
        print("-" * 40)
    print("\n--- SCORE BREAKDOWN (Why it was chosen) ---")
    # Iterating through the DataFrame rows to see the math
    for index, row in recs.head(10).iterrows():
        print(f"🎵 {row['track_name']} - {row['artist_name']}")
        print(f"   Final Score: {row['similarity_score']:.4f}")
        print(f"   ┣━ Cosine (SVD) Match:  {row['cosine_svd_score']:.4f}")
        print(f"   ┗━ Jaccard (Genre) Match: {row['jaccard_score']:.4f}\n")

    # 4. Calculate Scores
    all_labels = recommender.df["song_label"].tolist()
    y_true = []
    y_pred = []
    
    for label in all_labels:
        # Skip the seed song itself so it doesn't skew the grade
        if label == test_song_label: continue
        
        # Ground Truth: 1 if it's in our known relevant list, else 0
        y_true.append(1 if label in ground_truth_positives else 0)
        
        # Prediction: 1 if our model recommended it in the Top K, else 0
        y_pred.append(1 if label in recommended_labels else 0)

    # 5. Print the Lab 7 metrics to the terminal
    conf_matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Positives  (Relevant & Recommended): {conf_matrix[1][1]}")
    print(f"False Positives (Irrelevant but Recommended): {conf_matrix[0][1]}")
    print(f"False Negatives (Relevant but Missed): {conf_matrix[1][0]}")
    print(f"True Negatives  (Irrelevant & Ignored): {conf_matrix[0][0]}")
    
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    run_evaluation()