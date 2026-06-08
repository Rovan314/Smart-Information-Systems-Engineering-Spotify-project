import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from recommender import SpotifyRecommender


def calculate_ranking_metrics(recommended_list, ground_truth_list, k=20):
    """
    Calculates Precision@K, Recall@K, and NDCG@K.
    """
    top_k_recs = recommended_list[:k]
    hits = [1 if item in ground_truth_list else 0 for item in top_k_recs]
    num_hits = sum(hits)

    precision_at_k = num_hits / k
    recall_at_k = num_hits / len(ground_truth_list) if len(ground_truth_list) > 0 else 0.0

    dcg = 0.0
    for i, rel in enumerate(hits):
        dcg += rel / np.log2(i + 2)

    ideal_hits = [1] * min(len(ground_truth_list), k)
    idcg = 0.0
    for i, rel in enumerate(ideal_hits):
        idcg += rel / np.log2(i + 2)

    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

    return precision_at_k, recall_at_k, ndcg_at_k


def run_evaluation():
    print("Loading data and building feature matrix...")
    # 1. Initialize the recommender
    recommender = SpotifyRecommender(
        modern_path="data/spotify_data clean.csv",
        classic_path="data/track_data_final.csv"
    )
    recommender.load_and_prepare_data()
    recommender.build_feature_matrix()

    # 2. Define your test case (Use 'Track Name - Artist Name' format)
    test_song_label = "3 - Britney Spears" 
    
    # Define ~5 to 10 songs that are objectively similar to the test song.
    # Make sure they are typed exactly as they appear in the dataset!
    ground_truth_positives = [
        "Toxic - Britney Spears",
        "Oops!...I Did It Again - Britney Spears",
        "Womanizer - Britney Spears",
        "Circus - Britney Spears",
        "Gimme More - Britney Spears",
        "Espresso - Sabrina Carpenter", 
        "Love Me Harder - Ariana Grande", 
        "Training Season - Dua Lipa",
        "Don’t Smile - Sabrina Carpenter",
        "Tears - Sabrina Carpenter"
    ]

    print(f"Testing recommendations for track: {test_song_label}")
    
    # 3. Get Top 20 recommendations
    recs = recommender.recommend(selected_label=test_song_label, top_n=20)
    
    if recs.empty:
        print(f"Error: Could not find '{test_song_label}'. Check spelling!")
        return

    # Reconstruct the labels for the recommended tracks
    recs["song_label"] = recs["track_name"].astype(str) + " - " + recs["artist_name"].astype(str)
    recommended_labels = recs["song_label"].tolist()

    K = 20
    p_at_k, r_at_k, ndcg_at_k = calculate_ranking_metrics(
        recommended_list=recommended_labels,
        ground_truth_list=ground_truth_positives,
        k=K,
    )

    print("\n" + "=" * 50)
    print(f" RANK-AWARE METRICS (@K={K})")
    print("=" * 50)
    print(f"Precision@{K}: {p_at_k:.4f}")
    print(f"Recall@{K}:    {r_at_k:.4f}")
    print(f"NDCG@{K}:      {ndcg_at_k:.4f}")
    print("=" * 50)

    # True Positives: Songs it recommended that ARE on our list
    true_positives_list = [song for song in recommended_labels if song in ground_truth_positives]
    
    # False Positives: Songs it recommended that are NOT on our list
    false_positives_list = [song for song in recommended_labels if song not in ground_truth_positives]
    
    # False Negatives: Songs on our list that the model MISSED
    false_negatives_list = [song for song in ground_truth_positives if song not in recommended_labels]


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

    # 5. Print the metrics to the terminal
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Try to unpack into TN, FP, FN, TP for easier reporting (binary classification)
    tn = fp = fn = tp = None
    if getattr(conf_matrix, "shape", None) == (2, 2):
        tn, fp, fn, tp = conf_matrix.ravel()
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    if tn is not None:
        print(f"TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")
    
    print(f"\n✅ TRUE POSITIVES ({len(true_positives_list)})")
    print("   We wanted these, and the model successfully found them:")
    for song in true_positives_list:
        print(f"    + {song}")

    print(f"\n❌FALSE POSITIVES ({len(false_positives_list)})")
    print("   The model recommended these, but they weren't on our target list:")
    for song in false_positives_list:
        print(f"    - {song}")

    print(f"\n👻 FALSE NEGATIVES ({len(false_negatives_list)})")
    print("   We explicitly asked for these, but the model missed them:")
    for song in false_negatives_list:
        print(f"    - {song}")
    print("\n")

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
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    run_evaluation()