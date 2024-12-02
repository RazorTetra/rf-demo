# src/tahap4_model/random_forest_model.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def create_output_directories(output_path):
    """
    Membuat direktori output jika belum ada
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Direktori baru dibuat: {output_path}")

def load_features(feature_dir):
    """
    Memuat fitur TF-IDF dari format sparse matrix
    """
    # Muat TF-IDF matrix
    tfidf_path = os.path.join(feature_dir, "tfidf_matrix.npz")
    loader = np.load(tfidf_path)
    
    # Rekonstruksi sparse matrix
    X = csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                   shape=loader['shape'])
    
    return X

def train_model(X, y, output_dir):
    """
    Melatih model Random Forest dengan cross validation
    """
    # Inisialisasi model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1  
    )
    
    # Cross validation
    print("Melakukan 5-fold cross validation...")
    cv_scores = cross_val_score(rf_model, X, y, cv=5)
    print(f"Skor CV: {cv_scores}")
    print(f"Rata-rata akurasi CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train model dengan semua data
    print("\nMelatih model final dengan semua data...")
    rf_model.fit(X, y)
    
    # Simpan model
    print("\nMenyimpan model...")
    model_path = os.path.join(output_dir, "random_forest_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Simpan hasil evaluasi
    with open(os.path.join(output_dir, 'model_evaluation.txt'), 'w') as f:
        f.write("EVALUASI MODEL RANDOM FOREST\n")
        f.write("="*50 + "\n\n")
        f.write("Cross Validation Scores:\n")
        f.write(f"Scores: {cv_scores}\n")
        f.write(f"Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
    
    return rf_model

def main():
    # Setup direktori
    output_dir = "output/model-random-forest"
    create_output_directories(output_dir)
    
    try:
        # Load data
        print("\nMemuat data training...")
        feature_dir = "output/feature-extraction/normalized_data"
        X = load_features(feature_dir)
        print(f"Jumlah fitur dimuat: {X.shape[1]}")
        
        # Load labels
        labels_path = "output/text-normalization/training-data-normalized.csv"
        y = pd.read_csv(labels_path)['Konsentrasi']
        print(f"Jumlah data training: {len(y)}")
        
        # Train model
        print("\nMelatih model Random Forest...")
        model = train_model(X, y, output_dir)
        
        print(f"\nPelatihan selesai. Model dan evaluasi tersimpan di: {output_dir}")
        
    except FileNotFoundError:
        print("Error: File input tidak ditemukan")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()