# src/tahap3_modify/extract_test_features.py

import pandas as pd
import numpy as np
import pickle
import os

def create_output_directories(output_path):
    """
    Membuat direktori output jika belum ada
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Direktori baru dibuat: {output_path}")

def extract_test_features(tokenized_dir, transformer_path, output_dir):
    """
    Ekstraksi fitur dari data test yang sudah ditokenisasi
    
    Args:
        tokenized_dir (str): Direktori hasil tokenisasi test
        transformer_path (str): Path file TF-IDF transformer hasil training
        output_dir (str): Direktori untuk menyimpan hasil
    """
    print("\nEkstraksi fitur data test...")
    
    # Baca hasil tokenisasi
    token_path = os.path.join(tokenized_dir, 'tokenized_text.npz')
    token_matrix = pd.read_csv(token_path)
    
    # Load TF-IDF transformer yang sudah dilatih
    with open(transformer_path, 'rb') as f:
        tfidf_transformer = pickle.load(f)
    
    # Transform data test
    X = tfidf_transformer.transform(token_matrix)
    
    # Simpan matrix TF-IDF
    tfidf_path = os.path.join(output_dir, 'tfidf_matrix.npz')
    np.savez_compressed(tfidf_path, 
                       data=X.data,
                       indices=X.indices,
                       indptr=X.indptr,
                       shape=X.shape)
    
    print(f"Ekstraksi fitur selesai. Hasil disimpan di: {output_dir}")

def main():
    # Setup direktori
    output_dir = "output/feature-extraction/test_data"
    create_output_directories(output_dir)
    
    try:
        # Path input dan transformer
        tokenized_dir = "output/tokenization/test_data"
        transformer_path = "output/feature-extraction/normalized_data/tfidf_transformer.pkl"
        
        # Ekstraksi fitur
        extract_test_features(tokenized_dir, transformer_path, output_dir)
        
    except FileNotFoundError:
        print("Error: File input tidak ditemukan")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()