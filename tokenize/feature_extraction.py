# src/tahap3_modify/feature_extraction.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import os

def create_output_directories(output_path):
    """
    Membuat direktori output jika belum ada
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Direktori baru dibuat: {output_path}")

def extract_features(tokenized_dir, output_dir, dataset_name):
    """
    Mengekstrak fitur dari teks yang sudah ditokenisasi
    
    Args:
        tokenized_dir (str): Direktori hasil tokenisasi
        output_dir (str): Direktori untuk menyimpan hasil
        dataset_name (str): Nama dataset untuk penamaan file
    """
    print(f"\nEkstraksi fitur dataset: {dataset_name}")
    
    # Baca hasil tokenisasi
    tokenized_path = os.path.join(tokenized_dir, 'tokenized_text.npz')
    token_matrix = pd.read_csv(tokenized_path)
    
    # Inisialisasi TF-IDF Transformer
    tfidf_transformer = TfidfTransformer()
    
    # Fit dan transform
    X = tfidf_transformer.fit_transform(token_matrix)
    
    # Buat direktori output
    dataset_dir = os.path.join(output_dir, dataset_name)
    create_output_directories(dataset_dir)
    
    # Simpan matrix TF-IDF
    tfidf_path = os.path.join(dataset_dir, 'tfidf_matrix.npz')
    np.savez_compressed(tfidf_path, 
                       data=X.data,
                       indices=X.indices,
                       indptr=X.indptr,
                       shape=X.shape)
    
    # Simpan transformer
    transformer_path = os.path.join(dataset_dir, 'tfidf_transformer.pkl')
    with open(transformer_path, 'wb') as f:
        pickle.dump(tfidf_transformer, f)
    
    # Simpan informasi fitur
    feature_info = pd.DataFrame({
        'Feature': token_matrix.columns,
        'IDF_Weight': tfidf_transformer.idf_
    })
    feature_info.to_csv(os.path.join(dataset_dir, 'feature_info.csv'), 
                       index=False)
    
    print(f"Ekstraksi fitur selesai. Hasil disimpan di: {dataset_dir}")

def main():
    # Setup direktori output
    output_dir = "output/feature-extraction"
    create_output_directories(output_dir)
    
    try:
        # Base dirs
        token_base_dir = "output/tokenization"
        
        # Proses dataset utama
        if os.path.exists(os.path.join(token_base_dir, "normalized_data")):
            extract_features(
                os.path.join(token_base_dir, "normalized_data"),
                output_dir,
                "normalized_data"
            )
        
        # Proses hasil sampling
        sampling_dirs = ["undersampling", "oversampling", "combined"]
        
        for name in sampling_dirs:
            token_dir = os.path.join(token_base_dir, name)
            if os.path.exists(token_dir):
                extract_features(token_dir, output_dir, name)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()