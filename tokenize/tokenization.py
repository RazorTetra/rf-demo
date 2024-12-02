# src/tahap3_modify/tokenization.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os

def create_output_directories(output_path):
    """
    Membuat direktori output jika belum ada
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Direktori baru dibuat: {output_path}")

def tokenize_text(input_path, output_dir, dataset_name):
    """
    Melakukan tokenisasi teks
    
    Args:
        input_path (str): Path file input
        output_dir (str): Direktori untuk menyimpan hasil
        dataset_name (str): Nama dataset untuk penamaan file
    """
    print(f"\nTokenisasi dataset: {dataset_name}")
    
    # Baca dataset
    df = pd.read_csv(input_path)
    
    # Inisialisasi tokenizer
    tokenizer = CountVectorizer(
        lowercase=True,
        strip_accents='unicode',
        token_pattern=r'\b[A-Za-z]+\b',
        max_df=0.95,
        min_df=2
    )
    
    # Fit dan transform
    X = tokenizer.fit_transform(df['Judul'])
    
    # Buat direktori output
    dataset_dir = os.path.join(output_dir, dataset_name)
    create_output_directories(dataset_dir)
    
    # Simpan hasil tokenisasi
    token_path = os.path.join(dataset_dir, 'tokenized_text.npz')
    pd.DataFrame(
        X.toarray(),
        columns=tokenizer.get_feature_names_out()
    ).to_csv(token_path, index=False)
    
    # Simpan tokenizer
    tokenizer_path = os.path.join(dataset_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(f"Tokenisasi selesai. Hasil disimpan di: {dataset_dir}")
    return dataset_dir

def main():
    # Setup direktori output
    output_dir = "output/tokenization"
    create_output_directories(output_dir)
    
    try:
        # Proses dataset utama
        main_input = "output/text-normalization/training-data-normalized.csv"
        if os.path.exists(main_input):
            tokenize_text(main_input, output_dir, "normalized_data")
        
        # Proses hasil sampling jika ada
        sampling_inputs = {
            "undersampling": "output/balance-handling/reduced_to_156.csv",
            "oversampling": "output/balance-handling/increased_to_330.csv",
            "combined": "output/balance-handling/balanced_to_250.csv"
        }
        
        for name, path in sampling_inputs.items():
            if os.path.exists(path):
                tokenize_text(path, output_dir, name)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()