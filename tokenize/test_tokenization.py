# src/tahap3_modify/tokenize_test.py

import pandas as pd
import pickle
import os

def create_output_directories(output_path):
    """
    Membuat direktori output jika belum ada
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Direktori baru dibuat: {output_path}")

def tokenize_test_data(input_path, tokenizer_path, output_dir):
    """
    Tokenisasi data test menggunakan tokenizer dari data training
    
    Args:
        input_path (str): Path file input test
        tokenizer_path (str): Path file tokenizer hasil training
        output_dir (str): Direktori untuk menyimpan hasil
    """
    print("\nTokenisasi data test...")
    
    # Baca data test
    df = pd.read_csv(input_path)
    
    # Load tokenizer yang sudah dilatih
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Transform data test
    X = tokenizer.transform(df['Judul'])
    
    # Simpan hasil tokenisasi
    token_path = os.path.join(output_dir, 'tokenized_text.npz')
    pd.DataFrame(
        X.toarray(),
        columns=tokenizer.get_feature_names_out()
    ).to_csv(token_path, index=False)
    
    print(f"Tokenisasi selesai. Hasil disimpan di: {output_dir}")

def main():
    # Setup direktori
    output_dir = "output/tokenization/test_data"
    create_output_directories(output_dir)
    
    try:
        # Path input dan tokenizer
        input_path = "output/tes-data.csv"
        tokenizer_path = "output/tokenization/normalized_data/tokenizer.pkl"
        
        # Tokenisasi
        tokenize_test_data(input_path, tokenizer_path, output_dir)
        
    except FileNotFoundError:
        print("Error: File input tidak ditemukan")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()