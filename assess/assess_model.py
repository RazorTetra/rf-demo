# src/tahap5_assess/assess_model.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os

def create_output_directories(output_path):
    """
    Membuat direktori output jika belum ada
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Direktori baru dibuat: {output_path}")

def plot_cv_accuracy(output_dir):
    """
    Plot accuracy dari hasil cross validation
    """
    # Baca hasil evaluasi model
    with open("output/model-random-forest/model_evaluation.txt", 'r') as f:
        eval_text = f.read()
    
    # Ekstrak skor CV dari text
    import re
    scores_text = re.findall(r"Scores: \[(.*?)\]", eval_text)[0]
    cv_scores = [float(x) for x in scores_text.split()]
    
    # Plot skor CV
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-')
    plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', 
                label=f'Mean Accuracy: {np.mean(cv_scores):.4f}')
    
    plt.title('Cross Validation Accuracy Scores')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Tambahkan nilai di atas titik
    for i, score in enumerate(cv_scores):
        plt.text(i + 1, score, f'{score:.4f}', 
                horizontalalignment='center', 
                verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_accuracy.png'))
    plt.close()

def plot_prediction_confidence(results, output_dir):
    """
    Visualisasi tingkat kepercayaan model terhadap prediksinya
    """
    # Ambil probability tertinggi untuk setiap prediksi
    max_probs = []
    for idx, row in results.iterrows():
        probs = [row[col] for col in results.columns if col.startswith('Probability_')]
        max_probs.append(max(probs))
        
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=20, edgecolor='black')
    plt.title('Distribusi Tingkat Kepercayaan Prediksi')
    plt.xlabel('Probability')
    plt.ylabel('Jumlah Prediksi')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'prediction_confidence.png'))
    plt.close()
    
    # Simpan statistik confidence
    confidence_stats = {
        'mean_confidence': np.mean(max_probs),
        'median_confidence': np.median(max_probs),
        'min_confidence': np.min(max_probs),
        'max_confidence': np.max(max_probs)
    }
    
    with open(os.path.join(output_dir, 'confidence_stats.json'), 'w') as f:
        json.dump(confidence_stats, f, indent=4)

def load_test_features(feature_dir):
    """
    Memuat fitur TF-IDF dari data test
    """
    tfidf_path = os.path.join(feature_dir, "tfidf_matrix.npz")
    loader = np.load(tfidf_path)
    X = csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                   shape=loader['shape'])
    return X

def predict_and_save(model, X_test, output_dir):
    """
    Melakukan prediksi dan menyimpan hasilnya
    """
    y_pred = model.predict(X_test)
    probas = model.predict_proba(X_test)
    
    results = pd.DataFrame({
        'Predicted_Class': y_pred
    })
    
    for i, class_name in enumerate(model.classes_):
        results[f'Probability_{class_name}'] = probas[:, i]
    
    results.to_csv(os.path.join(output_dir, 'prediction_results.csv'), index=False)
    return results

def plot_prediction_distribution(results, output_dir):
    """
    Visualisasi distribusi kelas hasil prediksi
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=results, x='Predicted_Class')
    plt.title('Distribusi Kelas Hasil Prediksi')
    plt.xlabel('Kelas')
    plt.ylabel('Jumlah Prediksi')
    plt.xticks(rotation=45)
    
    # Tambahkan label jumlah di atas bar
    for i, count in enumerate(results['Predicted_Class'].value_counts()):
        plt.text(i, count, str(count), 
                horizontalalignment='center', 
                verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'))
    plt.close()

def main():
    output_dir = "output/model-assessment" 
    create_output_directories(output_dir)
    
    try:
        # Load model
        print("\nMemuat model...")
        with open('output/model-random-forest/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Plot CV accuracy
        print("Membuat visualisasi accuracy...")
        plot_cv_accuracy(output_dir)
        
        # Load dan proses data test
        print("Memuat data test...")
        feature_dir = "output/feature-extraction/test_data"
        X_test = load_test_features(feature_dir)
        print(f"Jumlah data test: {X_test.shape[0]}")
        
        # Prediksi
        print("\nMelakukan prediksi...")
        results = predict_and_save(model, X_test, output_dir)
        
        # Visualisasi
        print("\nMembuat visualisasi prediksi...")
        plot_prediction_distribution(results, output_dir)
        plot_prediction_confidence(results, output_dir)
        
        print(f"\nAnalisis selesai. Hasil tersimpan di: {output_dir}")
        
        # Tampilkan ringkasan
        print("\nRingkasan Prediksi:")
        print("-"*50)
        print("Distribusi kelas:")
        print(results['Predicted_Class'].value_counts())
        
        # Load dan tampilkan statistik confidence
        with open(os.path.join(output_dir, 'confidence_stats.json'), 'r') as f:
            conf_stats = json.load(f)
        print("\nStatistik Confidence:")
        print(f"Rata-rata: {conf_stats['mean_confidence']:.4f}")
        print(f"Median  : {conf_stats['median_confidence']:.4f}")
        print(f"Min     : {conf_stats['min_confidence']:.4f}")
        print(f"Max     : {conf_stats['max_confidence']:.4f}")
        
    except FileNotFoundError:
        print("Error: File input tidak ditemukan")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()