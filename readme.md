# Panduan
## Versi 1.1

### 1. Clone repository atau download zip
Clone dengan perintah berikut di cmd atau powershell:

    git clone https://github.com/RazorTetra/rf-demo.git

atau download zip:

    https://github.com/RazorTetra/rf-demo/archive/refs/heads/main.zip


### 2. Install Dependecies
Pastikan versi python yang terinstal adalah 3.11.4 atau buat envs

Jalankan perintah berikut di cmd atau powershell

    pip install -r requirements.txt
    pip install matplotlib seaborn


### 3. Tokenize Training Data dan Tes Data
Jalankan perintah berikut di terminal secara berurutan di direktori project

    python tokenize/tokenization.py
    python tokenize/feature_extraction.py
    python tokenize/test_tokenization.py
    python tokenize/text_feature.py

### 4. Buat Model RF
Jalankan perintah berikut untuk membuat model

    python model/random_forest_model.py

### 5. Lakukan Assess Step
Jalankan perintah berikut untuk melihat assess dari model

    python assess/assess_model.py
