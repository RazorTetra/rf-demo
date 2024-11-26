import streamlit as st
import joblib
import os
import pandas as pd
import plotly.graph_objects as go

def load_models():
    try:
        models_dir = os.path.join('public', 'models')
        model = joblib.load(os.path.join(models_dir, 'random_forest_model.joblib'))
        vectorizer = joblib.load(os.path.join(models_dir, 'final_vectorizer.joblib'))
        le = joblib.load(os.path.join(models_dir, 'final_label_encoder.joblib'))
        return model, vectorizer, le
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.stop()

def create_gauge_chart(value, title):
    # Convert probability to percentage
    percentage = value * 100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        delta = {'reference': 33.33}, 
        number = {'suffix': "%", 'font': {'size': 26}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#1F77B4", 'thickness': 0.6},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33.33], 'color': '#EBF5FB'},
                {'range': [33.33, 66.66], 'color': '#AED6F1'},
                {'range': [66.66, 100], 'color': '#3498DB'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 33.33
            }
        }
    ))
    
    # Visualisasi
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"}
    )
    
    return fig

def main():
    # Set dark theme
    st.set_page_config(
        page_title="Prediksi Konsentrasi Skripsi PTIK",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # CSS untuk dark theme
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        .stTextArea > div > div > textarea {
            background-color: #262730;
            color: white;
        }
        .stButton button {
            background-color: #00ABB3;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            border-radius: 5px;
        }
        .stButton button:hover {
            background-color: #008B93;
        }
        h1, h2, h3 {
            color: white !important;
        }
        .plot-container {
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load models
    model, vectorizer, le = load_models()
    
    # Title and description
    st.title("🎓 Prediksi Konsentrasi Skripsi PTIK")
    st.markdown("""
        <div style='background-color: #262730; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        Sistem prediksi konsentrasi skripsi menggunakan algoritma Random Forest untuk menentukan konsentrasi
        yang paling sesuai berdasarkan judul yang diajukan. Model telah dilatih menggunakan data historis
        skripsi mahasiswa PTIK. - By Diego
        </div>
    """, unsafe_allow_html=True)
    
    # Input
    title = st.text_area(
        "Masukkan Judul Skripsi",
        height=100,
        placeholder="Contoh: Pengembangan Sistem Informasi Manajemen Berbasis Web"
    )
    
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        predict_button = st.button(
            "🔍 Prediksi Konsentrasi",
            type="primary",
            use_container_width=True,
        )
    
    if predict_button and title.strip():
        try:
            with st.spinner('Memproses prediksi...'):
                # Vectorize input
                vector = vectorizer.transform([title])
                
                # Get prediction dan probabilities
                prediction = model.predict(vector)
                probabilities = model.predict_proba(vector)[0]
                
                predicted_class = le.inverse_transform(prediction)[0]
                
                # Tampilkan hasil
                st.markdown("""
                    <div style='background-color: #262730; padding: 30px; border-radius: 10px; margin: 30px 0;'>
                        <h2>Hasil Prediksi</h2>
                        <h3 style='color: #00ABB3 !important;'>Konsentrasi yang Diprediksi:</h3>
                        <h1 style='color: #00ABB3 !important; font-size: 48px;'>{}</h1>
                    </div>
                """.format(predicted_class), unsafe_allow_html=True)
                
                # Confidence scores
                st.markdown("<h3>Confidence Scores</h3>", unsafe_allow_html=True)
                
                # Kolom Kelas
                cols = st.columns(len(le.classes_))
                
                # Charts
                for i, (label, prob) in enumerate(zip(le.classes_, probabilities)):
                    with cols[i]:
                        fig = create_gauge_chart(prob, label)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Kata Kunci
                st.markdown("<h3>Analisis Kata Kunci</h3>", unsafe_allow_html=True)
                
                feature_names = vectorizer.get_feature_names_out()
                important_words = pd.DataFrame({
                    'Kata': feature_names,
                    'Bobot': vector.toarray()[0]
                })
                important_words = important_words[important_words['Bobot'] > 0]
                important_words = important_words.sort_values('Bobot', ascending=False)
                
                if not important_words.empty:
                    chart_data = important_words.head(10)
                    fig = go.Figure(go.Bar(
                        x=chart_data['Kata'],
                        y=chart_data['Bobot'],
                        marker_color='#00ABB3'
                    ))
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "white"},
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis={'showgrid': False},
                        yaxis={'showgrid': True, 'gridcolor': '#333'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        # Error 
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {str(e)}")
            
    # Error jika form kosong
    elif predict_button:
        st.warning("⚠️ Mohon masukkan judul skripsi terlebih dahulu!")

if __name__ == "__main__":
    main()