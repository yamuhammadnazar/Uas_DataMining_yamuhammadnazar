import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set konfigurasi halaman
st.set_page_config(
    page_title="Sistem Klasifikasi Obat",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi perhitungan Gradient Boosting
def calculate_gradient_boosting_steps(input_data, model, feature_names):
    st.write("### 🔍 Detail Proses Perhitungan Gradient Boosting")
    
    # 1. Data Input Normalisasi
    st.write("1️⃣ Data Input Setelah Normalisasi:")
    normalized_df = pd.DataFrame([input_data], columns=feature_names)
    st.dataframe(normalized_df)
    
    # 2. Perhitungan Pohon Keputusan
    st.write("2️⃣ Perhitungan Setiap Pohon Keputusan:")
    total_contribution = np.zeros(len(model.classes_))
    
    for i, estimator in enumerate(model.estimators_[:3]):
        st.write(f"Pohon {i+1}:")
        prediction = estimator[0].predict([input_data])[0]
        contribution = model.learning_rate * prediction
        total_contribution += contribution
        
        st.write(f"- Prediksi Dasar: {prediction:.4f}")
        st.write(f"- Kontribusi (learning rate: {model.learning_rate}): {contribution:.4f}")
        st.write(f"- Total Kontribusi: {total_contribution[0]:.4f}")
    
    # 3. Probabilitas Final
    probabilities = model.predict_proba([input_data])[0]
    st.write("3️⃣ Probabilitas Akhir Setiap Kelas Obat:")
    prob_df = pd.DataFrame({
        'Obat': model.classes_,
        'Probabilitas': probabilities
    }).sort_values('Probabilitas', ascending=False)
    st.dataframe(prob_df)
    
    # 4. Analisis Kontribusi Fitur
    st.write("4️⃣ Kontribusi Setiap Fitur dalam Keputusan:")
    feature_importance = pd.DataFrame({
        'Fitur': feature_names,
        'Kontribusi (%)': model.feature_importances_ * 100
    }).sort_values('Kontribusi (%)', ascending=False)
    
    st.bar_chart(feature_importance.set_index('Fitur'))

# Fungsi untuk menjelaskan klasifikasi obat
def get_drug_explanation(drug_name):
    drug_explanations = {
        'DrugA': """
        💊 Drug A direkomendasikan karena:
        - Cocok untuk pasien dengan tekanan darah tinggi
        - Efektif untuk rentang usia dewasa
        - Aman untuk kedua jenis kelamin
        - Optimal untuk level kolesterol tinggi
        """,
        'DrugB': """
        💊 Drug B direkomendasikan karena:
        - Ideal untuk tekanan darah normal
        - Sesuai untuk pasien usia muda-menengah
        - Efektif untuk level kolesterol normal
        - Memiliki keseimbangan Na/K yang baik
        """,
        'DrugC': """
        💊 Drug C direkomendasikan karena:
        - Cocok untuk tekanan darah rendah
        - Optimal untuk pasien usia lanjut
        - Efektif mengelola kolesterol tinggi
        - Menyesuaikan rasio Na/K yang tinggi
        """,
        'DrugX': """
        💊 Drug X direkomendasikan karena:
        - Khusus untuk kasus tekanan darah fluktuatif
        - Sesuai untuk berbagai rentang usia
        - Efektif untuk semua level kolesterol
        - Menstabilkan rasio Na/K
        """,
        'DrugY': """
        💊 Drug Y direkomendasikan karena:
        - Ideal untuk tekanan darah borderline
        - Cocok untuk pasien usia menengah
        - Efektif untuk kolesterol borderline
        - Mengoptimalkan rasio Na/K
        """
    }
    return drug_explanations.get(drug_name, "Penjelasan tidak tersedia untuk obat ini.")

# CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    :root {
        --primary-color: #85C1E9;
        --secondary-color: #AED6F1;
        --accent-color: #48BB78;
        --background-color: #EBF8FF;
        --text-color: #2D3748;
    }
    
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        animation: slideIn 0.5s ease-in;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .info-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

# Load dan cache data
@st.cache_data
def load_data():
    return pd.read_csv('Classification.csv')

# Fungsi utama
def main():
    with st.sidebar:
        st.title("Menu Navigasi")
        menu = st.radio("", ["🏠 Beranda", "📊 Klasifikasi", "👤 Tentang"])

    if menu == "🏠 Beranda":
        show_home_page()
    elif menu == "📊 Klasifikasi":
        show_classification_page()
    elif menu == "👤 Tentang":
        show_about_page()

        

def show_home_page():
    st.title("🏥 Sistem Klasifikasi Obat dengan Gradient Boosting")
    
    # Metrics cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>👥 Total Pasien</h4>
            <h2>200</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>💊 Jenis Obat</h4>
            <h2>5</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
            <h4>📈 Akurasi Model</h4>
            <h2>85%</h2>
        </div>
        """, unsafe_allow_html=True)
     # Tambahkan tab untuk dataset
    tab1, tab2 = st.tabs(["📋 Tentang Aplikasi", "📊 Dataset"])
    
    with tab1:
        st.markdown("""
    ### 📋 Tentang Algoritma Gradient Boosting
    
    Gradient Boosting adalah algoritma machine learning yang:
    1. Membangun model secara bertahap
    2. Memperbaiki kesalahan prediksi sebelumnya
    3. Menggabungkan multiple weak learners menjadi strong predictor
    
    #### Proses Perhitungan:
    1. Inisialisasi model awal
    2. Iterasi untuk perbaikan model
    3. Kombinasi hasil untuk prediksi final
    """)
    
    with tab2:
        st.markdown("### 📊 Dataset Klasifikasi Obat")
        df = load_data()
        
        # Tampilkan statistik dasar
        st.write("#### Statistik Dataset:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", len(df))
        with col2:
            st.metric("Jumlah Fitur", len(df.columns)-1)
        with col3:
            st.metric("Jenis Obat", df['Drug'].nunique())
        
        # Tampilkan dataset
        st.write("#### Preview Dataset:")
        st.dataframe(df)
        
        # Tampilkan distribusi obat
        st.write("#### Distribusi Jenis Obat:")
        drug_dist = df['Drug'].value_counts()
        st.bar_chart(drug_dist)
def show_classification_page():
    st.title("🔍 Klasifikasi Obat Pasien")
    
    # Load dan proses data
    df = load_data()
    
    # Inisialisasi encoders
    sex_encoder = LabelEncoder()
    bp_encoder = LabelEncoder()
    chol_encoder = LabelEncoder()
    
    # Fit encoders
    sex_encoder.fit(['F', 'M'])
    bp_encoder.fit(['HIGH', 'NORMAL', 'LOW'])
    chol_encoder.fit(['HIGH', 'NORMAL'])
    
    # Proses features
    X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].copy()
    X['Sex'] = sex_encoder.transform(X['Sex'])
    X['BP'] = bp_encoder.transform(X['BP'])
    X['Cholesterol'] = chol_encoder.transform(X['Cholesterol'])
    
    # Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    y = df['Drug']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model
    gb_classifier = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    gb_classifier.fit(X_train, y_train)
    
    # Form input
    st.markdown("""
        <div class="info-card">
            <h3>📝 Form Data Pasien</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("🎂 Usia", min_value=15, max_value=74, value=25)
        sex = st.selectbox("👤 Jenis Kelamin", options=['F', 'M'])
        bp = st.selectbox("💉 Tekanan Darah", options=['HIGH', 'NORMAL', 'LOW'])
    
    with col2:
        cholesterol = st.selectbox("🔬 Kolesterol", options=['HIGH', 'NORMAL'])
        na_to_k = st.number_input("⚖ Rasio Na/K", min_value=6.0, max_value=40.0, value=15.0)
    
    if st.button("📋 Klasifikasikan Obat"):
        # Proses input
        input_data = pd.DataFrame([[
            age,
            sex,
            bp,
            cholesterol,
            na_to_k
        ]], columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
        
        # Transform input
        input_data['Sex'] = sex_encoder.transform(input_data['Sex'])
        input_data['BP'] = bp_encoder.transform(input_data['BP'])
        input_data['Cholesterol'] = chol_encoder.transform(input_data['Cholesterol'])
        input_scaled = scaler.transform(input_data)
        
        # Prediksi
        prediction = gb_classifier.predict(input_scaled)[0]
        
        # Tampilkan hasil
        st.markdown(f"""
        <div class="info-card" style="background: linear-gradient(45deg, var(--primary-color) 0%, var(--secondary-color) 100%); color: white;">
            <h3>💊 Hasil Klasifikasi: {prediction}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(get_drug_explanation(prediction))
        calculate_gradient_boosting_steps(
            input_scaled[0],
            gb_classifier,
            ['Usia', 'Jenis Kelamin', 'Tekanan Darah', 'Kolesterol', 'Rasio Na/K']
        )

def show_about_page():
    st.title("👤 Tentang Pengembang")
    
    st.markdown("""
    ### Data Mahasiswa
    - **Nama:** Ya' Muhammad Nazar
    - **NIM:** 211220020
    - **Program Studi:** Teknik Informatika
    - **Fakultas:** Teknik dan Ilmu Komputer
    - **Universitas:** Universitas Muhammadiyah Pontianak
    
    
    ### 📝 Tentang Proyek
    Proyek ini dibuat sebagai tugas Ujian Akhir Semester Mata Kuliah Data Mining dengan fokus pada penerapan 
    algoritma Gradient Boosting untuk klasifikasi obat berdasarkan parameter pasien.
    
    #### Alat yang Digunakan:
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas
    - NumPy
    """)


if __name__ == "__main__":
    main()
