# =============================================================================
# DASHBOARD STREAMLIT - TAREFA 3 (BÔNUS INOVAÇÃO)
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Hotel Cancellation Predictor", layout="wide")
st.title("🏨 Modelo de Cancelamento de Reservas - Tarefa 3 (UnB-FT/EPR)")
st.write("**Ajuste os parâmetros abaixo e compare os modelos de Machine Learning.**")

# Carregar e preparar dados
@st.cache_data
def load_and_prepare_data():
    # Substitua pelo caminho correto do seu CSV
    df = pd.read_csv('hotel_bookings.csv')  # Já está correto se o arquivo estiver na mesma pasta
    
    # Limpeza básica
    df = df.dropna(subset=['children'])
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)
    
    # Remover vazamento de dados
    vazamento = [col for col in df.columns if 'reservation_status' in col]
    df = df.drop(columns=vazamento, errors='ignore')
    
    # Selecionar variáveis
    variaveis = ['hotel', 'lead_time', 'arrival_date_month', 'stays_in_weekend_nights',
                 'stays_in_week_nights', 'adults', 'children', 'market_segment',
                 'distribution_channel', 'customer_type', 'adr', 'total_of_special_requests']
    
    df_modelo = df[variaveis + ['is_canceled']].copy()
    
    # Codificar categóricas
    df_encoded = pd.get_dummies(df_modelo, 
                                columns=['hotel', 'arrival_date_month', 'market_segment',
                                        'distribution_channel', 'customer_type'],
                                drop_first=True)
    
    # Separar X e y
    X = df_encoded.drop('is_canceled', axis=1)
    y = df_encoded['is_canceled']
    
    # Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Normalizar
    colunas_num = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
                   'adults', 'children', 'adr', 'total_of_special_requests']
    scaler = StandardScaler()
    X_train[colunas_num] = scaler.fit_transform(X_train[colunas_num])
    X_test[colunas_num] = scaler.transform(X_test[colunas_num])
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    return X_train_bal, X_test, y_train_bal, y_test

# Carregar dados
try:
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    st.success(f"✅ Dados carregados: {len(X_train):,} amostras de treino, {len(X_test):,} de teste")
except Exception as e:
    st.error(f"❌ Erro ao carregar dados: {e}")
    st.stop()

# Interface de controle
col1, col2, col3 = st.columns(3)

with col1:
    modelo_escolhido = st.selectbox("🤖 Escolha o algoritmo:", 
                                   ["Regressão Logística", "KNN", "Árvore de Decisão"])

with col2:
    if modelo_escolhido == "KNN":
        k = st.slider("Valor de K:", 3, 15, 5, step=2)
    else:
        k = 5

with col3:
    C = st.slider("C (regularização):", 0.1, 10.0, 1.0)

# Treinar modelo
if st.button("🚀 Treinar e Avaliar Modelo"):
    with st.spinner("Treinando modelo..."):
        if modelo_escolhido == "Regressão Logística":
            modelo = LogisticRegression(max_iter=1000, C=C, random_state=42)
        elif modelo_escolhido == "KNN":
            modelo = KNeighborsClassifier(n_neighbors=k)
        else:
            modelo = DecisionTreeClassifier(max_depth=6, random_state=42)
        
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None
        
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        
        # Exibir resultados
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Acurácia", f"{acc:.4f}")
        col_m2.metric("Precisão", f"{prec:.4f}")
        col_m3.metric("Recall", f"{rec:.4f}")
        col_m4.metric("F1-Score", f"{f1:.4f}")
        
        if auc:
            st.metric("AUC", f"{auc:.4f}")
        
        # Curva ROC
        if y_prob is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
            ax.set_title(f"Curva ROC - {modelo_escolhido}")
            st.pyplot(fig)
        
        st.success("✅ Modelo treinado com sucesso!")

st.markdown("---")
st.info("💡 **Dica:** Experimente diferentes valores de K (para KNN) ou C (para regularização) e observe o impacto nas métricas.")
