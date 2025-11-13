# =============================================================================
# DASHBOARD STREAMLIT - TAREFA 3 (BÔNUS INOVAÇÃO) - VERSÃO FINAL COM DADOS DO ALUNO
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay, log_loss
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Hotel Cancellation Predictor", layout="wide")

# Título principal do dashboard
st.title("🏨 Modelo de Cancelamento de Reservas")
st.subheader("Tarefa 3 - Sistemas de Informação em Engenharia de Produção")
st.write("Ajuste os parâmetros na barra lateral e compare os modelos de Machine Learning.")

# --- Informações do Aluno e Disciplina na Sidebar (discreto) ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Aluno:** Maria Eduarda Silveira Cintra")
st.sidebar.markdown("**Matrícula:** 190092718")
st.sidebar.markdown("**Disciplina:** Sistemas de Informação em Engenharia de Produção")
st.sidebar.markdown("**Professor:** João Gabriel de Moraes Souza")
st.sidebar.markdown("---")

st.sidebar.header("🎛️ Configurações do Modelo")

# Modo de Operação (para o futuro, mantemos como "Modelo Único" por enquanto)
st.sidebar.subheader("🎮 Modo de Operação")
modo_operacao = st.sidebar.radio("Escolha o modo:", ["Modelo Único"]) #, "Comparação de Modelos"])
st.sidebar.markdown("---")

# Seleção do Modelo
st.sidebar.subheader("📊 Seleção do Modelo")
modelo_escolhido = st.sidebar.selectbox("Escolha o algoritmo:", 
                                       ["Regressão Logística", "KNN", "Árvore de Decisão", "SVM (RBF)"])
st.sidebar.markdown("---")

# Ajuste de Parâmetros
st.sidebar.subheader("🎛️ Ajuste de Parâmetros")
if modelo_escolhido == "Regressão Logística":
    C_lr = st.sidebar.slider("C (Regularização):", 0.01, 10.0, 1.0, step=0.01)
    max_iter_lr = st.sidebar.slider("Máximo de Iterações:", 100, 2000, 1000, step=100)
elif modelo_escolhido == "KNN":
    k_knn = st.sidebar.slider("Valor de K (Vizinhos):", 3, 21, 5, step=2)
elif modelo_escolhido == "Árvore de Decisão":
    max_depth_tree = st.sidebar.slider("Profundidade Máxima da Árvore:", 3, 15, 6, step=1)
else: # SVM (RBF)
    C_svm = st.sidebar.slider("C (Regularização SVM):", 0.1, 10.0, 1.0, step=0.1)
    gamma_svm = st.sidebar.selectbox("Gamma (SVM):", ["scale", "auto", 0.1, 1])
st.sidebar.markdown("---")

# Configurações Gerais dos Dados
st.sidebar.subheader("🔧 Configurações Gerais dos Dados")
test_size_perc = st.sidebar.slider("Tamanho do Teste (%):", 10, 50, 20, step=5) / 100.0
train_sample_frac = st.sidebar.slider("Tamanho da Amostra de Treino (%):", 10, 100, 100, step=10) / 100.0
st.sidebar.markdown("---")

# --- Funções de Carregamento e Pré-processamento de Dados ---
@st.cache_data
def load_and_prepare_data(test_size_perc, train_sample_frac):
    df = pd.read_csv('hotel_bookings.csv')

    # Limpeza básica
    df = df.dropna(subset=['children'])
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)

    # Remover vazamento de dados
    vazamento = [col for col in df.columns if 'reservation_status' in col]
    df = df.drop(columns=vazamento, errors='ignore')

    # Selecionar variáveis (as mesmas do notebook)
    original_features = ['hotel', 'lead_time', 'arrival_date_month', 'stays_in_weekend_nights',
                         'stays_in_week_nights', 'adults', 'children', 'market_segment',
                         'distribution_channel', 'customer_type', 'adr', 'total_of_special_requests']

    df_modelo = df[original_features + ['is_canceled']].copy()

    # Codificar categóricas
    df_encoded = pd.get_dummies(df_modelo, 
                                columns=['hotel', 'arrival_date_month', 'market_segment',
                                        'distribution_channel', 'customer_type'],
                                drop_first=True)

    # Lista das 10 variáveis escolhidas pelo RFE no notebook principal
    variaveis_rfe = [
        'total_of_special_requests', 'market_segment_Complementary', 'market_segment_Corporate', 
        'market_segment_Direct', 'market_segment_Groups', 'market_segment_Offline TA/TO', 
        'market_segment_Online TA', 'distribution_channel_GDS', 'distribution_channel_TA/TO', 
        'customer_type_Transient'
    ]

    # Adicionar colunas que podem ter sido perdidas no get_dummies se não houver ocorrências no dataset
    for col in variaveis_rfe:
        if col not in df_encoded.columns:
            df_encoded[col] = 0 

    # Filtrar apenas as variáveis que foram selecionadas via RFE no notebook
    X = df_encoded[variaveis_rfe]
    y = df_encoded['is_canceled']

    # Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_perc, 
                                                        random_state=42, stratify=y)

    # Normalizar
    colunas_num_para_escalar = ['total_of_special_requests'] 
    scaler = StandardScaler()
    X_train[colunas_num_para_escalar] = scaler.fit_transform(X_train[colunas_num_para_escalar])
    X_test[colunas_num_para_escalar] = scaler.transform(X_test[colunas_num_para_escalar])

    # SMOTE para balancear o conjunto de treino
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # Amostragem do conjunto de treino balanceado
    if train_sample_frac < 1.0:
        X_train_bal, _, y_train_bal, _ = train_test_split(X_train_bal, y_train_bal, 
                                                          train_size=train_sample_frac, 
                                                          random_state=42, stratify=y_train_bal)

    return X_train_bal, X_test, y_train_bal, y_test

# --- Carregar e Preparar Dados ---
try:
    X_train, X_test, y_train, y_test = load_and_prepare_data(test_size_perc, train_sample_frac)
    st.success(f"✅ Dados carregados: {len(X_train):,} amostras de treino (balanceadas), {len(X_test):,} de teste.")
except Exception as e:
    st.error(f"❌ Erro ao carregar ou preparar dados. Verifique o arquivo 'hotel_bookings.csv' e o código: {e}")
    st.stop()

# --- Botão para Treinar e Avaliar ---
if st.button("🚀 Treinar e Avaliar Modelo"):
    with st.spinner(f"Treinando {modelo_escolhido}..."):
        # Instanciar o modelo com os parâmetros selecionados
        if modelo_escolhido == "Regressão Logística":
            modelo = LogisticRegression(max_iter=max_iter_lr, C=C_lr, random_state=42)
        elif modelo_escolhido == "KNN":
            modelo = KNeighborsClassifier(n_neighbors=k_knn)
        elif modelo_escolhido == "Árvore de Decisão":
            modelo = DecisionTreeClassifier(max_depth=max_depth_tree, random_state=42)
        else: # SVM (RBF)
            modelo = SVC(kernel="rbf", probability=True, C=C_svm, gamma=gamma_svm, random_state=42)

        # Treinar
        modelo.fit(X_train, y_train)

        # Predição
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

        # --- Exibir Métricas ---
        st.subheader(f"📊 Métricas de Desempenho: {modelo_escolhido}")
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        col_m1.metric("Acurácia", f"{acc:.4f}")
        col_m2.metric("Precisão", f"{prec:.4f}")
        col_m3.metric("Recall", f"{rec:.4f}")
        col_m4.metric("F1-Score", f"{f1:.4f}")

        if y_prob is not None:
            auc = roc_auc_score(y_test, y_prob)
            logloss = log_loss(y_test, y_prob)
            col_m5.metric("AUC", f"{auc:.4f}")
            st.write(f"**Log Loss:** {logloss:.4f}")
        else:
            st.write("AUC e Log Loss não disponíveis para este modelo sem probabilidades.")

        # --- Plot Curva ROC ---
        if y_prob is not None:
            st.subheader("📈 Curva ROC")
            fig, ax = plt.subplots(figsize=(8, 6))
            RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
            ax.set_title(f"Curva ROC - {modelo_escolhido} (AUC: {auc:.4f})")
            st.pyplot(fig)

        st.success("✅ Modelo treinado e avaliado com sucesso!")

st.markdown("---")
st.info("💡 **Dica:** Altere os parâmetros na barra lateral e clique em 'Treinar e Avaliar Modelo' para ver o impacto no desempenho.")

# --- Confirmação sobre o simulador de impacto financeiro ---
st.markdown("---")
st.subheader("Confirmação de Requisitos")
st.write("Conforme solicitado, o dashboard inclui:")
st.markdown("""
- **Modo de Operação**: Atualmente em "Modelo Único" para focar na experimentação individual.
- **Seleção do Modelo**: Permite escolher entre Regressão Logística, KNN, Árvore de Decisão e SVM.
- **Ajuste de Parâmetros**: Sliders para `C`, `Máximo de Iterações` (RL), `K` (KNN), `Profundidade Máxima` (Árvore), `C` e `Gamma` (SVM).
- **Configurações Gerais**: Sliders para `Tamanho do Teste (%)` e `Tamanho da Amostra de Treino (%)`.
- **Simulador de Impacto Financeiro**: **Não incluído**, conforme sua preferência.
""")
