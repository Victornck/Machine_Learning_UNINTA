import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="Previsão de Casas",
    page_icon=None,
    layout="centered"
)

# ==========================================
# CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0a0f1e;
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

.stNumberInput input, .stSelectbox select {
    background-color: #111827 !important;
    color: white !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}

.stNumberInput label, .stSelectbox label {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
}

[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e293b;
    padding: 18px 20px;
    border-radius: 16px;
    text-align: center;
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
}

[data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 22px !important;
}

.stButton > button {
    width: 100%;
    height: 56px;
    border: none;
    border-radius: 14px;
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.04em;
    transition: all 0.2s ease;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.35);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    box-shadow: 0 6px 28px rgba(59, 130, 246, 0.5);
    transform: translateY(-1px);
}

.bloco {
    background: #111827;
    padding: 32px;
    border-radius: 22px;
    border: 1px solid #1e293b;
    margin-bottom: 28px;
}

.stApp h2, .stApp h3 {
    color: white !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}

.stAlert {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #94a3b8 !important;
}

hr {
    border-color: #1e293b !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "modelo_preco_casas.pkl"
METRICAS_PATH = "metricas_modelo.json"

@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)

modelo = carregar_modelo()

# ==========================================
# SESSION STATE
# ==========================================

if "resultado" not in st.session_state:
    st.session_state.resultado = None

# ==========================================
# HEADER
# ==========================================

components.html("""
<div style="
    background: #111827;
    padding: 40px 32px;
    border-radius: 22px;
    border: 1px solid #1e293b;
    margin-bottom: 28px;
    text-align: center;
    font-family: 'DM Sans', sans-serif;
">

    <h1 style="
        font-size: 30px;
        font-weight: 800;
        margin: 0 0 10px 0;
        letter-spacing: -0.02em;
        color: white;
    ">
        Previsão de Preço de Casas
    </h1>

    <p style="
        color: #64748b;
        font-size: 15px;
        margin: 0;
    ">
        Estime o valor de mercado de um imóvel com Machine Learning
    </p>

</div>
""", height=170)

# ==========================================
# FORM
# ==========================================

st.markdown('<div class="bloco">', unsafe_allow_html=True)

st.markdown("### Dados do Imóvel")
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    tamanho = st.number_input(
        "Tamanho (m²)",
        min_value=20.0,
        max_value=1000.0,
        value=100.0
    )

    quartos = st.number_input(
        "Quartos",
        min_value=1,
        max_value=10,
        value=3
    )

    garagem = st.number_input(
        "Vagas de Garagem",
        min_value=0,
        max_value=10,
        value=2
    )

with col2:
    banheiros = st.number_input(
        "Banheiros",
        min_value=1,
        max_value=10,
        value=2
    )

    idade = st.number_input(
        "Idade do Imóvel (anos)",
        min_value=0,
        max_value=100,
        value=5
    )

    localizacao = st.selectbox(
        "Localização",
        [0, 1, 2],
        format_func=lambda x: {
            0: "Periferia",
            1: "Zona Intermediária",
            2: "Centro / Bairro Nobre"
        }[x]
    )

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# BUTTON
# ==========================================

if st.button("Estimar Preço"):

    banheiros_por_quarto = banheiros / quartos

    if idade <= 5:
        idade_categoria = "nova"
    elif idade <= 15:
        idade_categoria = "media"
    else:
        idade_categoria = "antiga"

    dados = pd.DataFrame([{
        "tamanho": tamanho,
        "quartos": quartos,
        "banheiros": banheiros,
        "idade": idade,
        "garagem": garagem,
        "localizacao": localizacao,
        "banheiros_por_quarto": banheiros_por_quarto,
        "idade_categoria": idade_categoria
    }])

    preco_log = modelo.predict(dados)[0]

    preco = float(np.exp(preco_log))
    preco = max(preco, 50000)

    with open(METRICAS_PATH, "r") as f:
        metricas = json.load(f)

    rmse = metricas["rmse"]

    st.session_state.resultado = {
        "preco": preco,
        "preco_min": max(preco - rmse, 0),
        "preco_max": preco + rmse,
        "metricas": metricas,
        "importancias": modelo.named_steps["modelo"].feature_importances_.tolist()
    }

# ==========================================
# RESULTADO
# ==========================================

if st.session_state.resultado is not None:

    r = st.session_state.resultado

    preco = r["preco"]
    preco_min = r["preco_min"]
    preco_max = r["preco_max"]
    metricas = r["metricas"]
    importancias = np.array(r["importancias"])

    # ==========================================
    # CARD PRINCIPAL
    # ==========================================

    components.html(f"""
    <div style="
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        padding: 40px 32px;
        border-radius: 22px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(37, 99, 235, 0.4);
        margin: 8px 0 28px 0;
        font-family: 'DM Sans', sans-serif;
    ">
        <p style="
            color: #bfdbfe;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin: 0 0 12px 0;
        ">
            VALOR ESTIMADO
        </p>

        <p style="
            color: white;
            font-size: 52px;
            font-weight: 800;
            font-family: monospace;
            margin: 0 0 24px 0;
            letter-spacing: -0.02em;
            line-height: 1;
        ">
            R$ {preco:,.0f}
        </p>

        <div style="
            background: rgba(255,255,255,0.12);
            border-radius: 12px;
            padding: 16px 28px;
            display: inline-block;
        ">
            <p style="
                color: #bfdbfe;
                font-size: 11px;
                margin: 0 0 6px 0;
                font-weight: 700;
                letter-spacing: 0.1em;
                text-transform: uppercase;
            ">
                FAIXA ESTIMADA
            </p>

            <p style="
                color: white;
                font-size: 18px;
                font-weight: 700;
                margin: 0;
                font-family: monospace;
            ">
                R$ {preco_min:,.0f} — R$ {preco_max:,.0f}
            </p>
        </div>
    </div>
    """, height=260)

    # ==========================================
    # MÉTRICAS
    # ==========================================

    st.markdown("### Qualidade do Modelo")
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("R²", f"{metricas['r2']:.4f}")

    with c2:
        st.metric("MAE", f"R$ {metricas['mae']:,.0f}")

    with c3:
        st.metric("RMSE", f"R$ {metricas['rmse']:,.0f}")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    st.markdown("### Importância das Variáveis")
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    nomes = [
        "Tamanho",
        "Quartos",
        "Banheiros",
        "Idade",
        "Garagem",
        "Banheiros/Quarto",
        "Localização 0",
        "Localização 1",
        "Localização 2",
        "Idade Antiga",
        "Idade Média",
        "Idade Nova"
    ]

    idx = np.argsort(importancias)

    nomes_ord = [nomes[i] for i in idx]
    imp_ord = importancias[idx]

    max_imp = max(imp_ord)

    cores = [
        f"#{int(37 + (59 - 37) * (v / max_imp)):02x}"
        f"{int(99 + (130 - 99) * (v / max_imp)):02x}"
        f"{int(235 + (246 - 235) * (v / max_imp)):02x}"
        for v in imp_ord
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(
        nomes_ord,
        imp_ord,
        color=cores,
        height=0.65
    )

    for bar, val in zip(bars, imp_ord):
        ax.text(
            val + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            color="#94a3b8",
            fontsize=9,
            fontfamily="monospace"
        )

    ax.set_facecolor("#0a0f1e")
    fig.patch.set_facecolor("#111827")

    ax.tick_params(
        colors="#94a3b8",
        labelsize=10
    )

    ax.xaxis.set_major_formatter(
        mticker.FormatStrFormatter("%.2f")
    )

    ax.set_xlabel(
        "Importância",
        color="#64748b",
        fontsize=10
    )

    ax.set_xlim(0, max_imp * 1.18)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(
        axis="x",
        color="#1e293b",
        linewidth=0.8,
        linestyle="--"
    )

    ax.set_axisbelow(True)

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        "Modelo treinado com Random Forest Regressor — os valores refletem padrões aprendidos nos dados de treino."
    )