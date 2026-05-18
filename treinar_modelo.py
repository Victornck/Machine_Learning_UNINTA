import pandas as pd
import numpy as np
import joblib
import json
import os
import logging

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

# ==========================================
# CONFIGURACOES
# ==========================================

CSV_PATH = 'dataset_casas.csv'

MODEL_PATH = 'modelo_preco_casas.pkl'

METRICAS_PATH = 'metricas_modelo.json'

RANDOM_STATE = 42

# ==========================================
# CARREGAR DATASET
# ==========================================

if not os.path.exists(CSV_PATH):

    raise FileNotFoundError(
        f"Arquivo '{CSV_PATH}' nao encontrado."
    )

base_dados = pd.read_csv(CSV_PATH)

logging.info(
    f'Dataset carregado com '
    f'{len(base_dados)} registros.'
)

# ==========================================
# VALIDACOES
# ==========================================

COLUNAS_ENTRADA = [
    'tamanho',
    'quartos',
    'banheiros',
    'idade',
    'garagem',
    'localizacao'
]

COLUNA_SAIDA = 'preco'

colunas_faltando = [

    c for c in (
        COLUNAS_ENTRADA +
        [COLUNA_SAIDA]
    )

    if c not in base_dados.columns
]

if colunas_faltando:

    raise ValueError(
        f'Colunas ausentes: '
        f'{colunas_faltando}'
    )

logging.info('Verificando valores nulos...')

print(base_dados.isnull().sum())


base_dados = base_dados.dropna()


base_dados = base_dados[
    base_dados['preco'] > 0
]

# ==========================================
# FEATURE ENGINEERING
# ==========================================


base_dados['banheiros_por_quarto'] = (
    base_dados['banheiros'] /
    base_dados['quartos']
)


base_dados['idade_categoria'] = pd.cut(
    base_dados['idade'],
    bins=[-1, 5, 15, 100],
    labels=[
        'nova',
        'media',
        'antiga'
    ]
)

# ==========================================
# ENTRADAS E SAIDA
# ==========================================

X = base_dados[[
    'tamanho',
    'quartos',
    'banheiros',
    'idade',
    'garagem',
    'localizacao',
    'banheiros_por_quarto',
    'idade_categoria'
]]


y = np.log(
    base_dados[COLUNA_SAIDA]
)

# ==========================================
# TREINO / TESTE
# ==========================================

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

logging.info(
    f'Treino: {len(X_treino)} | '
    f'Teste: {len(X_teste)}'
)

# ==========================================
# FEATURES
# ==========================================

features_numericas = [
    'tamanho',
    'quartos',
    'banheiros',
    'idade',
    'garagem',
    'banheiros_por_quarto'
]

features_categoricas = [
    'localizacao',
    'idade_categoria'
]

# ==========================================
# PREPROCESSAMENTO
# ==========================================

preprocessador = ColumnTransformer([

    (
        'num',
        StandardScaler(),
        features_numericas
    ),

    (
        'cat',
        OneHotEncoder(
            handle_unknown='ignore'
        ),
        features_categoricas
    )
])

# ==========================================
# MODELO
# ==========================================

modelo = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE
)

# ==========================================
# PIPELINE
# ==========================================

pipeline = Pipeline([

    ('prep', preprocessador),

    ('modelo', modelo)
])

# ==========================================
# VALIDACAO CRUZADA
# ==========================================

logging.info(
    'Executando validacao cruzada...'
)

scores_r2 = cross_val_score(
    pipeline,
    X_treino,
    y_treino,
    cv=5,
    scoring='r2'
)

scores_mae = -cross_val_score(
    pipeline,
    X_treino,
    y_treino,
    cv=5,
    scoring='neg_mean_absolute_error'
)

print('\n-- VALIDACAO CRUZADA --')

print(
    f'R2 medio : '
    f'{scores_r2.mean():.4f} '
    f'(+/- {scores_r2.std():.4f})'
)

print(
    f'MAE medio : '
    f'{scores_mae.mean():.4f}'
)

# ==========================================
# TREINAMENTO FINAL
# ==========================================

logging.info(
    'Treinando modelo final...'
)

pipeline.fit(
    X_treino,
    y_treino
)

# ==========================================
# PREVISOES
# ==========================================

previsoes_log = pipeline.predict(
    X_teste
)

previsoes = np.exp(
    previsoes_log
)

y_teste_real = np.exp(
    y_teste
)

# impedir negativos

previsoes = np.maximum(
    previsoes,
    50000
)

# ==========================================
# METRICAS
# ==========================================

mae = mean_absolute_error(
    y_teste_real,
    previsoes
)

rmse = np.sqrt(
    mean_squared_error(
        y_teste_real,
        previsoes
    )
)

r2 = r2_score(
    y_teste_real,
    previsoes
)

print('\n-- METRICAS FINAIS --')

print(
    f'MAE  : R$ {mae:,.2f}'
)

print(
    f'RMSE : R$ {rmse:,.2f}'
)

print(
    f'R2   : {r2:.4f}'
)

# ==========================================
# IMPORTANCIA DAS FEATURES
# ==========================================

modelo_rf = pipeline.named_steps['modelo']

importancias = (
    modelo_rf.feature_importances_
)

nomes_features = (

    features_numericas +

    list(
        pipeline
        .named_steps['prep']
        .named_transformers_['cat']
        .get_feature_names_out(
            features_categoricas
        )
    )
)

print('\n-- IMPORTANCIA DAS FEATURES --')

for nome, importancia in sorted(
    zip(nomes_features, importancias),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f'{nome:>30} : '
        f'{importancia:.4f}'
    )

# ==========================================
# SALVAR METRICAS
# ==========================================

metricas = {

    'mae': float(mae),

    'rmse': float(rmse),

    'r2': float(r2),

    'dataset_size': int(
        len(base_dados)
    ),

    'modelo': 'RandomForestRegressor'
}

with open(
    METRICAS_PATH,
    'w',
    encoding='utf-8'
) as f:

    json.dump(
        metricas,
        f,
        indent=4,
        ensure_ascii=False
    )

logging.info(
    f'Metricas salvas em '
    f'{METRICAS_PATH}'
)

# ==========================================
# SALVAR MODELO
# ==========================================

joblib.dump(
    pipeline,
    MODEL_PATH
)

logging.info(
    f'Modelo salvo em '
    f'{MODEL_PATH}'
)

# ==========================================
# FINAL
# ==========================================

print('\nTreinamento concluido com sucesso.')