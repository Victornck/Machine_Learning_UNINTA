````md
# 🏠 IA de Previsão de Preço de Casas

Sistema de Machine Learning para previsão de preços de imóveis utilizando Python, Scikit-Learn e Streamlit.

O projeto conta com:

- Interface web moderna com Streamlit
- Pipeline profissional de Machine Learning
- Feature Engineering
- Random Forest Regressor
- Pré-processamento automático
- Validação cruzada
- Métricas de avaliação
- Interface CLI para testes rápidos

---

# 📸 Preview

## Interface Web

- Dashboard moderno
- Predição em tempo real
- Métricas do modelo
- Gráfico de importância das variáveis

---

# 📂 Estrutura do Projeto

```bash
projeto_casas/
│
├── app.py
├── treinar_modelo.py
├── prever_cli.py
├── dataset_casas.csv
├── requirements.txt
├── README.md
│
├── modelo_preco_casas.pkl
└── metricas_modelo.json
```
````

---

# 🚀 Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Matplotlib
- Joblib

---

# 🧠 Modelo de Machine Learning

O sistema utiliza:

## Random Forest Regressor

O modelo foi escolhido por:

- Melhor generalização
- Capacidade de aprender relações não-lineares
- Robustez contra overfitting
- Melhor interpretação de variáveis

---

# ⚙️ Features Utilizadas

| Feature              | Descrição                    |
| -------------------- | ---------------------------- |
| tamanho              | Área do imóvel em m²         |
| quartos              | Quantidade de quartos        |
| banheiros            | Quantidade de banheiros      |
| idade                | Idade do imóvel              |
| garagem              | Quantidade de vagas          |
| localizacao          | Região do imóvel             |
| banheiros_por_quarto | Engenharia de feature        |
| idade_categoria      | Imóvel novo, médio ou antigo |

---

# 🏗️ Pipeline de Machine Learning

O projeto utiliza um pipeline completo com:

- StandardScaler
- OneHotEncoder
- ColumnTransformer
- RandomForestRegressor

---

# 📊 Avaliação do Modelo

O treinamento utiliza:

- Train/Test Split
- Cross Validation (5-Fold)
- MAE
- RMSE
- R² Score

## Métricas Esperadas

| Métrica | Valor esperado |
| ------- | -------------- |
| R²      | > 0.90         |
| MAE     | ~R$ 40.000     |
| RMSE    | ~R$ 60.000     |

---

# 📦 Instalação

## 1. Clone o repositório

```bash
git clone https://github.com/seuusuario/projeto_casas.git
```

---

## 2. Acesse a pasta

```bash
cd projeto_casas
```

---

## 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

# ▶️ Como Executar

## Gerar dataset

```bash
python gerar_dataset.py
```

---

## Treinar o modelo

```bash
python treinar_modelo.py
```

---

## Executar interface web

```bash
streamlit run app.py
```

---

## Executar interface CLI

```bash
python prever_cli.py
```

---

# 💡 Funcionalidades

## Interface Web

- Predição de preços em tempo real
- Interface moderna responsiva
- Dashboard visual
- Faixa de preço estimada
- Importância das variáveis
- Métricas automáticas

---

## Inteligência Artificial

- Prevenção de preços negativos
- Relações não-lineares
- Engenharia de features
- Tratamento de categorias
- Pré-processamento automático

---

# 📈 Melhorias Futuras

- Deploy online
- Integração com API
- Banco de dados
- Mapa geográfico
- Upload de datasets
- Comparação entre imóveis
- Histórico de previsões
- Exportação PDF
- SHAP Values
- XGBoost

---

# 👨‍💻 Autor

Projeto desenvolvido para fins acadêmicos, portfólio e estudos de Machine Learning aplicado ao mercado imobiliário.

```

```
