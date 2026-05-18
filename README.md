````md
# 🏠 IA de Previsão de Preço de Casas

<div align="center">

Sistema inteligente de previsão de preços imobiliários utilizando  
Machine Learning, Python e Streamlit.

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn">
<img src="https://img.shields.io/badge/Streamlit-Web_App-red?style=for-the-badge&logo=streamlit">
<img src="https://img.shields.io/badge/Status-Concluído-success?style=for-the-badge">

</div>

---

# 📌 Sobre o Projeto

Este projeto utiliza técnicas de **Machine Learning** para estimar o valor de imóveis com base em características como:

- Tamanho
- Quantidade de quartos
- Banheiros
- Idade do imóvel
- Garagem
- Localização

O sistema foi desenvolvido com foco em:

✅ Arquitetura profissional  
✅ Interface moderna  
✅ Pipeline completo de IA  
✅ Código organizado  
✅ Aplicação realista para portfólio

---

# ✨ Funcionalidades

## 🌐 Interface Web

- Dashboard moderno
- Predição em tempo real
- Interface responsiva
- Visual profissional
- Métricas automáticas
- Gráfico de importância das variáveis

---

## 🧠 Inteligência Artificial

- Random Forest Regressor
- Feature Engineering
- Pipeline profissional
- Tratamento automático de categorias
- Normalização de dados
- Validação cruzada
- Prevenção de previsões negativas

---

# 📸 Preview

## 🖥️ Interface Principal

> Adicione aqui screenshots do sistema futuramente.

---

# 🏗️ Estrutura do Projeto

```bash
projeto_casas/
│
├── app.py
├── treinar_modelo.py
├── prever_cli.py
├── gerar_dataset.py
│
├── dataset_casas.csv
├── modelo_preco_casas.pkl
├── metricas_modelo.json
│
├── requirements.txt
└── README.md
````

---

# 🚀 Tecnologias Utilizadas

<div align="center">

| Tecnologia   | Função                 |
| ------------ | ---------------------- |
| Python       | Linguagem principal    |
| Pandas       | Manipulação de dados   |
| NumPy        | Operações numéricas    |
| Scikit-Learn | Machine Learning       |
| Streamlit    | Interface Web          |
| Matplotlib   | Visualização           |
| Joblib       | Persistência do modelo |

</div>

---

# 🧠 Modelo de Machine Learning

## 🔹 Algoritmo Utilizado

### Random Forest Regressor

O modelo foi escolhido por possuir:

* Excelente generalização
* Baixo risco de overfitting
* Capacidade de aprender relações complexas
* Alta robustez
* Melhor desempenho em dados tabulares

---

# ⚙️ Features Utilizadas

| Feature              | Descrição               |
| -------------------- | ----------------------- |
| tamanho              | Área do imóvel em m²    |
| quartos              | Quantidade de quartos   |
| banheiros            | Quantidade de banheiros |
| idade                | Idade do imóvel         |
| garagem              | Quantidade de vagas     |
| localizacao          | Região do imóvel        |
| banheiros_por_quarto | Engenharia de feature   |
| idade_categoria      | Novo, médio ou antigo   |

---

# 🔄 Pipeline de Machine Learning

O projeto utiliza um pipeline completo com:

```python
StandardScaler
OneHotEncoder
ColumnTransformer
RandomForestRegressor
```

---

# 📊 Avaliação do Modelo

## Estratégias Utilizadas

* Train/Test Split
* Cross Validation (5-Fold)
* MAE
* RMSE
* R² Score

---

## 📈 Métricas Esperadas

| Métrica | Resultado  |
| ------- | ---------- |
| R²      | > 0.90     |
| MAE     | ~R$ 40.000 |
| RMSE    | ~R$ 60.000 |

---

# 📦 Instalação

## 1️⃣ Clonar o repositório

```bash
git clone https://github.com/Victornck/IA-de-Previs-o-de-Pre-o-de-Casas.git
```

---

## 2️⃣ Entrar na pasta

```bash
cd IA-de-Previs-o-de-Pre-o-de-Casas
```

---

## 3️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

---

# ▶️ Como Executar

## 🔹 Gerar Dataset

```bash
python gerar_dataset.py
```

---

## 🔹 Treinar Modelo

```bash
python treinar_modelo.py
```

---

## 🔹 Executar Aplicação Web

```bash
streamlit run app.py
```

---

## 🔹 Executar Interface CLI

```bash
python prever_cli.py
```

---

# 💻 Interface Web

A aplicação possui:

✅ Tema moderno
✅ Cards estilizados
✅ Responsividade
✅ Dashboard visual
✅ Predição dinâmica
✅ Métricas em tempo real

---

# 📈 Melhorias Futuras

* Deploy online
* Integração com API
* Banco de dados
* Comparação entre imóveis
* Histórico de previsões
* Exportação PDF
* SHAP Values
* XGBoost
* LightGBM
* Mapa geográfico

---

# 👨‍💻 Autor

<div align="center">

Desenvolvido por **Victor Berlinck**

Projeto voltado para:

🎓 Estudos acadêmicos
💼 Portfólio profissional
🧠 Aprendizado em Machine Learning

</div>

---

# ⭐ Considerações

Se este projeto te ajudou ou serviu de inspiração:

⭐ Deixe uma estrela no repositório.

```
```
