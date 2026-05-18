import pandas as pd
import numpy as np
from scipy import stats

np.random.seed(42)
n = 500

tamanho   = np.random.randint(40, 300, n).astype(float)
quartos   = np.clip(np.random.poisson(2.5, n) + 1, 1, 8).astype(int)
banheiros = np.clip(quartos - np.random.randint(0, 2, n), 1, quartos).astype(int)
idade     = np.random.randint(0, 50, n).astype(int)

garagem      = np.random.randint(0, 4, n).astype(int)   # 0 a 3 vagas
localizacao  = np.random.choice([1, 2, 3], n, p=[0.4, 0.4, 0.2])

bonus_localizacao = np.where(localizacao == 1, 0,
                    np.where(localizacao == 2, 40000, 100000))

ruido = np.random.normal(0, 25000, n)

preco = (
    tamanho   * 3500
    + quartos   * 15000
    + banheiros * 10000
    - idade     * 2000
    - (idade ** 1.4) * 400
    + garagem   * 20000
    + bonus_localizacao
    + 50000
    + ruido
).clip(60000)

preco = preco.round(-2)

df = pd.DataFrame({
    'tamanho':     tamanho,
    'quartos':     quartos,
    'banheiros':   banheiros,
    'idade':       idade,
    'garagem':     garagem,
    'localizacao': localizacao,
    'preco':       preco.astype(int)
})

z = np.abs(stats.zscore(df[['tamanho', 'preco']]))
df = df[(z < 3).all(axis=1)].reset_index(drop=True)

df.to_csv('dataset_casas.csv', index=False)
print(f'Dataset gerado com {len(df)} registros.')
print(df.describe())