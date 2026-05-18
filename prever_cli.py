```python id="v3s9qa"
import joblib
import os
import sys
import pandas as pd
import numpy as np

MODEL_PATH = 'modelo_preco_casas.pkl'

# =========================
# VERIFICAR MODELO
# =========================

if not os.path.exists(MODEL_PATH):

    print('Erro: Modelo nao encontrado.')
    print('Execute "treinar_modelo.py" antes de usar este script.')

    sys.exit(1)


modelo = joblib.load(MODEL_PATH)

# =========================
# CABECALHO
# =========================

print('=' * 45)
print('      IA DE PREVISAO DE PRECO DE CASAS')
print('=' * 45)

# =========================
# FUNCAO DE VALIDACAO
# =========================

def pedir_numero(
    mensagem,
    tipo=float,
    minimo=None,
    maximo=None
):
    """
    Le e valida numeros digitados.
    """

    while True:

        try:

            valor = tipo(input(mensagem))

            if minimo is not None and valor < minimo:

                print(
                    f'Valor minimo permitido: {minimo}'
                )

                continue

            if maximo is not None and valor > maximo:

                print(
                    f'Valor maximo permitido: {maximo}'
                )

                continue

            return valor

        except ValueError:

            print(
                'Digite apenas numeros validos.'
            )

# =========================
# ENTRADAS
# =========================

tamanho = pedir_numero(
    'Tamanho da casa (m2)        : ',
    float,
    minimo=20
)

quartos = pedir_numero(
    'Quantidade de quartos       : ',
    int,
    minimo=1
)

banheiros = pedir_numero(
    'Quantidade de banheiros     : ',
    int,
    minimo=1
)

idade = pedir_numero(
    'Idade da casa (anos)        : ',
    int,
    minimo=0
)

garagem = pedir_numero(
    'Vagas de garagem            : ',
    int,
    minimo=0,
    maximo=10
)

# =========================
# LOCALIZACAO
# =========================

print('\nLocalizacao:')

print('  1 - Periferia')
print('  2 - Zona intermediaria')
print('  3 - Centro / bairro nobre')

localizacao_opcao = pedir_numero(
    'Escolha (1, 2 ou 3)         : ',
    int,
    minimo=1,
    maximo=3
)


mapa_localizacao = {
    1: 'Periferia',
    2: 'Zona intermediária',
    3: 'Centro / bairro nobre'
}

localizacao = mapa_localizacao[
    localizacao_opcao
]

# =========================
# VALIDACOES EXTRAS
# =========================

if banheiros > quartos:

    print(
        '\nAviso: mais banheiros do que quartos '
        'e incomum.'
    )

if tamanho < 35 and quartos >= 4:

    print(
        '\nAviso: muitos quartos para uma '
        'metragem pequena.'
    )

# =========================
# PREPARAR DADOS
# =========================

dados = pd.DataFrame([{
    'tamanho': tamanho,
    'quartos': quartos,
    'banheiros': banheiros,
    'idade': idade,
    'garagem': garagem,
    'localizacao': localizacao
}])

# =========================
# PREVISAO
# =========================

try:

    preco_log = modelo.predict(dados)[0]


    preco_previsto = np.exp(preco_log)


    preco_previsto = max(
        preco_previsto,
        50000
    )

    # =========================
    # RESULTADO
    # =========================

    print()
    print('=' * 45)

    print(
        f'Preco previsto: '
        f'R$ {preco_previsto:,.2f}'
    )

    print('=' * 45)

    print(
        'Estimativa baseada no modelo treinado.'
    )

    print(
        'Resultados reais podem variar.'
    )

except Exception as erro:

    print('\nErro ao realizar previsao.')

    print(f'Detalhes: {erro}')
```
