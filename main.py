import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# -------------------------
#  DADOS SIMULADOS
# -------------------------

np.random.seed(42)  # deixa tudo reproduzível

usuarios = range(1, 501)  # 500 usuários fictícios
horas = range(0, 24)
clima = ["sol", "chuva", "nublado"]
categorias = ["pizza", "hamburguer", "sushi", "salada", "marmita", "massa"]
humor = ["feliz", "neutro", "triste"]

# cria 5000 registros
n = 5000

df = pd.DataFrame({
    "usuario_id": np.random.choice(usuarios, n),
    "hora": np.random.choice(horas, n),
    "clima": np.random.choice(clima, n),
    "dia_semana": np.random.choice(["seg", "ter", "qua", "qui", "sex", "sab", "dom"], n),
    "humor": np.random.choice(humor, n),
    "categoria_pedido": np.random.choice(categorias, n),
    "preco": np.random.uniform(15, 80, n).round(2)
})

# mostra as primeiras linhas
print(df.head())

print(df.head())

# -----------------------------------
# TRANSFORMANDO DADOS PARA O MODELO
# -----------------------------------

# separa os atributos (X) e o que queremos prever (y)
X = df.drop("categoria_pedido", axis=1)  # tudo, menos o alvo
y = df["categoria_pedido"]               # o alvo

# seleciona as colunas que são categóricas
colunas_categoricas = ["clima", "dia_semana", "humor"]

encoder = OneHotEncoder(handle_unknown='ignore')

# aplica a transformação
X_encoded = encoder.fit_transform(X[colunas_categoricas])

# transforma o resultado do encoder em DataFrame
df_encoded = pd.DataFrame(
    X_encoded.toarray(),
    columns=encoder.get_feature_names_out(colunas_categoricas)
)

# junta com as colunas numéricas originais (usuario_id, hora, preco)
df_final = pd.concat([X[["usuario_id", "hora", "preco"]].reset_index(drop=True), df_encoded], axis=1)

print("\nDADOS PRONTOS PARA IA:")
print(df_final.head())

# -------------------------------------
# SEPARANDO DADOS PARA TREINO E TESTE
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df_final, y, test_size=0.2, random_state=42
)

print("\nTAMANHO DOS DADOS:")
print("Treino:", len(X_train))
print("Teste:", len(X_test))

# -------------------------
# TREINANDO O MODELO DE IA
# -------------------------

modelo = RandomForestClassifier()

modelo.fit(X_train, y_train)

print("\nModelo treinado com sucesso!")

# -------------------------
# FUNÇÃO PARA FAZER PREVISÕES
# -------------------------

def prever_pedido(usuario_id, hora, clima, dia_semana, humor, preco):
    # cria um DataFrame com apenas 1 linha
    nova_linha = pd.DataFrame({
        "usuario_id": [usuario_id],
        "hora": [hora],
        "clima": [clima],
        "dia_semana": [dia_semana],
        "humor": [humor],
        "preco": [preco]
    })

    # aplica o mesmo encoding que usamos no treino
    nova_linha_encoded = encoder.transform(nova_linha[["clima", "dia_semana", "humor"]])

    df_novo = pd.DataFrame(
        nova_linha_encoded.toarray(),
        columns=encoder.get_feature_names_out(["clima", "dia_semana", "humor"])
    )

    # junta com dados numéricos
    df_novo_final = pd.concat([
        nova_linha[["usuario_id", "hora", "preco"]].reset_index(drop=True),
        df_novo
    ], axis=1)

    # faz a previsão
    predicao = modelo.predict(df_novo_final)[0]
    return predicao
# exemplo de teste
resultado = prever_pedido(
    usuario_id=123,
    hora=20,
    clima="sol",
    dia_semana="sex",
    humor="feliz",
    preco=40
)

print("\nPREVISÃO DO MODELO PARA ESSE USUÁRIO:")
print(resultado)


# --------------------------------------
# INTERFACE SIMPLES PARA O USUÁRIO
# --------------------------------------

print("\n=== PreviPrato IA — Seu aliado inteligente na escolha dos seus pedidos ===")

while True:
    print("\nDigite os dados do usuário:")

    usuario_id = int(input("ID do usuário (ex: 123): "))
    hora = int(input("Hora atual (0 a 23): "))
    clima = input("Clima (sol, chuva, nublado): ").lower()
    dia_semana = input("Dia da semana (seg, ter, qua, qui, sex, sab, dom): ").lower()
    humor = input("Humor (feliz, neutro, triste): ").lower()
    preco = float(input("Preço estimado (ex: 35.5): "))

    resultado = prever_pedido(usuario_id, hora, clima, dia_semana, humor, preco)

    print("\n🍽️ A IA acredita que o usuário vai pedir:")
    print("➡", resultado)

    continuar = input("\nQuer fazer outra previsão? (s/n): ").lower()
    if continuar != "s":
        print("\nObrigado por usar o PreviPrato IA!")
        break