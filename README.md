# 🍽️ PreviPrato — Sistema Preditivo de Recomendação de Pedidos

**Desenvolvido por: Luma Andrade**

O **PreviPrato** é um sistema inteligente que utiliza Machine Learning para prever qual categoria de comida um usuário provavelmente vai pedir, como: pizza, sushi, hambúrguer, salada e outras opções — analisando contexto, comportamento e preferências simuladas.

Este projeto demonstra minhas habilidades com **Python**, **dados**, **IA aplicada** e minha capacidade de aprender e evoluir rapidamente.

---

## 🎯 Por que criei este projeto

Desenvolvi o PreviPrato para mostrar que consigo:

* aprender novas tecnologias em pouco tempo
* aplicar IA em problemas reais
* estruturar pipelines de dados
* transformar ideias em soluções
* unir criatividade + técnica

Usei Inteligência Artificial como **apoio ao estudo**, não como substituição.
Todas as decisões, implementações e testes foram feitos por mim.

---

## 🧠 Como o PreviPrato funciona

O sistema recebe informações como:

* ID do usuário
* Hora
* Clima
* Dia da semana
* Humor
* Preço esperado

Com isso, o modelo aprende padrões e prevê qual categoria de comida o usuário vai pedir.

Pipeline principal:

* geração de dados sintéticos
* pré-processamento
* label encoding
* treino com RandomForest
* função de inferência
* interface interativa no terminal

---

## 🔧 Tecnologias utilizadas

* Python 3.10+
* Pandas
* NumPy
* Scikit-Learn
* RandomForestClassifier
* Joblib

---

## 📂 Estrutura do projeto

```
PreviPrato/
├─ data/
│  └─ synthetic_orders.csv
├─ models/
│  ├─ modelo.pkl
│  └─ encoders.pkl
├─ main.py
├─ requirements.txt
└─ README.md
```

---

## ▶️ Como executar o projeto

### 1. Criar ambiente virtual

```bash
python -m venv .venv
```

### 2. Ativar ambiente

**Windows**

```bash
.\.venv\Scripts\activate
```

**Mac/Linux**

```bash
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Rodar o projeto

```bash
python main.py
```

---

## 🌱 Sobre meu processo de aprendizado

Usei IA como ferramenta de apoio para aprender mais rápido, revisar conceitos e explorar abordagens.
Mas todo o código, refinamento e implementação foram feitos por mim.

Este projeto marca o **começo da minha jornada com IA aplicada**, mostrando dedicação, foco e capacidade de evolução constante.

---

## 🚀 Próximos passos

* Evoluir o modelo (LightGBM / Gradient Boosting)
* Criar API com FastAPI
* Dashboard de métricas (Streamlit / Power BI)
* Melhorar a base de dados
* Recomendação Top-K

---

## 📬 Contato

**Luma Andrade**
LinkedIn: *https://www.linkedin.com/in/lumawww-andrade-ferreira-2b973a245?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app*

GitHub: *https://github.com/lumandrade?tab=repositories*

E-mail: *luma.comercialandrade@gmail.com*

---
![Execução do PreviPrato](imagens/ProjetoExecutado.png)

## ✔️ Nota sobre o uso de IA

Este projeto foi desenvolvido com apoio de ferramentas de IA para estudo, pesquisa e escrita técnica, mas o desenvolvimento real foi conduzido por mim de ponta a ponta.



