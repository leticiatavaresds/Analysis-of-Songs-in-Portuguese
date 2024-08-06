# Analysis-of-Songs-in-Portuguese

# Instruções para Rodar os Códigos `.py`

Para executar os códigos `.py` presentes no repositório, siga as instruções abaixo. A criação de um ambiente virtual não é obrigatória, mas é considerada uma boa prática para garantir que as dependências do projeto não conflitem com outras instalações de Python no seu sistema.

## 1. Criação de um Ambiente Virtual

Primeiro, vamos criar um ambiente virtual com Python na versão "3.9.19", que foi a versão utilizada na criação dos scripts. Esta é a melhor forma de garantir que não haja erros. Siga as instruções de acordo com o seu sistema operacional.

### Linux e MacOS

1. **Instale o Python 3.9.19** (se ainda não estiver instalado):

    ```sh
    sudo apt-get update # Para distribuições baseadas em Debian/Ubuntu
    sudo apt-get install python3.12 python3.12-venv python3.12-dev
    ```

    No MacOS, você pode usar o Homebrew:
    ```sh
    brew install python@3.12
    ```

2. **Crie o ambiente virtual**:
    ```sh
    python3.12 -m venv env
    ```

3. **Ative o ambiente virtual**:
    ```sh
    source env/bin/activate
    ```

### Windows

1. **Instale o Python 3.9.19** (se ainda não estiver instalado). Baixe o instalador no site oficial do Python e siga as instruções.

2. **Crie o ambiente virtual**:
    ```sh
    python -m venv env
    ```

3. **Ative o ambiente virtual**:
    ```sh
    .\env\Scripts\activate
    ```

## 2. Instalação das Bibliotecas Necessárias

Com o ambiente virtual ativado, vamos instalar as bibliotecas necessárias listadas no arquivo `requirements.txt`.

1. **Instale as dependências**:
    ```sh
    conda activate ofc39pip install -r requirements.txt
    ```

## 3. Execução dos Códigos

Agora, com todas as dependências instaladas, você pode executar os scripts `.py` presentes na pasta. Por exemplo:
```sh
python script_name.py
```