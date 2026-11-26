# Redes-em-visao

Trabalho prático da disciplina **Visão Computacional** na **UFJF**, com o objetivo de aprofundar os conhecimentos sobre Redes Neurais e processos de classificação de imagens utilizando o dataset público **CIFAR-10**.

Para permitir comparações justas entre arquiteturas próprias e modelos baseados em *transfer learning*, o pipeline padroniza o redimensionamento das imagens para `224x224` com divisão estratificada de treino e validação, executando os experimentos em PyTorch com suporte a GPU AMD via ROCm.

---


### Organização dos Diretórios

* **`dataset/`**: contém os dados do CIFAR-10 (`data/`), os hiperparâmetros centralizados (`hyperParam.py`) e os utilitários de carregamento, normalização e divisão estratificada (`loader.py`)[cite: 1].
* **`selfNN/`**: reúne a arquitetura própria construída do zero com PyTorch, incluindo classes auxiliares (`auxClasses/`), definições dos modelos (`model/`), registros de execução (`results/`) e metadados (`metaData/`)[cite: 1].
* **`transferNN/`**: estrutura destinada aos modelos com *transfer learning* e arquiteturas pré-treinadas, com pastas equivalentes para `model/`, `results/` e `metaData/`[cite: 1].
* **Subpastas por abordagem (`model/`, `results/`, `metaData/`)**:
  * **`model/`**: código-fonte dos modelos utilizados nas etapas de treino, validação e teste[cite: 1].
  * **`results/`**: registro das métricas de treino, validação e teste (acurácia, *loss*, matriz de confusão e *checkpoints*) por execução (*run*).
  * **`metaData/`**: registro dos hiperparâmetros, camadas, otimizadores e funções de perda utilizados em cada experimento.

---

## Requisitos e Ambiente

1. **Ative o ambiente virtual (`venv`)** na raiz do repositório:
   ```bash
   source .venv/bin/activate
   ```[cite: 1]

2. **Instale as dependências do projeto**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Instale o pacote em modo editável (se necessário para resolução de imports como módulo)**:
   ```bash
   pip install -e .
   ```[cite: 1]

> **Nota sobre GPU / ROCm:** Caso utilize GPU AMD (como a RX 6600 via ROCm), certifique-se de instalar os *wheels* do PyTorch compatíveis com a versão do ROCm instalada no sistema antes ou em conjunto com o `requirements.txt`.

---

## Como Executar

Execute os módulos a partir da raiz do projeto usando `python -m` (ou `python3 -m`):

* **Baseline (`selfNN`)**:
  ```bash
  python -m selfNN.model.baselineModel
  ```

* **Utilitário de carga e inspeção do dataset**:
  ```bash
  python -m dataset.loader
  ```[cite: 1]