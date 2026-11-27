# Redes-em-visao

Trabalho prático da disciplina **Visão Computacional** na **UFJF**, com o objetivo de aprofundar os conhecimentos sobre Redes Neurais e processos de classificação de imagens utilizando o dataset público **CIFAR-10**.

Para permitir comparações justas entre arquiteturas próprias e modelos baseados em *transfer learning*, o pipeline padroniza o redimensionamento das imagens para `224x224` com divisão estratificada de treino e validação, executando os experimentos em PyTorch com suporte a GPU AMD via ROCm.

---


### Organização dos Diretórios

* **`train.py`**: ponto de entrada único para treinar qualquer modelo (CNN própria ou pré-treinado).
* **`core/`**: código compartilhado entre as abordagens: laço de treino/validação e medição de tempo (`trainer.py`), gravação de métricas, gráficos e metadados (`filePlot.py`) e caminhos do projeto (`paths.py`).
* **`dataset/`**: contém os dados do CIFAR-10 (`data/`), os hiperparâmetros centralizados (`hyperParam.py`) e os utilitários de carregamento, normalização, divisão estratificada e conjunto de teste (`loader.py`).
* **`selfNN/`**: CNN própria construída do zero, com definições dos modelos (`model/`), resultados (`results/`) e metadados (`metaData/`).
* **`transferNN/`**: modelos pré-treinados (ResNet18 e MobileNetV2) e estratégias de congelamento (`model/transferModels.py`), com pastas equivalentes para `results/` e `metaData/`.
* **Subpastas por abordagem (`model/`, `results/`, `metaData/`)**:
  * **`model/`**: apenas as definições dos modelos.
  * **`results/<experimento>/runN/`**: métricas de treino e validação (acurácia, *loss*, matriz de confusão e *checkpoint*) por execução.
  * **`metaData/<experimento>/runN_metadata.json`**: hiperparâmetros, arquitetura, otimizador, função de perda, número de parâmetros e tempos de treino e inferência.

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

* **CNN própria (`selfNN`)**:
  ```bash
  python -m train --model baseline
  ```

* **Transfer learning (`transferNN`)**: `--model` em `resnet18` ou `mobilenet_v2` e `--strategy` em `frozen`, `partial`, `full` ou `scratch`:
  ```bash
  python -m train --model resnet18 --strategy frozen
  python -m train --model mobilenet_v2 --strategy full --epochs 10
  ```

* **Utilitário de carga e inspeção do dataset**:
  ```bash
  python -m dataset.loader
  ```[cite: 1]