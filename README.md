# Redes-em-visao

Trabalho prático da disciplina **Visão Computacional** da **UFJF**: classificação de imagens do dataset **CIFAR-10** com redes neurais convolucionais, comparando uma **CNN própria** (e variações arquiteturais) com modelos **pré-treinados no ImageNet** via _transfer learning_.

Todas as imagens são redimensionadas para `224x224` (tamanho de entrada dos modelos pré-treinados) e todos os modelos usam a mesma divisão de dados, o mesmo laço de treino e as mesmas métricas, para que a comparação seja justa. Os experimentos rodam em PyTorch, em GPU NVIDIA (CUDA) ou AMD (ROCm).

---

## Organização dos diretórios

```
Redes-em-visao/
├── train.py          # ponto de entrada único para treinar qualquer modelo
├── analyze.py        # tabelas e gráficos de todos os experimentos (validação)
├── evaluate.py       # avaliação final no conjunto de teste
├── core/             # código compartilhado entre as abordagens
├── dataset/          # carregamento dos dados e hiperparâmetros
├── self_nn/          # CNN própria e variações
├── transfer_nn/      # modelos pré-treinados
└── scripts/          # scripts para dividir os treinos entre GPUs
```

- **`core/`**: laço de treino/validação e medição de tempo (`trainer.py`), gravação de métricas, gráficos e metadados (`file_plot.py`), caminhos do projeto (`paths.py`) e fixação da seed (`reproducibility.py`).
- **`dataset/`**: dados do CIFAR-10 (`data/`, fora do git), hiperparâmetros centralizados (`hyper_param.py`) e carregamento, normalização, divisão estratificada e conjunto de teste (`loader.py`).
- **`self_nn/`** e **`transfer_nn/`**: cada um com
  - **`model/`**: apenas as definições dos modelos;
  - **`results/<experimento>/runN/`**: curvas (CSV e gráfico), matriz de confusão, métricas por classe e _checkpoint_ do melhor modelo de cada execução;
  - **`metadata/<experimento>/runN_metadata.json`**: hiperparâmetros, arquitetura, número de parâmetros, melhor época e tempos de treino e inferência.
- **`analysis/`** e **`test_results/`**: gerados por `analyze.py` e `evaluate.py`.

---

## Modelos

### CNN própria (`self_nn`)

Cada variação altera **uma única característica** do baseline, para isolar o efeito de cada decisão de projeto.

| `--model`    | Descrição                                                                                                | Parâmetros |
| ------------ | -------------------------------------------------------------------------------------------------------- | ---------- |
| `baseline`   | 4 convoluções (10, 20, 40, 40 filtros) + 3 camadas totalmente conectadas                                 | 600.990    |
| `batch_norm` | baseline + `BatchNorm2d` após cada convolução                                                            | 601.210    |
| `wide`       | baseline com o dobro de filtros (20, 40, 80, 80)                                                         | 1.248.370  |
| `gap`        | baseline com Global Average Pooling + uma única camada linear no lugar das camadas totalmente conectadas | 49.790     |

### Transfer learning (`transfer_nn`)

Arquiteturas `resnet18` e `mobilenet_v2`, com pesos do ImageNet e a última camada trocada para 10 classes.

| `--strategy` | O que é treinado                                                                   |
| ------------ | ---------------------------------------------------------------------------------- |
| `frozen`     | apenas a nova camada final (backbone congelado)                                    |
| `partial`    | último bloco + camada final (`layer4` na ResNet18; `features[14:]` na MobileNetV2) |
| `full`       | a rede inteira (_fine-tuning_ completo)                                            |
| `scratch`    | a rede inteira, **sem** pesos pré-treinados (comparação com treino do zero)        |

As camadas de BatchNorm das partes congeladas são mantidas em modo de avaliação, para não alterar as estatísticas aprendidas no ImageNet.

---

## Protocolo experimental

- **Divisão dos dados**: as 50.000 imagens de treino do CIFAR-10 são divididas uma única vez, de forma estratificada, em **45.000 para treino** e **5.000 para validação** (seed fixa). As **10.000 imagens de teste** só são usadas na avaliação final.
- **Pré-processamento**: redimensionamento bicúbico para `224x224`. Normalização com média e desvio do CIFAR-10 na CNN própria e no `scratch`, e com os do ImageNet nos modelos pré-treinados. Não há _data augmentation_.
- **Seleção de modelos**: em cada execução é salvo o modelo da época com maior acurácia de validação; a melhor configuração de cada abordagem também é escolhida pela validação.
- **Reprodutibilidade**: seeds fixadas (`random`, NumPy e PyTorch) para inicialização dos pesos, Dropout e embaralhamento dos dados. Pequenas variações entre execuções podem ocorrer devido a operações não determinísticas da GPU.

### Hiperparâmetros (`dataset/hyper_param.py`)

|                     | CNN própria               | Transfer learning                                     |
| ------------------- | ------------------------- | ----------------------------------------------------- |
| Função de perda     | CrossEntropy              | CrossEntropy                                          |
| Otimizador          | AdamW (weight decay 0,01) | AdamW (weight decay 0,01)                             |
| Taxa de aprendizado | 1e-3                      | 1e-3 (`frozen`, `scratch`) / 1e-4 (`partial`, `full`) |
| Batch               | 64                        | 64                                                    |
| Épocas              | `HYPER_PARAMS["EPOCHS"]`  | `TRANSFER_HYPER_PARAMS["EPOCHS"]`                     |

A taxa de aprendizado menor em `partial` e `full` evita destruir os pesos pré-treinados; menos épocas no transfer learning se justificam pela convergência mais rápida dos modelos pré-treinados.

---

## Setup

Todos os comandos devem ser executados **na raiz do projeto**. No Windows, use o **Git Bash**.

1. **Clone o repositório** (na branch de trabalho):

   ```bash
   git clone -b feat/etapa-2 https://github.com/Carlos-Robrt-S/Redes-em-visao.git
   cd Redes-em-visao
   ```

2. **Crie e ative o ambiente virtual**:

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux
   source venv/Scripts/activate    # Windows (Git Bash)
   ```

3. **Instale o PyTorch com suporte à sua GPU** usando o comando do [seletor do pytorch.org](https://pytorch.org/get-started/locally/) (CUDA para NVIDIA, ROCm para AMD). No Google Colab o PyTorch já vem instalado.

   > **GPU AMD sem suporte oficial ao ROCm** (ex.: RX 6600): rode `export HSA_OVERRIDE_GFX_VERSION=10.3.0` antes de treinar.

4. **Instale as demais dependências**:

   ```bash
   pip install tqdm scikit-learn matplotlib numpy
   ```

   Evite `pip install -r requirements.txt` depois do passo 3: ele pode substituir o PyTorch com GPU por outra versão.

5. **Confira se a GPU foi detectada** (deve imprimir `True` e o nome da placa):

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

6. **Baixe o CIFAR-10** (uma vez por computador, cerca de 170 MB). O dataset não é versionado no git; cada máquina precisa baixar o seu.

   > ⚠️ Rode o comando **na raiz do projeto** (a pasta onde está o `train.py`). O caminho `dataset/data` é relativo à pasta atual: rodando de outro lugar (ex.: de dentro de `scripts/`), o dataset é salvo no lugar errado e o treino falha com `Dataset not found or corrupted`.

   ```bash
   python -c "import torchvision.datasets as d; d.CIFAR10('dataset/data', train=True, download=True); d.CIFAR10('dataset/data', train=False, download=True)"
   ```

   O resultado deve ficar assim:

   ```
   Redes-em-visao/
   └── dataset/
       └── data/
           ├── cifar-10-batches-py/     ← usado pelo código
           └── cifar-10-python.tar.gz   ← arquivo baixado (pode ser apagado)
   ```

   Para conferir: `ls dataset/data` deve mostrar `cifar-10-batches-py`.

---

## Como executar

### Um experimento

```bash
python -m train --model baseline
python -m train --model resnet18 --strategy frozen
python -m train --model mobilenet_v2 --strategy full --epochs 1   # teste rápido
```

`--epochs` e `--lr` sobrescrevem os valores padrão apenas naquela execução (o valor usado fica registrado nos metadados).

### Todos os experimentos, divididos em três GPUs

Cada script roda uma família de modelos, para que os tempos de treino dentro de cada comparação venham da mesma GPU:

| Script                                  | Experimentos                                        |
| --------------------------------------- | --------------------------------------------------- |
| `bash scripts/run_gpu1_cnn.sh`          | `baseline`, `batch_norm`, `wide`, `gap`             |
| `bash scripts/run_gpu2_resnet18.sh`     | ResNet18: `frozen`, `full`, `partial`, `scratch`    |
| `bash scripts/run_gpu3_mobilenet_v2.sh` | MobileNetV2: `frozen`, `full`, `partial`, `scratch` |

Cada script:

- registra a GPU usada em `logs/<grupo>/hardware.txt` e o log de cada treino em `logs/<grupo>/<experimento>.log`;
- **pula experimentos que já têm resultado** (se a máquina cair, basta rodar de novo);
- continua quando um experimento falha e lista as falhas no final;
- gera `results_<grupo>.zip` com resultados, metadados e logs.

Para juntar os resultados das três máquinas, extraia os zips na raiz do projeto:

```bash
python -m zipfile -e results_cnn.zip .
python -m zipfile -e results_resnet18.zip .
python -m zipfile -e results_mobilenet_v2.zip .
```

**Dicas de ambiente:**

- **Colab**: use `!bash scripts/...` e reduza os workers para o número de núcleos disponíveis antes de treinar: `!sed -i 's/"NUM_WORKERS": 4/"NUM_WORKERS": 2/' dataset/hyper_param.py`.
- **Windows**: se aparecer `$'\r': command not found`, rode `sed -i 's/\r$//' scripts/*.sh`.

### Análise (conjunto de validação)

```bash
python -m analyze
```

Lê todos os runs e gera em `analysis/`:

- `validation_summary.csv`: todos os runs, com acurácia de validação, melhor época, diferença treino − validação (indicador de _overfitting_), parâmetros e tempos;
- `validation_summary.tex`: tabela pronta para o relatório em LaTeX (pacote `booktabs`);
- `curves_self_nn`, `curves_resnet18`, `curves_mobilenet_v2`: curvas de treino e validação sobrepostas;
- `complexity_vs_accuracy`: número de parâmetros x acurácia.

Os gráficos são salvos em PDF (vetorial) e PNG. Não usa o conjunto de teste e pode ser executado quantas vezes for necessário.

### Avaliação final (conjunto de teste)

```bash
python -m evaluate
```

> ⚠️ Execute **uma única vez**, depois que **todos** os experimentos terminarem.

Para cada abordagem (CNN própria, ResNet18 e MobileNetV2), escolhe a execução com a maior acurácia de **validação** e avalia somente ela nas 10.000 imagens de teste. Gera em `test_results/`:

- `comparison.csv` e `comparison.pdf`: acurácia de validação e de teste, parâmetros, tempo de treino e tempo de inferência por imagem;
- `per_class_accuracy.csv`: acurácia de cada classe em cada modelo;
- uma pasta por modelo com matriz de confusão (PDF e PNG) e precision, recall e F1-score por classe.

---

## Requisitos

- Python 3.10 ou superior
- PyTorch e torchvision (com CUDA ou ROCm para treinar em GPU)
- NumPy, scikit-learn, matplotlib e tqdm (ver `requirements.txt`)
