from pathlib import Path

# Caminhos absolutos a partir da raiz do projeto: funcionam de qualquer diretório
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "dataset" / "data"

SELF_RESULTS_DIR = ROOT / "selfNN" / "results"
SELF_METADATA_DIR = ROOT / "selfNN" / "metaData"

TRANSFER_RESULTS_DIR = ROOT / "transferNN" / "results"
TRANSFER_METADATA_DIR = ROOT / "transferNN" / "metaData"

TEST_RESULTS_DIR = ROOT / "testResults"