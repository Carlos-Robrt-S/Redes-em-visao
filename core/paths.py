from pathlib import Path

# Caminhos absolutos a partir da raiz do projeto: funcionam de qualquer diretório
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "dataset" / "data"

SELF_RESULTS_DIR = ROOT / "self_nn" / "results"
SELF_METADATA_DIR = ROOT / "self_nn" / "metadata"

TRANSFER_RESULTS_DIR = ROOT / "transfer_nn" / "results"
TRANSFER_METADATA_DIR = ROOT / "transfer_nn" / "metadata"

TEST_RESULTS_DIR = ROOT / "test_results"