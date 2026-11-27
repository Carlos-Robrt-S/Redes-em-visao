#!/usr/bin/env bash
# Funções compartilhadas pelos scripts de cada GPU. Não execute este arquivo diretamente.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GROUP="$1"
LOG_DIR="logs/$GROUP"
mkdir -p "$LOG_DIR"
FAILED=()

python -c "
import platform, torch
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (sem GPU)'
print('GPU:', name)
print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| ROCm/HIP:', torch.version.hip)
print('Python:', platform.python_version())
" | tee "$LOG_DIR/hardware.txt"

# run <pasta de metadados> <nome do experimento> <argumentos do train.py...>
# Pula o experimento se ele já tiver um run salvo (permite retomar depois de uma queda).
run() {
    local metadata_dir="$1" name="$2"
    shift 2
    if compgen -G "$metadata_dir/$name/run*_metadata.json" > /dev/null; then
        echo "[pulando] $name já tem resultado salvo"
        return
    fi
    echo "[$(date '+%d/%m %H:%M')] iniciando $name"
    if python -m train "$@" 2>&1 | tee "$LOG_DIR/$name.log"; then
        echo "[$(date '+%d/%m %H:%M')] $name concluído"
    else
        echo "[ERRO] $name falhou; veja $LOG_DIR/$name.log"
        FAILED+=("$name")
    fi
}

# Empacota os resultados deste grupo para juntar com os das outras GPUs
finish() {
    local zip_name="results_${GROUP}.zip"
    rm -f "$zip_name"
    # Guarda os caminhos completos (ex.: transfer_nn/results/...) para extrair direto na raiz do projeto
    if python - "$zip_name" "$@" "$LOG_DIR" <<'PYEOF'
import sys, zipfile
from pathlib import Path
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as zf:
    for root in map(Path, sys.argv[2:]):
        for path in ([root] if root.is_file() else sorted(root.rglob("*"))):
            if path.is_file():
                zf.write(path, path.as_posix())
PYEOF
    then
        echo "Resultados empacotados em $zip_name"
    else
        echo "[ERRO] não foi possível criar $zip_name; os resultados continuam nas pastas originais"
    fi
    if [ ${#FAILED[@]} -gt 0 ]; then
        echo "Experimentos que falharam: ${FAILED[*]}"
        exit 1
    fi
    echo "Todos os experimentos de $GROUP terminaram."
}
