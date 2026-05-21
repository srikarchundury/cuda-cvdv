#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DV_QUBITS="${DV_QUBITS:-4}"
CV_QUBITS_CSV="${CV_QUBITS_CSV:-10,11,12}"
RUNS="${RUNS:-10}"
WARMUP="${WARMUP:-2}"
LAM="${LAM:-0.29}"
QCVDV_METHODS="${QCVDV_METHODS:-dense_matrix_gpu,dense_matrix_gpuv1,torch}"
NO_BOSONIC="${NO_BOSONIC:-0}"
NO_QCVDV="${NO_QCVDV:-0}"
OUTPUT="${OUTPUT:-$SCRIPT_DIR/results/same_final_circuits_runtime.json}"

usage() {
	cat <<'EOF'
Usage: ./run_same_final.sh [options]

Options:
  --dv-qubits N           Number of DV qubits (default: 4)
  --cv-qubits CSV         CV qubits list, e.g. 10,11,12
  --runs N                Timed runs per config (default: 10)
  --warmup N              Warmup runs per config (default: 2)
  --lam X                 Interaction parameter lambda (default: 0.29)
  --qcvdv-methods CSV     qcvdv methods CSV (default: dense_matrix_gpu,dense_matrix_gpuv1,torch)
  --no-bosonic            Skip bosonic backend
  --no-qcvdv              Skip qcvdv backend
  --output PATH           Output JSON path
  -h, --help              Show this help

Env overrides (equivalent):
  PYTHON_BIN DV_QUBITS CV_QUBITS_CSV RUNS WARMUP LAM QCVDV_METHODS NO_BOSONIC NO_QCVDV OUTPUT
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--dv-qubits)
			DV_QUBITS="$2"
			shift 2
			;;
		--cv-qubits)
			CV_QUBITS_CSV="$2"
			shift 2
			;;
		--runs)
			RUNS="$2"
			shift 2
			;;
		--warmup)
			WARMUP="$2"
			shift 2
			;;
		--lam)
			LAM="$2"
			shift 2
			;;
		--qcvdv-methods)
			QCVDV_METHODS="$2"
			shift 2
			;;
		--no-bosonic)
			NO_BOSONIC=1
			shift
			;;
		--no-qcvdv)
			NO_QCVDV=1
			shift
			;;
		--output)
			OUTPUT="$2"
			shift 2
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Unknown option: $1" >&2
			usage
			exit 1
			;;
	esac
done

CMD=(
	"$PYTHON_BIN" same_final_circuits_run.py
	--dv-qubits "$DV_QUBITS"
	--cv-qubits "$CV_QUBITS_CSV"
	--runs "$RUNS"
	--warmup "$WARMUP"
	--lam "$LAM"
	--qcvdv-methods "$QCVDV_METHODS"
	--output "$OUTPUT"
)

if [[ "$NO_BOSONIC" == "1" ]]; then
	CMD+=(--no-bosonic)
fi
if [[ "$NO_QCVDV" == "1" ]]; then
	CMD+=(--no-qcvdv)
fi

"${CMD[@]}"
