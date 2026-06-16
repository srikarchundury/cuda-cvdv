#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# App4 state-transfer campaign runner for Perlmutter GPU
#
# Calls bench_cvdv.py / bench_bosonic.py / bench_qcvdv.py (this directory).
# Output structure (all relative to this directory):
#   raw_data/perlmutter_gpu/app4/runs/<campaign>/
#   app4__<method>__<tag>__<cfg_label>.out
#
# Usage: ./run.sh [options]
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ACCOUNT="${ACCOUNT:-m4916}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
QOS="${QOS:-regular}"
CONSTRAINT="${CONSTRAINT:-gpu}"
NODELIST="${NODELIST:-}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
NUM_NODES="${NUM_NODES:-4}"
ITERS="${ITERS:-10}"
WARMUP="${WARMUP:-2}"
GPU_OMP_THREAD_LIST="${GPU_OMP_THREAD_LIST:-64}"
OMP_BIND_LIST="${OMP_BIND_LIST:-spread}"
HINT_MODE_LIST="${HINT_MODE_LIST:-none}"
DRY_RUN=0

VENV_ROOT="${VENV_ROOT:-/pscratch/sd/s/schundu3/projects/cvdv/experiments_gpu}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${VENV_ROOT}/expQcvdvVirtEnv/bin/activate}"

DV_QUBITS_CSV="${DV_QUBITS_CSV:-2,4,6}"
CV_QUBITS_CSV="${CV_QUBITS_CSV:-4,6,8,10,12}"
LAM="${LAM:-0.29}"

# Method names use underscore notation matching bench_qcvdv.py's QCVDV_GPU_METHODS.
# "cuda-cvdv" → bench_cvdv.py
# "c2qa_gpu"  → bench_bosonic.py (--cv-cutoff = 2^cv_qubits)
# "qcvdv_*"   → bench_qcvdv.py --method <name>
APP4_METHODS=(
	"c2qa_gpu"
	"cuda-cvdv"
	"qcvdv_scipy_gpu"
	"qcvdv_eigen_gpu"
	"qcvdv_eigen_tensor_gpu"
	"qcvdv_torch"
	"qcvdv_torch_tensor_gpu"
	"qcvdv_diaq_gpu"
)

BENCHMARK_NAME="app4"

die()   { echo "[ERROR] $*" >&2; exit 1; }
banner(){ echo; echo "============================================================"; echo "$1"; echo "============================================================"; }

usage() {
	cat <<'EOF'
Usage: ./run.sh [options]

Options:
  --nodes N              Number of sbatch nodes/jobs (default: 4)
  --iters N              Timed iterations per run (default: 10)
  --warmup N             Warmup iterations (default: 2)
  --dv-qubits CSV        DV qubit values (default: 2,4,6)
  --cv-qubits CSV        CV qubit values (default: 4,6,8,10,12)
  --lam X                Lambda (default: 0.29)
  --omp-thread-list CSV  OMP thread sweep (default: 64)
  --omp-bind-list CSV    OMP bind sweep (default: spread)
  --hint-mode-list CSV   Slurm hint sweep (default: none)
  --gpus N               GPUs per job (default: 1)
  --account NAME         Slurm account (default: m4916)
  --time HH:MM:SS        Slurm time limit (default: 02:00:00)
  --qos NAME             Slurm qos (default: regular)
  --constraint NAME      Slurm constraint (default: gpu)
  --nodelist LIST        Slurm nodelist (optional)
  --dry-run              Print job info without submitting
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--nodes)           NUM_NODES="$2";           shift 2;;
		--iters)           ITERS="$2";               shift 2;;
		--warmup)          WARMUP="$2";              shift 2;;
		--dv-qubits)       DV_QUBITS_CSV="$2";       shift 2;;
		--cv-qubits)       CV_QUBITS_CSV="$2";       shift 2;;
		--lam)             LAM="$2";                 shift 2;;
		--omp-thread-list) GPU_OMP_THREAD_LIST="$2"; shift 2;;
		--omp-bind-list)   OMP_BIND_LIST="$2";       shift 2;;
		--hint-mode-list)  HINT_MODE_LIST="$2";      shift 2;;
		--gpus)            GPUS_PER_JOB="$2";        shift 2;;
		--account)         ACCOUNT="$2";             shift 2;;
		--time)            TIME_LIMIT="$2";          shift 2;;
		--qos)             QOS="$2";                 shift 2;;
		--constraint)      CONSTRAINT="$2";          shift 2;;
		--nodelist)        NODELIST="$2";            shift 2;;
		--dry-run)         DRY_RUN=1;               shift;;
		-h|--help)         usage; exit 0;;
		*)                 die "Unknown option: $1";;
	esac
done

[[ "$NUM_NODES" =~ ^[0-9]+$ && NUM_NODES -ge 1 ]] || die "NUM_NODES must be a positive integer"
[[ "$ITERS"    =~ ^[0-9]+$ ]] || die "ITERS must be an integer"
[[ "$WARMUP"   =~ ^[0-9]+$ ]] || die "WARMUP must be an integer"

IFS=',' read -r -a DV_QUBITS          <<< "$DV_QUBITS_CSV"
IFS=',' read -r -a CV_QUBITS          <<< "$CV_QUBITS_CSV"
IFS=',' read -r -a OMP_THREAD_OPTIONS <<< "$GPU_OMP_THREAD_LIST"
IFS=',' read -r -a OMP_PROC_BIND_OPTIONS <<< "$OMP_BIND_LIST"
IFS=',' read -r -a SLURM_HINT_OPTIONS <<< "$HINT_MODE_LIST"

[[ ${#DV_QUBITS[@]} -gt 0 ]] || die "No DV qubits configured"
[[ ${#CV_QUBITS[@]} -gt 0 ]] || die "No CV qubits configured"
[[ -f "$VENV_ACTIVATE" ]]    || die "Virtualenv not found: $VENV_ACTIVATE"
[[ -f "$SCRIPT_DIR/bench_cvdv.py"   ]] || die "bench_cvdv.py not found in $SCRIPT_DIR"
[[ -f "$SCRIPT_DIR/bench_bosonic.py" ]] || die "bench_bosonic.py not found in $SCRIPT_DIR"
[[ -f "$SCRIPT_DIR/bench_qcvdv.py"  ]] || die "bench_qcvdv.py not found in $SCRIPT_DIR"

CAMPAIGN_TAG="$(date +%Y%m%d_%H%M%S)_iters${ITERS}_nodes${NUM_NODES}"
RAW_DIR="${SCRIPT_DIR}/raw_data/perlmutter_gpu/${BENCHMARK_NAME}"
RUN_DIR="${RAW_DIR}/runs/${CAMPAIGN_TAG}"
MANIFEST_DIR="${RAW_DIR}/manifests/${CAMPAIGN_TAG}"
BATCH_LOG_DIR="${RAW_DIR}/batch_logs/${CAMPAIGN_TAG}"
CHUNK_DIR="${MANIFEST_DIR}/chunks"

mkdir -p "$RUN_DIR" "$MANIFEST_DIR" "$BATCH_LOG_DIR" "$CHUNK_DIR"

# Build manifest: one row per (method, dv, cv) — cmd dispatches to correct bench script
MANIFEST_FILE="${MANIFEST_DIR}/all_tasks.tsv"
: > "$MANIFEST_FILE"

for dv in "${DV_QUBITS[@]}"; do
	for cv in "${CV_QUBITS[@]}"; do
		tag="n_dv_qubits_${dv}__cv_qubits_${cv}__lam_${LAM}"
		for method in "${APP4_METHODS[@]}"; do
			if [[ "$method" == "cuda-cvdv" ]]; then
				cmd="${PYTHON_BIN} bench_cvdv.py --dv-qubits ${dv} --cv-qubits ${cv} --runs ${ITERS} --warmup ${WARMUP}"
			elif [[ "$method" == "c2qa_gpu" ]]; then
				cv_cutoff=$(( 1 << cv ))
				cmd="${PYTHON_BIN} bench_bosonic.py --dv-qubits ${dv} --cv-cutoff ${cv_cutoff} --runs ${ITERS} --warmup ${WARMUP}"
			else
				cmd="${PYTHON_BIN} bench_qcvdv.py --dv-qubits ${dv} --cv-qubits ${cv} --method ${method} --runs ${ITERS} --warmup ${WARMUP}"
			fi
			printf '%s\t%s\t%s\n' "$tag" "$method" "$cmd" >> "$MANIFEST_FILE"
		done
	done
done

# Split manifest into chunks (round-robin)
for ((i = 0; i < NUM_NODES; i++)); do
	: > "${CHUNK_DIR}/chunk_${i}.tsv"
done

idx=0
while IFS= read -r line; do
	[[ -n "$line" ]] || continue
	printf '%s\n' "$line" >> "${CHUNK_DIR}/chunk_$(( idx % NUM_NODES )).tsv"
	(( idx += 1 ))
done < "$MANIFEST_FILE"

TASK_COUNT=$(wc -l < "$MANIFEST_FILE" | tr -d ' ')

banner "Campaign summary"
echo "Benchmark  : ${BENCHMARK_NAME}"
echo "Campaign   : ${CAMPAIGN_TAG}"
echo "Tasks      : ${TASK_COUNT}"
echo "Nodes      : ${NUM_NODES}"
echo "Iters      : ${ITERS}  Warmup: ${WARMUP}"
echo "DV qubits  : ${DV_QUBITS[*]}"
echo "CV qubits  : ${CV_QUBITS[*]}"
echo "Lambda     : ${LAM}"
echo "OMP threads: ${OMP_THREAD_OPTIONS[*]}"
echo "Run dir    : ${RUN_DIR}"

constraint_line=""
[[ -n "$CONSTRAINT" ]] && constraint_line="#SBATCH --constraint=${CONSTRAINT}"
nodelist_line=""
[[ -n "$NODELIST" ]] && nodelist_line="#SBATCH -w ${NODELIST}"

submitted=0

for ((i = 0; i < NUM_NODES; i++)); do
	chunk_file="${CHUNK_DIR}/chunk_${i}.tsv"

	for omp_threads in "${OMP_THREAD_OPTIONS[@]}"; do
		for omp_bind in "${OMP_PROC_BIND_OPTIONS[@]}"; do
			for hint_mode in "${SLURM_HINT_OPTIONS[@]}"; do
				hint_line=""
				[[ "$hint_mode" == "multithread" ]] && hint_line="#SBATCH --hint=multithread"
				cfg_label="thr${omp_threads}__bind${omp_bind}__hint${hint_mode}"

				if [[ "$DRY_RUN" -eq 1 ]]; then
					echo "[DRY-RUN] chunk=${i} cfg=${cfg_label} file=${chunk_file}"
					continue
				fi

				submit_out="$(
sbatch <<EOT
#!/bin/bash
#SBATCH -J ${BENCHMARK_NAME}_chunk${i}_${cfg_label}
#SBATCH -o ${BATCH_LOG_DIR}/${BENCHMARK_NAME}__chunk${i}__${cfg_label}__%j.out
#SBATCH -e ${BATCH_LOG_DIR}/${BENCHMARK_NAME}__chunk${i}__${cfg_label}__%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${omp_threads}
#SBATCH --gpus=${GPUS_PER_JOB}
#SBATCH --exclusive
${hint_line}
#SBATCH -A ${ACCOUNT}
#SBATCH -t ${TIME_LIMIT}
#SBATCH --qos=${QOS}
${constraint_line}
${nodelist_line}

set -u -o pipefail

echo "[INFO] Starting chunk=${i} cfg=${cfg_label} on \$(hostname)"

cd "${VENV_ROOT}"
source "${VENV_ACTIVATE}"
if command -v nvidia-smi >/dev/null 2>&1; then
	echo "[INFO] GPU info on \$(hostname):"
	nvidia-smi
fi

export OMP_NUM_THREADS=${omp_threads}
export OMP_PLACES=cores
export OMP_PROC_BIND=${omp_bind}
export OMP_DYNAMIC=FALSE

cd "${SCRIPT_DIR}"

if [[ ! -s "${chunk_file}" ]]; then
	echo "[INFO] No work in chunk ${i}."
	exit 0
fi

while IFS=\$'\t' read -r tag method full_cmd; do
	[[ -n "\$tag" ]] || continue
	base_name="${BENCHMARK_NAME}__\${method}__\${tag}__${cfg_label}"
	out_file="${RUN_DIR}/\${base_name}.out"
	err_file="${RUN_DIR}/\${base_name}.err"
	echo "[RUN] \$full_cmd"
	bash -lc "\$full_cmd" > "\$out_file" 2> "\$err_file"
	rc=\$?
	if [[ \$rc -ne 0 ]]; then
		echo "[FAIL] rc=\$rc  tag=\$tag  method=\$method"
	else
		echo "[OK]   tag=\$tag  method=\$method"
	fi
done < "${chunk_file}"

echo "[INFO] Finished chunk=${i} cfg=${cfg_label} on \$(hostname)"
EOT
				)"
				echo "$submit_out"
				(( submitted += 1 ))
			done
		done
	done
done

banner "Done"
echo "Jobs submitted: ${submitted}"
echo "Run dir: ${RUN_DIR}"
