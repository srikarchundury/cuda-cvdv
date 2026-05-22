#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ACCOUNT="${ACCOUNT:-m4916_g}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
CONSTRAINT="${CONSTRAINT:-gpu}"
QOS="${QOS:-regular}"
PARTITION="${PARTITION:-}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
NUM_JOBS="${NUM_JOBS:-4}"
RUNS="${RUNS:-10}"
WARMUP="${WARMUP:-2}"
LAM="${LAM:-0.29}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/results_w_same_final_circuit}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/pscratch/sd/s/schundu3/projects/cvdv/experiments_gpu/expQcvdvVirtEnv/bin/activate}"

DV_QUBITS_CSV="${DV_QUBITS_CSV:-2,3,4,5,6,7,8}"
CVDV_CV_QUBITS_CSV="${CVDV_CV_QUBITS_CSV:-10,11,12}"
BOSONIC_CV_QUBITS_CSV="${BOSONIC_CV_QUBITS_CSV:-10,11,12}"
QCVDV_CV_QUBITS_CSV="${QCVDV_CV_QUBITS_CSV:-10,11,12}"
QCVDV_METHODS_CSV="${QCVDV_METHODS_CSV:-dense_matrix_gpu,dense_matrix_gpuv1,torch}"

DRY_RUN=0

usage() {
	cat <<'EOF'
Usage: ./run_same_final.sh [options]

Options:
  --jobs N                  Number of sbatch jobs to submit
  --runs N                  Number of timed iterations per benchmark run
  --warmup N                Warmup iterations per run
  --lam X                   Interaction parameter lambda (default: 0.29)
  --dv-qubits CSV           Comma-separated DV qubit values
  --cvdv-cv-qubits CSV      Comma-separated CVDV CV qubit values
  --bosonic-cv-qubits CSV   Comma-separated Bosonic CV qubit values
  --qcvdv-cv-qubits CSV     Comma-separated qcvdv CV qubit values
  --qcvdv-methods CSV       Comma-separated qcvdv methods
  --output-root PATH        Root directory for campaign outputs
  --account NAME            Slurm account (default: m4916_g)
  --time HH:MM:SS           Slurm time limit (default: 02:00:00)
  --constraint NAME         Slurm constraint (default: gpu)
  --qos NAME                Slurm qos (default: regular)
  --partition NAME          Slurm partition (optional)
  --gpus N                  GPUs per job (default: 1)
  --cpus N                  CPUs per task (default: 4)
  --dry-run                 Print generated sbatch commands without submitting
  -h, --help                Show this help
EOF
}

die() {
	echo "[ERROR] $*" >&2
	exit 1
}

sanitize() {
	local s="$1"
	s="${s//:/_}"
	s="${s//,/__}"
	s="${s// /_}"
	s="${s//\//_}"
	printf '%s' "$s"
}

csv_to_array() {
	local csv="$1"
	local -n out_arr="$2"
	IFS=',' read -r -a out_arr <<< "$csv"
}

contains_value() {
	local needle="$1"
	shift
	local item
	for item in "$@"; do
		if [[ "$item" == "$needle" ]]; then
			return 0
		fi
	done
	return 1
}

parse_args() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
			--jobs)
				NUM_JOBS="$2"
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
			--dv-qubits)
				DV_QUBITS_CSV="$2"
				shift 2
				;;
			--cvdv-cv-qubits)
				CVDV_CV_QUBITS_CSV="$2"
				shift 2
				;;
			--bosonic-cv-qubits)
				BOSONIC_CV_QUBITS_CSV="$2"
				shift 2
				;;
			--qcvdv-cv-qubits)
				QCVDV_CV_QUBITS_CSV="$2"
				shift 2
				;;
			--qcvdv-methods)
				QCVDV_METHODS_CSV="$2"
				shift 2
				;;
			--output-root)
				OUTPUT_ROOT="$2"
				shift 2
				;;
			--account)
				ACCOUNT="$2"
				shift 2
				;;
			--time)
				TIME_LIMIT="$2"
				shift 2
				;;
			--constraint)
				CONSTRAINT="$2"
				shift 2
				;;
			--qos)
				QOS="$2"
				shift 2
				;;
			--partition)
				PARTITION="$2"
				shift 2
				;;
			--gpus)
				GPUS_PER_JOB="$2"
				shift 2
				;;
			--cpus)
				CPUS_PER_TASK="$2"
				shift 2
				;;
			--dry-run)
				DRY_RUN=1
				shift
				;;
			-h|--help)
				usage
				exit 0
				;;
			*)
				die "Unknown option: $1"
				;;
		esac
	done
}

parse_args "$@"

[[ -x "$PYTHON_BIN" || -n "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" ]] || die "Python binary not found: $PYTHON_BIN"
[[ -f "$SCRIPT_DIR/same_final_circuits_run.py" ]] || die "Missing benchmark runner: $SCRIPT_DIR/same_final_circuits_run.py"

csv_to_array "$DV_QUBITS_CSV" DV_QUBITS
csv_to_array "$CVDV_CV_QUBITS_CSV" CVDV_CV_QUBITS
csv_to_array "$BOSONIC_CV_QUBITS_CSV" BOSONIC_CV_QUBITS
csv_to_array "$QCVDV_CV_QUBITS_CSV" QCVDV_CV_QUBITS

[[ ${#DV_QUBITS[@]} -gt 0 ]] || die "No DV qubits configured"
[[ ${#BOSONIC_CV_QUBITS[@]} -gt 0 || ${#CVDV_CV_QUBITS[@]} -gt 0 || ${#QCVDV_CV_QUBITS[@]} -gt 0 ]] || die "No CV qubits configured for any backend"
[[ "$NUM_JOBS" =~ ^[0-9]+$ ]] || die "NUM_JOBS must be an integer"
[[ "$RUNS" =~ ^[0-9]+$ ]] || die "RUNS must be an integer"
[[ "$WARMUP" =~ ^[0-9]+$ ]] || die "WARMUP must be an integer"
(( NUM_JOBS >= 1 )) || die "NUM_JOBS must be >= 1"

CAMPAIGN_TAG="$(date +%Y%m%d_%H%M%S)_jobs${NUM_JOBS}_runs${RUNS}_warmup${WARMUP}"
CAMPAIGN_ROOT="$OUTPUT_ROOT/$CAMPAIGN_TAG"
MANIFEST_DIR="$CAMPAIGN_ROOT/manifest"
CHUNKS_DIR="$CAMPAIGN_ROOT/chunks"
RUNS_DIR="$CAMPAIGN_ROOT/runs"
LOGS_DIR="$CAMPAIGN_ROOT/batch_logs"

mkdir -p "$MANIFEST_DIR" "$CHUNKS_DIR" "$RUNS_DIR" "$LOGS_DIR"

MANIFEST_FILE="$MANIFEST_DIR/all_tasks.tsv"
CHUNK_SPECS_FILE="$MANIFEST_DIR/chunk_assignments.tsv"
: > "$MANIFEST_FILE"
: > "$CHUNK_SPECS_FILE"

declare -a TASK_RECORDS=()

emit_task() {
	local dv="$1"
	local cv="$2"
	local cost="$3"
	local tag="$4"
	local do_cvdv="$5"
	local do_bosonic="$6"
	local do_qcvdv="$7"
	local line="$cost|$dv|$cv|$tag"
	TASK_RECORDS+=("$line")
	printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$dv" "$cv" "$cost" "$do_cvdv" "$do_bosonic" "$do_qcvdv" >> "$MANIFEST_FILE"
}

declare -A CV_UNION_MAP=()
for cv in "${CVDV_CV_QUBITS[@]}" "${BOSONIC_CV_QUBITS[@]}" "${QCVDV_CV_QUBITS[@]}"; do
	[[ -n "$cv" ]] || continue
	CV_UNION_MAP["$cv"]=1
done

mapfile -t UNION_CV_QUBITS < <(printf '%s\n' "${!CV_UNION_MAP[@]}" | sort -n)
[[ ${#UNION_CV_QUBITS[@]} -gt 0 ]] || die "No CV qubits configured across any backend"

for dv in "${DV_QUBITS[@]}"; do
	for cv in "${UNION_CV_QUBITS[@]}"; do
		[[ "$dv" =~ ^[0-9]+$ ]] || die "Invalid DV qubit value: $dv"
		[[ "$cv" =~ ^[0-9]+$ ]] || die "Invalid CV qubit value: $cv"
		tag="dv${dv}__cv${cv}"
		do_cvdv=0
		do_bosonic=0
		do_qcvdv=0
		if contains_value "$cv" "${CVDV_CV_QUBITS[@]}"; then
			do_cvdv=1
		fi
		if contains_value "$cv" "${BOSONIC_CV_QUBITS[@]}"; then
			do_bosonic=1
		fi
		if contains_value "$cv" "${QCVDV_CV_QUBITS[@]}"; then
			do_qcvdv=1
		fi
		if (( do_cvdv == 0 && do_bosonic == 0 && do_qcvdv == 0 )); then
			continue
		fi
		cost=$((dv + cv))
		emit_task "$dv" "$cv" "$cost" "$tag" "$do_cvdv" "$do_bosonic" "$do_qcvdv"
	done
done

(( ${#TASK_RECORDS[@]} > 0 )) || die "No DV/CV combinations generated"

mapfile -t SORTED_TASKS < <(printf '%s\n' "${TASK_RECORDS[@]}" | sort -t'|' -k1,1nr -k2,2n -k3,3n)

job_count=${NUM_JOBS}
if (( job_count > ${#SORTED_TASKS[@]} )); then
	job_count=${#SORTED_TASKS[@]}
fi
(( job_count >= 1 )) || die "No jobs to submit"

declare -a JOB_LOADS=()
for ((i = 0; i < job_count; i++)); do
	JOB_LOADS+=(0)
done

declare -a CHUNK_FILES=()
for ((i = 0; i < job_count; i++)); do
	chunk_file="$CHUNKS_DIR/chunk_$(printf '%02d' "$i").tsv"
	CHUNK_FILES+=("$chunk_file")
	: > "$chunk_file"
done

for record in "${SORTED_TASKS[@]}"; do
	IFS='|' read -r cost dv cv tag <<< "$record"
	do_cvdv=0
	do_bosonic=0
	do_qcvdv=0
	if contains_value "$cv" "${CVDV_CV_QUBITS[@]}"; then
		do_cvdv=1
	fi
	if contains_value "$cv" "${BOSONIC_CV_QUBITS[@]}"; then
		do_bosonic=1
	fi
	if contains_value "$cv" "${QCVDV_CV_QUBITS[@]}"; then
		do_qcvdv=1
	fi

	best_idx=0
	best_load="${JOB_LOADS[0]}"
	for ((i = 1; i < job_count; i++)); do
		if (( JOB_LOADS[i] < best_load )); then
			best_load="${JOB_LOADS[i]}"
			best_idx=$i
		fi
	done

	printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$dv" "$cv" "$tag" "$cost" "$do_cvdv" "$do_bosonic" "$do_qcvdv" >> "${CHUNK_FILES[$best_idx]}"
	JOB_LOADS[$best_idx]=$((JOB_LOADS[$best_idx] + cost))
done

SBATCH_SCRIPT="$MANIFEST_DIR/same_final_job.sbatch"
cat > "$SBATCH_SCRIPT" <<'EOF'
#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=__CPUS_PER_TASK__
#SBATCH --gpus=__GPUS_PER_JOB__
#SBATCH --exclusive
set -euo pipefail

: "${SCRIPT_DIR:?missing SCRIPT_DIR}"
: "${RUNNER_PY:?missing RUNNER_PY}"
: "${CHUNK_FILE:?missing CHUNK_FILE}"
: "${RUNS_DIR:?missing RUNS_DIR}"
: "${PYTHON_BIN:?missing PYTHON_BIN}"
: "${QCVDV_METHODS_CSV:?missing QCVDV_METHODS_CSV}"
: "${RUNS:?missing RUNS}"
: "${WARMUP:?missing WARMUP}"
: "${LAM:?missing LAM}"

if [[ -f "${VENV_ACTIVATE:-}" ]]; then
	# shellcheck disable=SC1090
	source "$VENV_ACTIVATE"
fi

while IFS=$'\t' read -r dv cv tag cost do_cvdv do_bosonic do_qcvdv; do
	[[ -n "${dv:-}" ]] || continue
	output_dir="$RUNS_DIR/$tag"
	mkdir -p "$output_dir"
	output_json="$output_dir/same_final_circuits_runtime.json"

	echo "[JOB ${SLURM_JOB_ID:-local}] running tag=$tag dv=$dv cv=$cv cost=$cost cvdv=$do_cvdv bosonic=$do_bosonic qcvdv=$do_qcvdv output=$output_json"

	cmd=("$PYTHON_BIN" "$RUNNER_PY" \
		--dv-qubits "$dv" \
		--cv-qubits "$cv" \
		--runs "$RUNS" \
		--warmup "$WARMUP" \
		--lam "$LAM" \
		--qcvdv-methods "$QCVDV_METHODS_CSV" \
		--output "$output_json")

	if [[ "$do_cvdv" == "0" ]]; then
		cmd+=(--no-cvdv)
	fi
	if [[ "$do_bosonic" == "0" ]]; then
		cmd+=(--no-bosonic)
	fi
	if [[ "$do_qcvdv" == "0" ]]; then
		cmd+=(--no-qcvdv)
	fi

	"${cmd[@]}"
done < "$CHUNK_FILE"
EOF
sed -i \
	-e "s/__CPUS_PER_TASK__/${CPUS_PER_TASK}/g" \
	-e "s/__GPUS_PER_JOB__/${GPUS_PER_JOB}/g" \
	"$SBATCH_SCRIPT"
chmod +x "$SBATCH_SCRIPT"

echo "[INFO] Campaign root: $CAMPAIGN_ROOT"
echo "[INFO] Jobs requested: $NUM_JOBS"
echo "[INFO] Jobs to submit: $job_count"
echo "[INFO] Results will be stored under: $RUNS_DIR"
echo "[INFO] Manifest: $MANIFEST_FILE"

for ((i = 0; i < job_count; i++)); do
	chunk_file="${CHUNK_FILES[$i]}"
	chunk_rows=$(wc -l < "$chunk_file" | tr -d ' ')
	load="${JOB_LOADS[$i]}"
	job_name="same-final-$(printf '%02d' "$i")"
	log_prefix="$LOGS_DIR/${job_name}"
	printf '%s\t%s\t%s\n' "$job_name" "$chunk_file" "$load" >> "$CHUNK_SPECS_FILE"
	echo "[INFO] chunk $i: rows=$chunk_rows load=$load file=$chunk_file"

	sbatch_cmd=(sbatch --job-name="$job_name" \
		--account="$ACCOUNT" \
		--time="$TIME_LIMIT" \
		--constraint="$CONSTRAINT" \
		--output="$log_prefix.out" \
		--error="$log_prefix.err" \
		--export=ALL,SCRIPT_DIR="$SCRIPT_DIR",RUNNER_PY="$SCRIPT_DIR/same_final_circuits_run.py",CHUNK_FILE="$chunk_file",RUNS_DIR="$RUNS_DIR",PYTHON_BIN="$PYTHON_BIN",QCVDV_METHODS_CSV="$QCVDV_METHODS_CSV",RUNS="$RUNS",WARMUP="$WARMUP",LAM="$LAM",VENV_ACTIVATE="$VENV_ACTIVATE")

	if [[ -n "$QOS" ]]; then
		sbatch_cmd+=(--qos="$QOS")
	fi
	if [[ -n "$PARTITION" ]]; then
		sbatch_cmd+=(--partition="$PARTITION")
	fi
	sbatch_cmd+=("$SBATCH_SCRIPT")

	if (( DRY_RUN == 1 )); then
		echo "${sbatch_cmd[@]}"
		continue
	fi

	"${sbatch_cmd[@]}"
done

echo "[INFO] Submitted all chunks."
