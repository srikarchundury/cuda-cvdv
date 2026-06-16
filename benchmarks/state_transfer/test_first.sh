#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Smoke test: run each method with small params and check for ITER output.
# Calls bench_cvdv.py / bench_bosonic.py / bench_qcvdv.py in this directory.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/pscratch/sd/s/schundu3/projects/cvdv/experiments_gpu/expQcvdvVirtEnv/bin/activate}"

if [[ -f "$VENV_ACTIVATE" ]]; then
	# shellcheck disable=SC1090
	source "$VENV_ACTIVATE"
fi

N_DV_QUBITS="${N_DV_QUBITS:-2}"
CV_QUBITS="${CV_QUBITS:-4}"          # bench_cvdv.py / bench_qcvdv.py: exponent
CV_CUTOFF=$(( 1 << CV_QUBITS ))      # bench_bosonic.py: actual Fock cutoff = 2^CV_QUBITS
LAM="${LAM:-0.29}"
ITERS="${ITERS:-2}"
WARMUP="${WARMUP:-1}"

pass=0
fail=0
failed_methods=()

run_and_check() {
	local method="$1"
	local cmd=("${@:2}")

	echo ""
	echo "--- method: ${method} ---"
	set +e
	output=$("${cmd[@]}" 2>&1)
	rc=$?
	set -e
	echo "$output"

	iter_count=$(echo "$output" | grep -c " ITER " 2>/dev/null || true)
	has_done=$(echo "$output"  | grep -c "=== DONE ===" 2>/dev/null || true)

	if [[ $rc -eq 0 && "$iter_count" -ge "$ITERS" && "$has_done" -ge 1 ]]; then
		echo "[PASS] ${method}: ${iter_count} ITER lines"
		(( pass += 1 ))
	else
		echo "[FAIL] ${method}: rc=${rc} iter_count=${iter_count} done=${has_done}"
		(( fail += 1 ))
		failed_methods+=("$method")
	fi
}

echo "=== App4 smoke test ==="
echo "n_dv_qubits=${N_DV_QUBITS}  cv_qubits=${CV_QUBITS}  cv_cutoff=${CV_CUTOFF}  lam=${LAM}"
echo "iters=${ITERS}  warmup=${WARMUP}"

# cuda-cvdv
run_and_check "cuda-cvdv" \
	"$PYTHON_BIN" bench_cvdv.py \
	--dv-qubits "$N_DV_QUBITS" --cv-qubits "$CV_QUBITS" \
	--runs "$ITERS" --warmup "$WARMUP"

# c2qa_gpu (bosonic-qiskit — uses Fock cutoff, not qubit exponent)
run_and_check "c2qa_gpu" \
	"$PYTHON_BIN" bench_bosonic.py \
	--dv-qubits "$N_DV_QUBITS" --cv-cutoff "$CV_CUTOFF" \
	--runs "$ITERS" --warmup "$WARMUP"

# qcvdv backends
for qmethod in qcvdv_scipy_gpu qcvdv_eigen_gpu qcvdv_eigen_tensor_gpu qcvdv_torch qcvdv_torch_tensor_gpu qcvdv_diaq_gpu; do
	run_and_check "$qmethod" \
		"$PYTHON_BIN" bench_qcvdv.py \
		--dv-qubits "$N_DV_QUBITS" --cv-qubits "$CV_QUBITS" \
		--method "$qmethod" \
		--runs "$ITERS" --warmup "$WARMUP"
done

echo ""
echo "=== Summary: ${pass} passed, ${fail} failed ==="
if [[ ${fail} -gt 0 ]]; then
	echo "Failed methods: ${failed_methods[*]}"
	exit 1
fi
