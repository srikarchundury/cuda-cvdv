"""
QCVDV Benchmarks - CV-DV State Transfer Algorithm

Benchmarks qcvdv backends by building a bosonic-qiskit frontend circuit,
converting it to qcvdv, and running HybridSimulator.
"""

import time
import os
import sys
import io
import re
import contextlib
import numpy as np
from numpy import pi, sqrt

QCVDV_GPU_METHODS = [
    "c2qa_gpu",
    "qcvdv_scipy_gpu",
    "qcvdv_eigen_gpu",
    "qcvdv_eigen_tensor_gpu",
    "qcvdv_torch",
    "qcvdv_torch_tensor_gpu",
    "qcvdv_diaq_gpu",
]
PROFILER_COMPONENT_KEYS = [
	"matrix_generation",
	"apply",
	"build",
	"transfer",
	"cache_hit",
	"other",
	"total",
]


def _parse_profiler_breakdown(stdout_text: str):
	"""Parse profiler summary line from simulator stdout into seconds."""
	for line in reversed(stdout_text.splitlines()):
		if "[PROFILER" not in line:
			continue
		pairs = re.findall(r"([a-zA-Z_]+)=([0-9]*\.?[0-9]+)s", line)
		if not pairs:
			continue
		parsed = {}
		for key, value in pairs:
			try:
				parsed[key] = float(value)
			except ValueError:
				continue
		if parsed:
			return parsed
	return {}


def _ensure_qcvdv_on_path():
	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
	qcvdv_python = os.path.join(repo_root, 'qcvdv', 'python')
	if qcvdv_python not in sys.path and os.path.isdir(qcvdv_python):
		sys.path.insert(0, qcvdv_python)


def _normalize_backend(method: str) -> str:
	m = method.strip().lower()
	if m.startswith("qcvdv:"):
		m = m.split(":", 1)[1].strip()
	return m


def _clear_cache_if_possible(hc):
	try:
		hc.clear_gate_cache()
	except Exception:
		pass


def _actual_cutoff_from_cv_qubits(cv_qubits: int) -> int:
	return 2 ** int(cv_qubits)


def _build_cat_state(cutoff: int, alpha0: float = -1.0, alpha1: float = 2.0):
	cat0 = np.zeros(cutoff, dtype=np.complex128)
	cat1 = np.zeros(cutoff, dtype=np.complex128)

	cat0[0] = np.exp(-abs(alpha0) ** 2 / 2)
	cat1[0] = np.exp(-abs(alpha1) ** 2 / 2)

	for n in range(1, cutoff):
		cat0[n] = cat0[n - 1] * alpha0 / np.sqrt(n)
		cat1[n] = cat1[n - 1] * alpha1 / np.sqrt(n)

	cat = cat0 + cat1
	cat /= np.linalg.norm(cat)
	return cat


def _build_c2qa_circuit(n_dv_qubits: int, cv_qubits: int, lam: float):
	import bosonic_qiskit as c2qa
	from qiskit import QuantumRegister

	qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=cv_qubits, name="cv")
	qbr = QuantumRegister(n_dv_qubits, name="dv")
	circ = c2qa.CVCircuit(qbr, qmr)

	for i in range(n_dv_qubits):
		circ.h(qbr[i])

	cat = _build_cat_state(cutoff=_actual_cutoff_from_cv_qubits(cv_qubits), alpha0=-1.0, alpha1=2.0)
	circ.cv_initialize(cat, qmr[0])

	for k in range(1, n_dv_qubits + 1):
		qubit_idx = k - 1

		v_k = -pi / (2 * lam * (1 << k))
		circ.rx(pi / 2, qbr[qubit_idx])
		circ.cv_c_d(1j * v_k / sqrt(2), qmr[0], qbr[qubit_idx])
		circ.rx(-pi / 2, qbr[qubit_idx])

		w_k = lam * (1 << k) / 2 * (-1 if k == n_dv_qubits else 1)
		circ.cv_c_d(-w_k / sqrt(2), qmr[0], qbr[qubit_idx])

	return circ


def run_qcvdv_transfer_experiment(
	n_dv_qubits=4,
	cv_qubits=10,
	lam=0.29,
	method="eigen_cpu",
	shots=1,
	clear_cache_each_run=False,
):
	"""
	Run CV-to-DV state transfer experiment once using qcvdv.

	Returns dict with timing split and total time:
	  - time: build + run (excludes conversion)
	  - time_with_convert: build + convert + run
	  - build_time: c2qa frontend circuit construction
	  - convert_time: c2qa -> qcvdv conversion
	  - run_time: qcvdv simulation time
	  - transpile_time: always 0 for qcvdv
	"""
	backend = _normalize_backend(method)
	if backend not in QCVDV_GPU_METHODS:
		raise ValueError(f"Unknown qcvdv backend '{backend}'. Expected one of: {QCVDV_GPU_METHODS}")

	backend = backend.replace("qcvdv_", "")
	_ensure_qcvdv_on_path()
	from qcvdv.circuit import from_CVCircuit
	from qcvdv.simulator import HybridSimulator

	t0 = time.perf_counter()
	c2qa_circ = _build_c2qa_circuit(n_dv_qubits, cv_qubits, lam)
	build_time = time.perf_counter() - t0

	t1 = time.perf_counter()
	hc = from_CVCircuit(c2qa_circ)
	convert_time = time.perf_counter() - t1

	if clear_cache_each_run:
		_clear_cache_if_possible(hc)

	t2 = time.perf_counter()
	stdout_capture = io.StringIO()
	with contextlib.redirect_stdout(stdout_capture):
		state_vec = HybridSimulator(method=backend).run(hc, shots=shots)
	run_time = time.perf_counter() - t2
	profiler_breakdown = _parse_profiler_breakdown(stdout_capture.getvalue())
	if stdout_capture.getvalue():
		print(stdout_capture.getvalue(), end="")

	total_time = build_time + run_time
	total_with_convert_time = build_time + convert_time + run_time
	transpile_time = 0.0

	return {
		"time": total_time,
		"time_with_convert": total_with_convert_time,
		"build_time": build_time,
		"convert_time": convert_time,
		"run_time": run_time,
		"run_breakdown": profiler_breakdown,
		"transpile_time": transpile_time,
		"state": state_vec,
		"backend": backend,
	}


def benchmark_qcvdv_transfer(
	n_dv_qubits=4,
	cv_qubits=10,
	n_runs=10,
	warmup=2,
	method="eigen_cpu",
	shots=1,
	clear_cache_each_run=False,
):
	"""Benchmark CV-to-DV state transfer using qcvdv."""
	lam = 0.29

	total_times = []
	total_with_convert_times = []
	build_times = []
	convert_times = []
	run_times = []
	transpile_times = []
	profiler_component_times = {k: [] for k in PROFILER_COMPONENT_KEYS}

	for _ in range(warmup):
		run_qcvdv_transfer_experiment(
			n_dv_qubits=n_dv_qubits,
			cv_qubits=cv_qubits,
			lam=lam,
			method=method,
			shots=shots,
			clear_cache_each_run=clear_cache_each_run,
		)

	for _ in range(n_runs):
		res = run_qcvdv_transfer_experiment(
			n_dv_qubits=n_dv_qubits,
			cv_qubits=cv_qubits,
			lam=lam,
			method=method,
			shots=shots,
			clear_cache_each_run=clear_cache_each_run,
		)
		total_times.append(res["time"])
		total_with_convert_times.append(res.get("time_with_convert", res["time"]))
		build_times.append(res["build_time"])
		convert_times.append(res["convert_time"])
		run_times.append(res["run_time"])
		transpile_times.append(res["transpile_time"])
		for key in PROFILER_COMPONENT_KEYS:
			profiler_component_times[key].append(float(res.get("run_breakdown", {}).get(key, 0.0)))

	backend = _normalize_backend(method)
	results = {
		"mean": np.mean(total_times),
		"std": np.std(total_times),
		"min": np.min(total_times),
		"max": np.max(total_times),
		"all": total_times,
		"with_convert_mean": np.mean(total_with_convert_times),
		"with_convert_std": np.std(total_with_convert_times),
		"with_convert_min": np.min(total_with_convert_times),
		"with_convert_max": np.max(total_with_convert_times),
		"build_mean": np.mean(build_times),
		"build_std": np.std(build_times),
		"build_min": np.min(build_times),
		"build_max": np.max(build_times),
		"convert_mean": np.mean(convert_times),
		"convert_std": np.std(convert_times),
		"convert_min": np.min(convert_times),
		"convert_max": np.max(convert_times),
		"run_mean": np.mean(run_times),
		"run_std": np.std(run_times),
		"run_min": np.min(run_times),
		"run_max": np.max(run_times),
		"transpile_mean": np.mean(transpile_times),
		"transpile_std": np.std(transpile_times),
		"transpile_min": np.min(transpile_times),
		"transpile_max": np.max(transpile_times),
		"transpile_all": transpile_times,
		"config": {
			"n_dv_qubits": n_dv_qubits,
			"cv_qubits": cv_qubits,
			"cv_dimension": 2 ** cv_qubits,
			"method": backend,
			"shots": shots,
			"n_runs": n_runs,
			"warmup": warmup,
			"clear_cache_each_run": clear_cache_each_run,
		},
	}
	for key in PROFILER_COMPONENT_KEYS:
		vals = profiler_component_times[key]
		results[f"run_{key}_mean"] = np.mean(vals)
		results[f"run_{key}_std"] = np.std(vals)
		results[f"run_{key}_min"] = np.min(vals)
		results[f"run_{key}_max"] = np.max(vals)

	return results


def print_results(results):
	"""Print benchmark results in a readable format."""
	cfg = results["config"]
	print(f"\nQCVDV Configuration ({cfg['method']}):")
	print(f"  DV qubits: {cfg['n_dv_qubits']}")
	print(f"  CV qubits: {cfg['cv_qubits']} (dim={cfg['cv_dimension']})")
	print(f"  Shots: {cfg['shots']}")
	print(f"  Runs: {cfg['n_runs']} (+ {cfg['warmup']} warmup)")
	print("-" * 60)
	print(f"Total (no convert): {results['mean']*1000:7.2f} +- {results['std']*1000:5.2f} ms  "
		  f"[{results['min']*1000:6.2f}, {results['max']*1000:6.2f}]")
	print(f"Total (with convert): {results['with_convert_mean']*1000:7.2f} +- {results['with_convert_std']*1000:5.2f} ms  "
		  f"[{results['with_convert_min']*1000:6.2f}, {results['with_convert_max']*1000:6.2f}]")
	print(f"Build:      {results['build_mean']*1000:7.2f} +- {results['build_std']*1000:5.2f} ms")
	print(f"Convert:    {results['convert_mean']*1000:7.2f} +- {results['convert_std']*1000:5.2f} ms")
	print(f"Run:        {results['run_mean']*1000:7.2f} +- {results['run_std']*1000:5.2f} ms")
	print(f"Transpile:  {results['transpile_mean']*1000:7.2f} +- {results['transpile_std']*1000:5.2f} ms")
	print()


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Benchmark qcvdv state transfer')
	parser.add_argument('--dv-qubits', type=int, default=4)
	parser.add_argument('--cv-qubits', type=int, default=10)
	parser.add_argument('--method', type=str, default='eigen_cpu',
						help=f"qcvdv backend: {', '.join(QCVDV_GPU_METHODS)}")
	parser.add_argument('--shots', type=int, default=1)
	parser.add_argument('--runs', type=int, default=10)
	parser.add_argument('--warmup', type=int, default=2)
	parser.add_argument('--clear-cache-each-run', action='store_true', default=False)
	args = parser.parse_args()

	lam = 0.29
	method = args.method
	print(f"[INFO] n_dv_qubits={args.dv_qubits} cv_qubits={args.cv_qubits} lam={lam}")
	print(f"[INFO] method={method} shots={args.shots} iters={args.runs} warmup={args.warmup}")

	for i in range(args.warmup):
		res = run_qcvdv_transfer_experiment(
			n_dv_qubits=args.dv_qubits, cv_qubits=args.cv_qubits, lam=lam,
			method=method, shots=args.shots, clear_cache_each_run=args.clear_cache_each_run,
		)
		total = res['build_time'] + res['convert_time'] + res['run_time']
		print(f"{method} WARMUP {i+1}/{args.warmup}: build={res['build_time']:.9f} sec  convert={res['convert_time']:.9f} sec  run={res['run_time']:.9f} sec  transpile={res['transpile_time']:.9f} sec  total={total:.9f} sec")

	for i in range(args.runs):
		res = run_qcvdv_transfer_experiment(
			n_dv_qubits=args.dv_qubits, cv_qubits=args.cv_qubits, lam=lam,
			method=method, shots=args.shots, clear_cache_each_run=args.clear_cache_each_run,
		)
		total = res['build_time'] + res['convert_time'] + res['run_time']
		print(f"{method} ITER {i+1}/{args.runs}: build={res['build_time']:.9f} sec  convert={res['convert_time']:.9f} sec  run={res['run_time']:.9f} sec  transpile={res['transpile_time']:.9f} sec  total={total:.9f} sec")

	print("=== DONE ===")
