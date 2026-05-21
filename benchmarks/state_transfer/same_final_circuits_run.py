"""
Same-Final-Circuit Runtime Harness

Compares runtime across CVDV, bosonic-qiskit, and qcvdv using aligned
state-transfer circuit intent (same DV gate pattern and parameters).

Important:
- This benchmark is for runtime/scaling only.
- CV state representations differ across backends (grid vs Fock basis), so
  states are not physically identical even when parameters are aligned.
"""

import argparse
import json
import os
import time
from datetime import datetime
from statistics import mean, pstdev

import numpy as np
from numpy import pi, sqrt

from bench_cvdv import run_cvdv_transfer_experiment


def _build_cat_state(cutoff, alpha0=-1.0, alpha1=2.0):
	"""Build normalized cat coefficients in Fock basis."""
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


def _build_shared_c2qa_circuit(dv_qubits, cv_qubits, lam):
	"""Build the state-transfer circuit with the same gate order/parameters as bench_cvdv."""
	from qiskit import QuantumRegister
	from bosonic_qiskit import CVCircuit, QumodeRegister

	qmr = QumodeRegister(1, num_qubits_per_qumode=cv_qubits, name="cv")
	qbr = QuantumRegister(dv_qubits, name="dv")
	circ = CVCircuit(qbr, qmr)

	# setUniform(0) in CVDV corresponds to |+>^n, represented here by Hadamards.
	for i in range(dv_qubits):
		circ.h(qbr[i])

	cat = _build_cat_state(cutoff=2 ** int(cv_qubits), alpha0=-1.0, alpha1=2.0)
	circ.cv_initialize(cat, qmr[0])

	for k in range(1, dv_qubits + 1):
		qubit_idx = k - 1

		v_k = -pi / (2 * lam * (1 << k))
		circ.rx(pi / 2, qbr[qubit_idx])
		circ.cv_c_d(1j * v_k / sqrt(2), qmr[0], qbr[qubit_idx])
		circ.rx(-pi / 2, qbr[qubit_idx])

		w_k = lam * (1 << k) / 2 * (-1 if k == dv_qubits else 1)
		circ.cv_c_d(-w_k / sqrt(2), qmr[0], qbr[qubit_idx])

	return circ


def _run_bosonic_from_shared_circuit(dv_qubits, cv_qubits, lam):
	"""Run bosonic backend using the shared frontend circuit used by qcvdv conversion."""
	from bench_bosonic import _simulate_with_timing

	t_start = time.perf_counter()
	t_build0 = time.perf_counter()
	circ = _build_shared_c2qa_circuit(dv_qubits=dv_qubits, cv_qubits=cv_qubits, lam=lam)
	build_time = time.perf_counter() - t_build0

	_, _, transpile_time, run_time = _simulate_with_timing(circ, shots=1)
	t_total = time.perf_counter() - t_start

	return {
		"time": t_total,
		"build_time": build_time,
		"transpile_time": float(transpile_time),
		"run_time": float(run_time),
	}


def _run_qcvdv_from_shared_circuit(dv_qubits, cv_qubits, lam, method):
	"""Run qcvdv backend by converting the shared frontend circuit."""
	from bench_qcvdv import _normalize_backend, _pick_hyb_circ_class
	from qcvdv.circuit import from_CVCircuit
	from qcvdv.simulator import HybridSimulator

	backend = _normalize_backend(method)

	t0 = time.perf_counter()
	c2qa_circ = _build_shared_c2qa_circuit(dv_qubits=dv_qubits, cv_qubits=cv_qubits, lam=lam)
	build_time = time.perf_counter() - t0

	HybCirc = _pick_hyb_circ_class(backend)
	t1 = time.perf_counter()
	hc = from_CVCircuit(c2qa_circ, hyb_circ=HybCirc)
	convert_time = time.perf_counter() - t1

	t2 = time.perf_counter()
	HybridSimulator(method=backend).run(hc, shots=1)
	run_time = time.perf_counter() - t2

	return {
		"time": build_time + run_time,
		"time_with_convert": build_time + convert_time + run_time,
		"build_time": build_time,
		"convert_time": convert_time,
		"run_time": run_time,
	}


def _safe_stats(values):
	if not values:
		return None
	if len(values) == 1:
		v = float(values[0])
		return {
			"mean": v,
			"std": 0.0,
			"min": v,
			"max": v,
			"all": [v],
		}
	vals = [float(v) for v in values]
	return {
		"mean": float(mean(vals)),
		"std": float(pstdev(vals)),
		"min": float(min(vals)),
		"max": float(max(vals)),
		"all": vals,
	}


def _normalize_result(backend_name, method, cv_qubits, raw):
	"""Map per-backend run output to a common timing schema in seconds."""
	if backend_name == "cvdv":
		build_time = float(raw.get("build_time", 0.0))
		compile_time = 0.0
		run_time = float(raw.get("run_time", 0.0))
		total_time = float(raw.get("time", build_time + run_time))
	elif backend_name == "bosonic":
		build_time = float(raw.get("build_time", 0.0))
		compile_time = float(raw.get("transpile_time", 0.0))
		run_time = float(raw.get("run_time", raw.get("simulate_time", 0.0)))
		total_time = float(raw.get("time", build_time + compile_time + run_time))
	elif backend_name == "qcvdv":
		build_time = float(raw.get("build_time", 0.0))
		compile_time = float(raw.get("convert_time", 0.0))
		run_time = float(raw.get("run_time", 0.0))
		total_time = float(raw.get("time_with_convert", build_time + compile_time + run_time))
	else:
		raise ValueError(f"Unknown backend: {backend_name}")

	return {
		"backend": backend_name,
		"method": method,
		"cv_qubits": int(cv_qubits),
		"cv_dimension": int(2 ** cv_qubits),
		"build_time": build_time,
		"compile_time": compile_time,
		"run_time": run_time,
		"total_time": total_time,
		"success": True,
		"error": None,
	}


def _make_error_result(backend_name, method, cv_qubits, err_msg):
	return {
		"backend": backend_name,
		"method": method,
		"cv_qubits": int(cv_qubits),
		"cv_dimension": int(2 ** cv_qubits),
		"build_time": None,
		"compile_time": None,
		"run_time": None,
		"total_time": None,
		"success": False,
		"error": str(err_msg),
	}


def _run_one_backend(backend_name, method, dv_qubits, cv_qubits, lam):
	"""Run one backend once and return normalized schema."""
	try:
		if backend_name == "cvdv":
			raw = run_cvdv_transfer_experiment(
				n_dv_qubits=dv_qubits,
				cv_qubits=cv_qubits,
				lam=lam,
				return_plots=False,
			)
			return _normalize_result(backend_name, method, cv_qubits, raw)

		if backend_name == "bosonic":
			raw = _run_bosonic_from_shared_circuit(
				dv_qubits=dv_qubits,
				cv_qubits=cv_qubits,
				lam=lam,
			)
			return _normalize_result(backend_name, method, cv_qubits, raw)

		if backend_name == "qcvdv":
			raw = _run_qcvdv_from_shared_circuit(
				dv_qubits=dv_qubits,
				cv_qubits=cv_qubits,
				lam=lam,
				method=method,
			)
			return _normalize_result(backend_name, method, cv_qubits, raw)

		return _make_error_result(backend_name, method, cv_qubits, f"unsupported backend: {backend_name}")
	except Exception as exc:
		return _make_error_result(backend_name, method, cv_qubits, exc)


def _summarize_runs(run_results):
	successful = [r for r in run_results if r.get("success")]
	return {
		"success_count": len(successful),
		"failure_count": len(run_results) - len(successful),
		"build": _safe_stats([r["build_time"] for r in successful]),
		"compile": _safe_stats([r["compile_time"] for r in successful]),
		"run": _safe_stats([r["run_time"] for r in successful]),
		"total": _safe_stats([r["total_time"] for r in successful]),
		"errors": [r["error"] for r in run_results if not r.get("success")],
	}


def _print_summary_table(results):
	print("\n" + "=" * 100)
	print("Same-Final-Circuit Runtime Summary (total includes compile/convert)")
	print("=" * 100)
	print("Notes: aligned circuit intent; CV basis differs by backend (timing benchmark, not state-equivalence).")
	print("-" * 100)
	header = (
		f"{'Config':<12} {'Backend':<10} {'Method':<22} {'Build(ms)':>10} "
		f"{'Compile(ms)':>12} {'Run(ms)':>10} {'Total(ms)':>11} {'OK':>4}"
	)
	print(header)
	print("-" * 100)

	configs = results.get("results", {})
	for cv_qubits_str in sorted(configs.keys(), key=lambda x: int(x)):
		block = configs[cv_qubits_str]
		cfg_label = f"cv={cv_qubits_str}"

		for entry in block.get("backends", []):
			summary = entry.get("summary", {})
			total = (summary.get("total") or {}).get("mean")
			build = (summary.get("build") or {}).get("mean")
			comp = (summary.get("compile") or {}).get("mean")
			run = (summary.get("run") or {}).get("mean")
			ok = summary.get("success_count", 0)

			def ms(x):
				return f"{x*1000:,.2f}" if x is not None else "n/a"

			print(
				f"{cfg_label:<12} {entry['backend']:<10} {str(entry.get('method') or '-'): <22} "
				f"{ms(build):>10} {ms(comp):>12} {ms(run):>10} {ms(total):>11} {ok:>4}"
			)
	print("=" * 100 + "\n")


def run_same_final_circuits_benchmark(
	dv_qubits=4,
	cv_qubits_list=None,
	n_runs=10,
	warmup=2,
	lam=0.29,
	include_bosonic=True,
	include_qcvdv=True,
	qcvdv_methods=None,
):
	if cv_qubits_list is None:
		cv_qubits_list = [10, 11, 12]
	if qcvdv_methods is None:
		qcvdv_methods = ["dense_matrix_gpu", "dense_matrix_gpuv1", "torch"]

	benchmark = {
		"timestamp": datetime.now().isoformat(),
		"benchmark": "same_final_circuits_runtime",
		"disclaimer": (
			"Runtime comparison uses aligned state-transfer circuit intent with matched parameters. "
			"CV representations differ across backends (grid vs Fock), so this output should be "
			"interpreted as performance/scaling data, not strict physical state equivalence."
		),
		"config": {
			"dv_qubits": int(dv_qubits),
			"cv_qubits_list": [int(x) for x in cv_qubits_list],
			"n_runs": int(n_runs),
			"warmup": int(warmup),
			"lam": float(lam),
			"total_time_policy": "includes compile/convert",
		},
		"results": {},
	}

	for cv_qubits in cv_qubits_list:
		cv_key = str(int(cv_qubits))
		backends = [("cvdv", None)]
		if include_bosonic:
			backends.append(("bosonic", None))
		if include_qcvdv:
			for m in qcvdv_methods:
				backends.append(("qcvdv", m))

		block = {
			"cv_qubits": int(cv_qubits),
			"cv_dimension": int(2 ** cv_qubits),
			"backends": [],
		}

		for backend_name, method in backends:
			runs = []

			for _ in range(warmup):
				_run_one_backend(backend_name, method, dv_qubits, cv_qubits, lam)

			for _ in range(n_runs):
				runs.append(_run_one_backend(backend_name, method, dv_qubits, cv_qubits, lam))

			block["backends"].append(
				{
					"backend": backend_name,
					"method": method,
					"runs": runs,
					"summary": _summarize_runs(runs),
				}
			)

		benchmark["results"][cv_key] = block

	return benchmark


def _parse_int_list(csv_text):
	return [int(x.strip()) for x in csv_text.split(",") if x.strip()]


def _parse_str_list(csv_text):
	return [x.strip() for x in csv_text.split(",") if x.strip()]


def main():
	parser = argparse.ArgumentParser(description="Run runtime benchmarks on aligned final circuits")
	parser.add_argument("--dv-qubits", type=int, default=4, help="Number of DV qubits")
	parser.add_argument(
		"--cv-qubits",
		type=str,
		default="10,11,12",
		help="Comma-separated CV qubit list, e.g. 10,11,12",
	)
	parser.add_argument("--runs", type=int, default=10, help="Number of timed runs")
	parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
	parser.add_argument("--lam", type=float, default=0.29, help="Interaction parameter lambda")
	parser.add_argument(
		"--qcvdv-methods",
		type=str,
		default="dense_matrix_gpu,dense_matrix_gpuv1,torch",
		help="Comma-separated qcvdv methods",
	)
	parser.add_argument("--no-bosonic", action="store_true", help="Skip bosonic backend")
	parser.add_argument("--no-qcvdv", action="store_true", help="Skip qcvdv backend")
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Output JSON path (default: results/same_final_circuits_runtime.json)",
	)
	args = parser.parse_args()

	cv_qubits_list = _parse_int_list(args.cv_qubits)
	qcvdv_methods = _parse_str_list(args.qcvdv_methods)

	results = run_same_final_circuits_benchmark(
		dv_qubits=args.dv_qubits,
		cv_qubits_list=cv_qubits_list,
		n_runs=args.runs,
		warmup=args.warmup,
		lam=args.lam,
		include_bosonic=not args.no_bosonic,
		include_qcvdv=not args.no_qcvdv,
		qcvdv_methods=qcvdv_methods,
	)

	out_path = args.output
	if out_path is None:
		out_path = os.path.join(os.path.dirname(__file__), "results", "same_final_circuits_runtime.json")

	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(results, f, indent=2)

	_print_summary_table(results)
	print(f"Saved: {out_path}")


if __name__ == "__main__":
	main()
