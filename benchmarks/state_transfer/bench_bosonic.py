"""
Bosonic-Qiskit Benchmarks - CV-DV State Transfer Algorithm
Comparison benchmark using bosonic-qiskit library

Note: bosonic-qiskit uses Fock basis encoding (CPU-based numpy/scipy)
      while CVDV uses position wave function encoding (GPU-based CUDA)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt


def _simulate_with_timing(circuit, shots):
    """Simulate with explicit transpile/run timing on Aer GPU.

    This mirrors bosonic_qiskit.util.simulate behavior for our benchmark use case,
    but exposes transpile and simulator run timing separately.
    """
    import qiskit
    import qiskit_aer
    from qiskit.quantum_info import Statevector

    sim_circuit = circuit.copy()
    sim_circuit.save_statevector()

    circuit_compiled = sim_circuit
    simulator = qiskit_aer.AerSimulator(device="GPU")

    transpile_time = 0.0
    if circuit_compiled.requires_transpile():
        t_transpile_start = time.perf_counter()
        pm = qiskit.transpiler.preset_passmanagers.common.generate_translation_passmanager(
            target=simulator.target
        )
        circuit_compiled = pm.run(circuit_compiled)
        transpile_time = time.perf_counter() - t_transpile_start

    t_run_start = time.perf_counter()
    job = simulator.run(
        circuit_compiled,
        shots=shots,
        max_parallel_threads=0,
        noise_model=None,
    )
    aer_result = job.result()
    run_time = time.perf_counter() - t_run_start

    if not aer_result.success:
        # Surface the Aer error message (e.g. CUDA out-of-memory / chunk size)
        msg = getattr(aer_result, 'status', '') or str(aer_result)
        raise RuntimeError(f"Aer simulation failed: {msg}")

    state = Statevector(aer_result.get_statevector(circuit_compiled))
    return state, aer_result, transpile_time, run_time


def _force_aer_gpu_if_possible():
    """Force qiskit-aer simulator construction to use GPU when possible.

    Some bosonic_qiskit.util.simulate versions do not expose a device argument,
    so we patch AerSimulator constructor behavior instead.
    """
    try:
        import qiskit_aer
    except Exception:
        return

    if getattr(qiskit_aer, "_qcvdv_gpu_forced", False):
        return

    original_aer_simulator = qiskit_aer.AerSimulator

    def _gpu_aer_simulator(*args, **kwargs):
        kwargs.setdefault("device", "GPU")
        return original_aer_simulator(*args, **kwargs)

    qiskit_aer.AerSimulator = _gpu_aer_simulator
    qiskit_aer._qcvdv_gpu_forced = True

def run_bosonic_transfer_experiment(n_dv_qubits=4, cv_cutoff=64, lam=0.29, return_plots=False):
    """
    Run CV-to-DV state transfer experiment once using bosonic-qiskit.
    
    Args:
        n_dv_qubits: Number of DV qubits for encoding
        cv_cutoff: Fock space cutoff for CV mode (Hilbert space size)
        lam: Interaction parameter
        return_plots: If True, include initial/final state data for plotting
    
    Returns:
        dict: Contains 'time' and optionally 'plots' with initial/final state data
    
    Note:
        Requires: pip install bosonic-qiskit qiskit qiskit-aer
    """
    try:
        _force_aer_gpu_if_possible()
        from bosonic_qiskit import CVCircuit, QumodeRegister
        from bosonic_qiskit.util import trace_out_qubits
        from bosonic_qiskit.wigner import wigner
        from qiskit import QuantumRegister
    except ImportError as e:
        print(f"Error: {e}")
        print("Install bosonic-qiskit with: pip install bosonic-qiskit qiskit qiskit-aer")
        return None
    
    t_start = time.perf_counter()
    
    # Convert cutoff dimension to number of qubits
    build_start = time.perf_counter()
    cv_num_qubits = int(np.ceil(np.log2(cv_cutoff)))
    actual_cutoff = 2**cv_num_qubits
    
    # Create registers: DV (discrete) + CV (bosonic)
    dv_reg = QuantumRegister(n_dv_qubits, 'dv')
    cv_reg = QumodeRegister(num_qumodes=1, num_qubits_per_qumode=cv_num_qubits, name='cv')
    circuit = CVCircuit(dv_reg, cv_reg)
    
    # Initialize DV register to |++++> (uniform superposition)
    for i in range(n_dv_qubits):
        circuit.h(dv_reg[i])

    # Initialize CV register to cat state (|α=0⟩ + |α=2.0⟩)
    # Use numerically robust recurrence relation: c_n = α/√n · c_{n-1}
    alpha0 = -1.0
    alpha1 = 2.0
    cat0 = np.zeros(actual_cutoff, dtype=np.complex128)
    cat1 = np.zeros(actual_cutoff, dtype=np.complex128)
    
    # Initialize n=0 term
    cat0[0] = np.exp(-abs(alpha0)**2/2)
    cat1[0] = np.exp(-abs(alpha1)**2/2)
    for n in range(1, actual_cutoff):
        cat0[n] = cat0[n-1] * alpha0 / np.sqrt(n)
        cat1[n] = cat1[n-1] * alpha1 / np.sqrt(n)
    
    # Form cat state and normalize
    cat = cat0 + cat1
    cat = cat / np.linalg.norm(cat)
    circuit.cv_initialize(cat, cv_reg[0])
    
    # Capture initial state if plots are requested
    if return_plots:
        circuit_initial = circuit.copy()
        state_initial, _, _, _ = _simulate_with_timing(circuit_initial, shots=1)
        probs_dv_initial = state_initial.probabilities(list(range(n_dv_qubits)))
        rho_cv_initial = trace_out_qubits(circuit_initial, state_initial)
        wigner_cv_initial = wigner(state=rho_cv_initial, axes_min=-5, axes_max=5, axes_steps=101)
    
    # Apply encoding circuit
    for k in range(1, n_dv_qubits + 1):
        qubit_idx = k - 1
        
        # V_k: exp(i·v_k·q·σ_y) rotation
        v_k = -pi / (2 * lam * (1 << k))

        # Decompose into Pauli X rotation + conditional displacement
        circuit.rx(pi/2, dv_reg[qubit_idx])
        circuit.cv_c_d(
            1j * v_k / sqrt(2),  # theta displacement
            cv_reg[0],
            dv_reg[qubit_idx]
        )
        circuit.rx(-pi/2, dv_reg[qubit_idx])
        
        # W_k: exp(i·w_k·p·σ_x) rotation
        w_k = lam * (1 << k) / 2 * (-1 if k == n_dv_qubits else 1)
        circuit.cv_c_d(
            -w_k / sqrt(2),  # theta displacement
            cv_reg[0],
            dv_reg[qubit_idx]
        )
    build_time = time.perf_counter() - build_start
    
    # Explicit transpile/run timing split.
    try:
        statevector, _, transpile_time, run_time = _simulate_with_timing(circuit, shots=1)
    except Exception as e:
        print(f"Simulation failed and returned the following error message:\n{e}")
        return None

    t_total = time.perf_counter() - t_start
    
    # run_time is already measured after transpilation; do not subtract transpile_time again.
    simulate_time = float(run_time)

    result = {
        'time': t_total,
        'build_time': build_time,
        'run_time': run_time,
        'simulate_time': simulate_time,
        'transpile_time': float(transpile_time),
    }
    
    if return_plots:
        probs_dv_final = statevector.probabilities(list(range(n_dv_qubits)))
        rho_cv_final = trace_out_qubits(circuit, statevector)
        wigner_cv_final = wigner(state=rho_cv_final, axes_min=-5, axes_max=5, axes_steps=101)
        result['plots'] = {
            'probs_dv_initial': probs_dv_initial,
            'wigner_cv_initial': wigner_cv_initial,
            'probs_dv_final': probs_dv_final,
            'wigner_cv_final': wigner_cv_final,
            'n_dv_qubits': n_dv_qubits
        }
    
    return result


def benchmark_bosonic_transfer(n_dv_qubits=4, cv_cutoff=64, n_runs=10, warmup=2):
    """
    Benchmark CV-to-DV state transfer using bosonic-qiskit.
    
    Args:
        n_dv_qubits: Number of DV qubits for encoding
        cv_cutoff: Fock space cutoff for CV mode (Hilbert space size)
        n_runs: Number of timing runs
        warmup: Number of warmup runs to discard
    
    Returns:
        dict: Timing results with mean, std, min, max
    
    Note:
        Requires: pip install bosonic-qiskit qiskit qiskit-aer
    """
    lam = 0.29  # Interaction parameter
    
    # Convert cutoff dimension to number of qubits
    cv_num_qubits = int(np.ceil(np.log2(cv_cutoff)))
    actual_cutoff = 2**cv_num_qubits
    
    timings = []
    build_timings = []
    run_timings = []
    simulate_timings = []
    transpile_timings = []
    
    # Warmup runs
    for _ in range(warmup):
        result = run_bosonic_transfer_experiment(n_dv_qubits, cv_cutoff, lam, return_plots=False)
        if result is None:
            print(f"[Bosonic] Warmup failed for cv_cutoff={cv_cutoff}, skipping this size.")
            return None

    # Timed runs
    for i in range(n_runs):
        result = run_bosonic_transfer_experiment(n_dv_qubits, cv_cutoff, lam, return_plots=False)
        if result is None:
            print(f"[Bosonic] Run {i+1}/{n_runs} failed for cv_cutoff={cv_cutoff}, skipping remaining runs.")
            return None
        timings.append(result['time'])
        build_timings.append(result.get('build_time', 0.0))
        run_timings.append(result.get('run_time', result['time']))
        simulate_timings.append(result.get('simulate_time', result.get('run_time', result['time'])))
        transpile_timings.append(result.get('transpile_time', 0.0))
    
    # Compute statistics
    results = {
        'mean': np.mean(timings),
        'std': np.std(timings),
        'min': np.min(timings),
        'max': np.max(timings),
        'all': timings,
        'build_mean': np.mean(build_timings),
        'build_std': np.std(build_timings),
        'build_min': np.min(build_timings),
        'build_max': np.max(build_timings),
        'run_mean': np.mean(run_timings),
        'run_std': np.std(run_timings),
        'run_min': np.min(run_timings),
        'run_max': np.max(run_timings),
        'simulate_mean': np.mean(simulate_timings),
        'simulate_std': np.std(simulate_timings),
        'simulate_min': np.min(simulate_timings),
        'simulate_max': np.max(simulate_timings),
        'transpile_mean': np.mean(transpile_timings),
        'transpile_std': np.std(transpile_timings),
        'transpile_min': np.min(transpile_timings),
        'transpile_max': np.max(transpile_timings),
        'transpile_all': transpile_timings,
        'config': {
            'n_dv_qubits': n_dv_qubits,
            'cv_cutoff': actual_cutoff,
            'cv_num_qubits': cv_num_qubits,
            'n_runs': n_runs,
            'warmup': warmup
        }
    }
    
    return results


def print_results(results):
    """Print benchmark results in a readable format."""
    if results is None:
        return

    config = results['config']
    print(f"\nBosonic-Qiskit Configuration:")
    print(f"  DV qubits: {config['n_dv_qubits']}")
    print(f"  CV cutoff: {config['cv_cutoff']} (2^{config['cv_num_qubits']} Fock states)")
    print(f"  Runs: {config['n_runs']} (+ {config['warmup']} warmup)")
    print("-" * 60)
    print(f"Total Time: {results['mean']*1000:7.2f} ± {results['std']*1000:5.2f} ms  "
          f"[{results['min']*1000:6.2f}, {results['max']*1000:6.2f}]")
    print(
        f"Build:      {results['build_mean']*1000:7.2f} ± {results['build_std']*1000:5.2f} ms  "
        f"[{results['build_min']*1000:6.2f}, {results['build_max']*1000:6.2f}]"
    )
    print(
        f"Simulate:   {results['simulate_mean']*1000:7.2f} ± {results['simulate_std']*1000:5.2f} ms  "
        f"[{results['simulate_min']*1000:6.2f}, {results['simulate_max']*1000:6.2f}]"
    )
    print(
        f"Transpile:  {results['transpile_mean']*1000:7.2f} ± {results['transpile_std']*1000:5.2f} ms  "
        f"[{results['transpile_min']*1000:6.2f}, {results['transpile_max']*1000:6.2f}]"
    )
    print()


def visualize_bosonic_transfer(n_dv_qubits=4, cv_cutoff=64):
    """
    Visualize CV-to-DV state transfer with initial and final state plots.
    
    Args:
        n_dv_qubits: Number of DV qubits for encoding
        cv_cutoff: Fock space cutoff for CV mode
    
    Returns:
        tuple: (fig_initial, fig_final) matplotlib figure objects
    """
    lam = 0.29
    
    # Run experiment once with plot data
    result = run_bosonic_transfer_experiment(n_dv_qubits, cv_cutoff, lam, return_plots=True)
    if result is None:
        return None, None
    
    plots = result['plots']
    probs_dv_initial = plots['probs_dv_initial']
    wigner_cv_initial = plots['wigner_cv_initial']
    probs_dv_final = plots['probs_dv_final']
    wigner_cv_final = plots['wigner_cv_final']
    
    # Plot initial state
    fig_initial, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    indices_init = np.arange(len(probs_dv_initial))
    bit_rev_idx_init = np.array([sum((i >> b & 1) << (n_dv_qubits - 1 - b) for b in range(n_dv_qubits)) for i in indices_init])
    probs_dv_initial_reordered = probs_dv_initial[bit_rev_idx_init]
    
    axes[0].bar(range(len(probs_dv_initial_reordered)), probs_dv_initial_reordered, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Basis state')
    axes[0].set_ylabel('Probability')
    axes[0].set_title(f'Initial DV: |+⟩^⊗{n_dv_qubits}')
    axes[0].set_xlim(-0.5, len(probs_dv_initial) - 0.5)
    axes[0].grid(True, alpha=0.3)
    
    xvec = np.linspace(-5, 5, 101)
    vmax = np.max(np.abs(wigner_cv_initial))
    color_levels = np.linspace(-vmax, vmax, 100)
    cont = axes[1].contourf(xvec, xvec, wigner_cv_initial, color_levels, cmap='RdBu')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('p')
    axes[1].set_title('Initial CV')
    axes[1].set_aspect('equal', 'box')
    cb = plt.colorbar(cont, ax=axes[1])
    cb.set_label('W(x,p)', rotation=270, labelpad=15)
    plt.tight_layout()
    
    # Plot final state
    fig_final, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    indices = np.arange(len(probs_dv_final))
    bit_rev_idx = np.array([sum((i >> b & 1) << (n_dv_qubits - 1 - b) for b in range(n_dv_qubits)) for i in indices])
    probs_dv_reordered = probs_dv_final[bit_rev_idx]
    rev_indices = sum((indices >> i & 1) << (n_dv_qubits - 1 - i) for i in range(n_dv_qubits))
    shifted_idx = np.roll(rev_indices, len(probs_dv_reordered) // 2)[::-1]
    
    axes[0].bar(range(len(probs_dv_reordered)), probs_dv_reordered[shifted_idx], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Basis state')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Final DV')
    axes[0].set_xlim(-0.5, len(probs_dv_final) - 0.5)
    axes[0].grid(True, alpha=0.3)
    
    vmax = np.max(np.abs(wigner_cv_final))
    color_levels = np.linspace(-vmax, vmax, 100)
    cont = axes[1].contourf(xvec, xvec, wigner_cv_final, color_levels, cmap='RdBu')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('p')
    axes[1].set_title('Final CV')
    axes[1].set_aspect('equal', 'box')
    cb = plt.colorbar(cont, ax=axes[1])
    cb.set_label('W(x,p)', rotation=270, labelpad=15)
    plt.tight_layout()
    
    return fig_initial, fig_final


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Benchmark bosonic-qiskit state transfer')
    parser.add_argument('--dv-qubits', type=int, default=4, help='Number of DV qubits')
    parser.add_argument('--cv-cutoff', type=int, default=64, help='Fock space cutoff')
    parser.add_argument('--runs', type=int, default=10, help='Number of timing runs')
    parser.add_argument('--warmup', type=int, default=2, help='Number of warmup runs')
    args = parser.parse_args()

    print("="*70)
    print("Bosonic-Qiskit CV-to-DV State Transfer Benchmark")
    print("="*70)

    results = benchmark_bosonic_transfer(
        n_dv_qubits=args.dv_qubits,
        cv_cutoff=args.cv_cutoff,
        n_runs=args.runs,
        warmup=args.warmup
    )

    if results:
        print_results(results)

