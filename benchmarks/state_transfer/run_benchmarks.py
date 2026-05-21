
"""
Benchmark Runner - Compare CVDV vs Bosonic-Qiskit
"""

import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from bench_cvdv import benchmark_cvdv_transfer, print_results as print_cvdv, visualize_cvdv_transfer


def run_comparison(
    dv_qubits=4,
    cvdv_cv_qubits=[10, 11, 12],
    bosonic_cv_qubits=None,
    qcvdv_cv_qubits=None,
    qcvdv_methods=None,
    qcvdv_shots=1,
    qcvdv_clear_cache_each_run=False,
    n_runs=10,
    warmup=2,
):
    """
    Run benchmarks for both CVDV and bosonic-qiskit across multiple configurations.
    
    Args:
        dv_qubits: Number of DV qubits
        cvdv_cv_qubits: List of CV register sizes (in qubits) for CVDV to test
        bosonic_cv_qubits: List of CV register sizes (in qubits) for Bosonic to test (defaults to cvdv_cv_qubits)
        qcvdv_cv_qubits: List of CV register sizes (in qubits) for qcvdv to test (defaults to cvdv_cv_qubits)
        qcvdv_methods: List of qcvdv subbackends (e.g., dense_matrix, dense, scipy)
        qcvdv_shots: Shots used for qcvdv simulator
        qcvdv_clear_cache_each_run: Clear converted-circuit gate cache before each qcvdv run
        n_runs: Number of timing runs
        warmup: Number of warmup runs
    """
    if bosonic_cv_qubits is None:
        bosonic_cv_qubits = cvdv_cv_qubits
    if qcvdv_cv_qubits is None:
        qcvdv_cv_qubits = cvdv_cv_qubits
    if qcvdv_methods is None:
        qcvdv_methods = ["dense_matrix_gpu", "dense_matrix_gpuv1", "torch"]
    
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON: CUDA-CVDV vs Bosonic-Qiskit vs qcvdv")
    print("="*70)
    
    # Try to import bosonic-qiskit
    try:
        from bench_bosonic import benchmark_bosonic_transfer, print_results as print_bosonic
        has_bosonic = True
    except ImportError:
        print("\nWarning: bosonic-qiskit not available, running CVDV only")
        has_bosonic = False

    # Try to import qcvdv benchmark module
    try:
        from bench_qcvdv import benchmark_qcvdv_transfer, print_results as print_qcvdv
        has_qcvdv = True
    except ImportError:
        print("\nWarning: qcvdv not available, skipping qcvdv benchmarks")
        has_qcvdv = False
    
    # Use the union of all configs for plotting
    all_configs = sorted(set(cvdv_cv_qubits + bosonic_cv_qubits + qcvdv_cv_qubits))
    
    # Store results for all configurations
    cvdv_times = {}
    cvdv_results_all = {}
    bosonic_times = {}
    bosonic_results_all = {}
    qcvdv_times = {}
    qcvdv_results_all = {}
    
    # Initialize qcvdv containers up front.
    if has_qcvdv:
        for method in qcvdv_methods:
            qcvdv_times[method] = {}
            qcvdv_results_all[method] = {}

    # Run benchmarks grouped by CV size so partial runs still keep complete cross-backend comparisons
    # for the smallest dimensions when jobs hit wall-clock limits.
    for cv_qubits in all_configs:
        print(f"\n{'#'*70}")
        print(f"[SIZE GROUP] DV={dv_qubits} qubits, CV={cv_qubits} qubits (dim={2**cv_qubits})")
        print(f"{'#'*70}")

        if cv_qubits in cvdv_cv_qubits:
            print(f"\n{'='*70}")
            print(f"[CVDV] Configuration: DV={dv_qubits} qubits, CV={cv_qubits} qubits (dim={2**cv_qubits})")
            print(f"{'='*70}")
            try:
                cvdv_results = benchmark_cvdv_transfer(dv_qubits, cv_qubits, n_runs, warmup)
                print_cvdv(cvdv_results)
                cvdv_times[cv_qubits] = cvdv_results['mean'] * 1000  # Convert to ms
                cvdv_results_all[cv_qubits] = cvdv_results
            except Exception as e:
                print(f"[CVDV] Skipping cv_qubits={cv_qubits}: {e}")
                cvdv_times[cv_qubits] = None
                cvdv_results_all[cv_qubits] = None

        if has_bosonic and cv_qubits in bosonic_cv_qubits:
            print(f"\n{'='*70}")
            print(f"[Bosonic-Qiskit] Configuration: DV={dv_qubits} qubits, CV={cv_qubits} qubits (dim={2**cv_qubits})")
            print(f"{'='*70}")
            try:
                bosonic_results = benchmark_bosonic_transfer(dv_qubits, 2**cv_qubits, n_runs, warmup)
                if bosonic_results is not None:
                    print_bosonic(bosonic_results)
                    bosonic_times[cv_qubits] = bosonic_results['mean'] * 1000  # Convert to ms
                    bosonic_results_all[cv_qubits] = bosonic_results
                else:
                    bosonic_times[cv_qubits] = None
                    bosonic_results_all[cv_qubits] = None
            except Exception as e:
                print(f"[Bosonic] Skipping cv_qubits={cv_qubits}: {e}")
                bosonic_times[cv_qubits] = None
                bosonic_results_all[cv_qubits] = None

        if has_qcvdv and cv_qubits in qcvdv_cv_qubits:
            for method in qcvdv_methods:
                print(f"\n{'='*70}")
                print(f"[qcvdv:{method}] Configuration: DV={dv_qubits} qubits, CV={cv_qubits} qubits (dim={2**cv_qubits})")
                print(f"{'='*70}")
                try:
                    qcvdv_results = benchmark_qcvdv_transfer(
                        n_dv_qubits=dv_qubits,
                        cv_qubits=cv_qubits,
                        n_runs=n_runs,
                        warmup=warmup,
                        method=method,
                        shots=qcvdv_shots,
                        clear_cache_each_run=qcvdv_clear_cache_each_run,
                    )
                    print_qcvdv(qcvdv_results)
                    qcvdv_times[method][cv_qubits] = qcvdv_results['mean'] * 1000  # Convert to ms
                    qcvdv_results_all[method][cv_qubits] = qcvdv_results
                except Exception as e:
                    print(f"qcvdv benchmark failed for backend={method}, cv_qubits={cv_qubits}: {e}")
                    qcvdv_times[method][cv_qubits] = None
                    qcvdv_results_all[method][cv_qubits] = None
    
    # Save JSON results
    save_json_results(
        dv_qubits,
        cvdv_results_all,
        bosonic_results_all if has_bosonic else None,
        qcvdv_results_all if has_qcvdv else None,
        n_runs,
        warmup,
    )
    
    # Generate comparison plot
    plot_comparison(
        all_configs,
        cvdv_times,
        bosonic_times if has_bosonic else None,
        qcvdv_times if has_qcvdv else None,
    )

    # Generate stacked component plot (transpile+simulate for bosonic, convert+run for qcvdv)
    plot_component_stacks(
        cvdv_results_all,
        bosonic_results_all if has_bosonic else None,
        qcvdv_results_all if has_qcvdv else None,
    )
    
    # Generate state visualizations for a smaller CVDV config to keep memory bounded.
    if cvdv_cv_qubits:
        last_cvdv_config = min(cvdv_cv_qubits)
        print(f"\n{'='*70}")
        print(f"Generating CVDV state visualization for CV={last_cvdv_config} qubits...")
        print(f"{'='*70}")
        try:
            fig_initial, fig_final = visualize_cvdv_transfer(dv_qubits, last_cvdv_config)
            output_dir = os.path.join(os.path.dirname(__file__), 'results')
            os.makedirs(output_dir, exist_ok=True)
            fig_initial.savefig(os.path.join(output_dir, 'cvdv_initial_state.png'), dpi=300, bbox_inches='tight')
            fig_final.savefig(os.path.join(output_dir, 'cvdv_final_state.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_initial)
            plt.close(fig_final)
            print(f"✓ Saved: cvdv_initial_state.png")
            print(f"✓ Saved: cvdv_final_state.png")
        except Exception as e:
            print(f"Warning: CVDV visualization failed: {e}")
    
    if has_bosonic and bosonic_cv_qubits:
        last_bosonic_config = bosonic_cv_qubits[-1]
        print(f"\nGenerating Bosonic-Qiskit state visualization for CV={last_bosonic_config} qubits...")
        try:
            from bench_bosonic import visualize_bosonic_transfer
            fig_initial, fig_final = visualize_bosonic_transfer(dv_qubits, 2**last_bosonic_config)
            if fig_initial is not None and fig_final is not None:
                output_dir = os.path.join(os.path.dirname(__file__), 'results')
                fig_initial.savefig(os.path.join(output_dir, 'bosonic_initial_state.png'), dpi=300, bbox_inches='tight')
                fig_final.savefig(os.path.join(output_dir, 'bosonic_final_state.png'), dpi=300, bbox_inches='tight')
                plt.close(fig_initial)
                plt.close(fig_final)
                print(f"✓ Saved: bosonic_initial_state.png")
                print(f"✓ Saved: bosonic_final_state.png")
        except Exception as e:
            print(f"Warning: Bosonic visualization failed: {e}")


def save_json_results(dv_qubits, cvdv_results, bosonic_results=None, qcvdv_results=None, n_runs=10, warmup=2):
    """Save benchmark results to JSON file."""
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'dv_qubits': dv_qubits,
            'n_runs': n_runs,
            'warmup': warmup
        },
        'cvdv': {}
    }
    
    # CVDV results (convert numpy types to native Python)
    for cv_qubits, res in cvdv_results.items():
        results['cvdv'][str(cv_qubits)] = {
            'cv_dimension': 2**cv_qubits,
            'mean_ms': float(res['mean'] * 1000),
            'std_ms': float(res['std'] * 1000),
            'min_ms': float(res['min'] * 1000),
            'max_ms': float(res['max'] * 1000),
            'build_mean_ms': float(res.get('build_mean', 0.0) * 1000),
            'run_mean_ms': float(res.get('run_mean', 0.0) * 1000),
        }
    
    # Bosonic results
    if bosonic_results:
        results['bosonic'] = {}
        for cv_qubits, res in bosonic_results.items():
            if res is not None:
                results['bosonic'][str(cv_qubits)] = {
                    'cv_dimension': 2**cv_qubits,
                    'mean_ms': float(res['mean'] * 1000),
                    'std_ms': float(res['std'] * 1000),
                    'min_ms': float(res['min'] * 1000),
                    'max_ms': float(res['max'] * 1000),
                    'build_mean_ms': float(res.get('build_mean', 0.0) * 1000),
                    'simulate_mean_ms': float(res.get('simulate_mean', res.get('run_mean', 0.0)) * 1000),
                    'run_mean_ms': float(res.get('run_mean', 0.0) * 1000),
                    'transpile_mean_ms': float(res.get('transpile_mean', 0.0) * 1000),
                }

    # qcvdv results by backend
    if qcvdv_results:
        results['qcvdv'] = {}
        for method, method_results in qcvdv_results.items():
            results['qcvdv'][method] = {}
            for cv_qubits, res in method_results.items():
                if res is None:
                    continue
                results['qcvdv'][method][str(cv_qubits)] = {
                    'cv_dimension': 2**cv_qubits,
                    'mean_ms': float(res['mean'] * 1000),
                    'with_convert_mean_ms': float(res.get('with_convert_mean', res['mean']) * 1000),
                    'std_ms': float(res['std'] * 1000),
                    'min_ms': float(res['min'] * 1000),
                    'max_ms': float(res['max'] * 1000),
                    'build_mean_ms': float(res.get('build_mean', 0.0) * 1000),
                    'convert_mean_ms': float(res.get('convert_mean', 0.0) * 1000),
                    'run_mean_ms': float(res.get('run_mean', 0.0) * 1000),
                    'run_matrix_generation_mean_ms': float(res.get('run_matrix_generation_mean', 0.0) * 1000),
                    'run_apply_mean_ms': float(res.get('run_apply_mean', 0.0) * 1000),
                    'run_build_mean_ms': float(res.get('run_build_mean', 0.0) * 1000),
                    'run_transfer_mean_ms': float(res.get('run_transfer_mean', 0.0) * 1000),
                    'run_cache_hit_mean_ms': float(res.get('run_cache_hit_mean', 0.0) * 1000),
                    'run_other_mean_ms': float(res.get('run_other_mean', 0.0) * 1000),
                    'run_total_mean_ms': float(res.get('run_total_mean', 0.0) * 1000),
                    'transpile_mean_ms': float(res.get('transpile_mean', 0.0) * 1000),
                }
    
    json_path = os.path.join(output_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved: benchmark_results.json")


def plot_comparison(cv_qubit_configs, cvdv_times_dict, bosonic_times_dict=None, qcvdv_times_dict=None):
    """Generate comparison bar chart with modern styling."""
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX for simplicity
    
    # Modern color palette
    colors = {
        'cvdv': '#2E86AB',
        'bosonic': '#A23B72'
    }
    qcvdv_palette = ['#F18F01', '#1B998B', '#C73E1D', '#6A4C93', '#3A86FF']
    
    width = 0.5
    
    # Create figure with modern style
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F8F9FA')
    
    # Arrange bars: bosonic (low to high), then cvdv (low to high)
    bar_configs = []
    bar_values = []
    bar_colors = []
    bar_legend_labels = []
    
    # Add bosonic bars (sorted by dimension low to high)
    if bosonic_times_dict:
        for cv_q in sorted(bosonic_times_dict.keys()):
            if bosonic_times_dict[cv_q] is not None:
                bar_configs.append(('Bosonic', 2**cv_q))
                bar_values.append(bosonic_times_dict[cv_q])
                bar_colors.append(colors['bosonic'])
                bar_legend_labels.append('Bosonic-Qiskit (CPU)')

    # Add qcvdv bars (grouped by backend then by dimension)
    if qcvdv_times_dict:
        for method_idx, method in enumerate(sorted(qcvdv_times_dict.keys())):
            method_color = qcvdv_palette[method_idx % len(qcvdv_palette)]
            for cv_q in sorted(qcvdv_times_dict[method].keys()):
                if qcvdv_times_dict[method][cv_q] is not None:
                    bar_configs.append((f'qcvdv:{method}', 2**cv_q))
                    bar_values.append(qcvdv_times_dict[method][cv_q])
                    bar_colors.append(method_color)
                    bar_legend_labels.append(f'qcvdv:{method}')
    
    # Add cvdv bars (sorted by dimension low to high)
    for cv_q in sorted(cvdv_times_dict.keys()):
        if cvdv_times_dict[cv_q] is not None:
            bar_configs.append(('CVDV', 2**cv_q))
            bar_values.append(cvdv_times_dict[cv_q])
            bar_colors.append(colors['cvdv'])
            bar_legend_labels.append('CUDA-CVDV (GPU)')
    
    # Plot bars centered at their positions
    x_positions = np.arange(len(bar_configs))
    bars = ax.bar(x_positions, bar_values, width, color=bar_colors, alpha=0.85, 
                  edgecolor='white', linewidth=1.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Create legend manually
    from matplotlib.patches import Patch
    legend_elements = []
    legend_map = {}
    for label, color in zip(bar_legend_labels, bar_colors):
        if label not in legend_map:
            legend_map[label] = color
    for label, color in legend_map.items():
        legend_elements.append(Patch(facecolor=color, alpha=0.85,
                                     edgecolor='white', linewidth=1.5, label=label))
    
    # Styling
    ax.set_ylabel('Total Runtime (ms, log scale)', fontsize=12, fontweight='bold', color='#333')
    ax.set_xlabel('CV Mode Dimension', fontsize=12, fontweight='bold', color='#333')
    ax.set_title('Performance Comparison (Log Scale)',
                 fontsize=15, fontweight='bold', pad=20, color='#222')
    ax.set_yscale('log')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{lib}\n{dim}' for lib, dim in bar_configs], fontsize=10)
    ax.legend(handles=legend_elements, fontsize=12, frameon=True, shadow=True, fancybox=True, loc='upper left')
    
    # Modern grid style
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    plt.tight_layout()
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, 'comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n{'='*70}")
    print(f"Comparison plot saved to: {plot_file}")
    print(f"{'='*70}\n")
    plt.close(fig)  # Close figure to free memory


def plot_component_stacks(cvdv_results=None, bosonic_results=None, qcvdv_results=None):
    """Generate stacked bar plot for component timings.

    CVDV: build + run
    Bosonic: build + transpile + run
    qcvdv: build + convert + run subcomponents (when profiler data is available)
    """
    # Collect plotting rows as tuples:
    # (label, [component_values_ms], [component_names], [component_colors])
    rows = []

    # Colors tuned to remain close to existing style.
    cvdv_build_c = '#6FA8DC'
    cvdv_run_c = '#2E86AB'
    bosonic_build_c = '#E5B8D5'
    bosonic_transpile_c = '#D081B3'
    bosonic_run_c = '#A23B72'
    qcvdv_build_palette = ['#FFE8A3', '#FCDDBB', '#FFE3B3', '#F7E1A3', '#FCE5B3']
    qcvdv_convert_palette = ['#FFD166', '#F4A261', '#F6BD60', '#E9C46A', '#FAC05E']
    qcvdv_run_palette = ['#F18F01', '#1B998B', '#C73E1D', '#6A4C93', '#3A86FF']
    qcvdv_run_split_colors = {
        'matrix_generation': '#5B8E7D',
        'apply': '#7B6D8D',
        'build': '#E07A5F',
        'transfer': '#3D5A80',
        'cache_hit': '#98C1D9',
        'other': '#BFC0C0',
    }

    # Build rows in strict per-dimension sequence:
    # cuda-cvdv -> bosonic-gpu -> qcvdv:<method 1..N>
    cv_keys = set()
    if cvdv_results:
        cv_keys.update(cvdv_results.keys())
    if bosonic_results:
        cv_keys.update(bosonic_results.keys())
    if qcvdv_results:
        for method in qcvdv_results:
            cv_keys.update(qcvdv_results[method].keys())

    ordered_cv_keys = sorted(cv_keys)
    ordered_qcvdv_methods = sorted(qcvdv_results.keys()) if qcvdv_results else []

    group_end_indices = []

    for cv_q in ordered_cv_keys:
        if cvdv_results and cv_q in cvdv_results and cvdv_results[cv_q] is not None:
            res = cvdv_results[cv_q]
            build_ms = float(res.get('build_mean', 0.0) * 1000)
            run_ms = float(res.get('run_mean', 0.0) * 1000)
            rows.append(
                (
                    f'cuda-cvdv\\n2^{cv_q}',
                    [build_ms, run_ms],
                    ['build', 'run'],
                    [cvdv_build_c, cvdv_run_c],
                )
            )

        if bosonic_results and cv_q in bosonic_results and bosonic_results[cv_q] is not None:
            res = bosonic_results[cv_q]
            build_ms = float(res.get('build_mean', 0.0) * 1000)
            transpile_ms = float(res.get('transpile_mean', 0.0) * 1000)
            run_ms = float(res.get('run_mean', 0.0) * 1000)
            rows.append(
                (
                    f'bosonic-gpu\\n2^{cv_q}',
                    [build_ms, transpile_ms, run_ms],
                    ['build', 'transpile', 'run'],
                    [bosonic_build_c, bosonic_transpile_c, bosonic_run_c],
                )
            )

        for method_idx, method in enumerate(ordered_qcvdv_methods):
            method_rows = qcvdv_results.get(method, {}) if qcvdv_results else {}
            if cv_q not in method_rows or method_rows[cv_q] is None:
                continue
            res = method_rows[cv_q]
            build_c = qcvdv_build_palette[method_idx % len(qcvdv_build_palette)]
            convert_c = qcvdv_convert_palette[method_idx % len(qcvdv_convert_palette)]
            run_c = qcvdv_run_palette[method_idx % len(qcvdv_run_palette)]
            build_ms = float(res.get('build_mean', 0.0) * 1000)
            convert_ms = float(res.get('convert_mean', 0.0) * 1000)
            run_ms = float(res.get('run_mean', 0.0) * 1000)

            run_split_keys = ['matrix_generation', 'apply', 'build', 'transfer', 'cache_hit', 'other']
            run_split_values = [float(res.get(f'run_{k}_mean', 0.0) * 1000) for k in run_split_keys]
            has_run_split = any(v > 0 for v in run_split_values)
            if has_run_split:
                component_values = [build_ms, convert_ms] + run_split_values
                component_names = ['build', 'convert'] + [f'run:{k}' for k in run_split_keys]
                component_colors = [build_c, convert_c] + [qcvdv_run_split_colors[k] for k in run_split_keys]
            else:
                component_values = [build_ms, convert_ms, run_ms]
                component_names = ['build', 'convert', 'run']
                component_colors = [build_c, convert_c, run_c]

            rows.append(
                (
                    f'qcvdv:{method}\\n2^{cv_q}',
                    component_values,
                    component_names,
                    component_colors,
                )
            )

        if rows:
            group_end_indices.append(len(rows) - 1)

    if not rows:
        print("No component timing data available for stacked plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F8F9FA')

    x = np.arange(len(rows))
    width = 0.68

    for idx, (label, values, names, colors) in enumerate(rows):
        bottom = 0.0
        for comp_val, comp_name, comp_color in zip(values, names, colors):
            ax.bar(idx, comp_val, width=width, bottom=bottom, color=comp_color, alpha=0.9, edgecolor='white', linewidth=1.2)
            bottom += comp_val

        total = bottom
        ax.text(idx, total, f'{total:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Time (ms, log scale)', fontsize=12, fontweight='bold', color='#333')
    ax.set_xlabel('Backend / CV Dimension', fontsize=12, fontweight='bold', color='#333')
    ax.set_title('Component Timing Breakdown (Log Scale)', fontsize=15, fontweight='bold', pad=20, color='#222')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10)

    # Draw separators between CV-dimension groups.
    for gi in group_end_indices[:-1]:
        ax.axvline(gi + 0.5, color='#BBBBBB', linewidth=1.0, alpha=0.6, linestyle='--')

    # Build legend from unique (name, color) pairs while preserving order.
    from matplotlib.patches import Patch

    seen = set()
    legend_elements = []
    for _, values, names, colors in rows:
        for n, c in zip(names, colors):
            key = (n, c)
            if key in seen:
                continue
            seen.add(key)
            legend_elements.append(Patch(facecolor=c, edgecolor='white', linewidth=1.2, label=n))

    ax.legend(handles=legend_elements, fontsize=11, frameon=True, fancybox=True, loc='upper left')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')

    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, 'comparison_components_stacked.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Component breakdown plot saved to: {plot_file}")
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Run benchmarks and compare CVDV vs bosonic-qiskit vs qcvdv'
    )
    parser.add_argument('--dv-qubits', type=int, default=4,
                        help='Number of DV qubits (default: 4)')
    parser.add_argument('--cvdv-cv-qubits', type=int, nargs='+', default=[4, 5, 6, 7, 8, 9, 10, 11, 12],
                        help='CVDV: CV register qubits to test (default: 4 5 6 7 8 9 10 11 12)')
    parser.add_argument('--bosonic-cv-qubits', type=int, nargs='+', default=None,
                        help='Bosonic: CV register qubits to test (default: same as --cvdv-cv-qubits)')
    parser.add_argument('--qcvdv-cv-qubits', type=int, nargs='+', default=None,
                        help='qcvdv: CV register qubits to test (default: same as --cvdv-cv-qubits)')
    parser.add_argument('--qcvdv-methods', type=str, nargs='+', default=['dense_matrix_gpu', 'dense_matrix_gpuv1'],
                        help='qcvdv subbackends to test (default: dense_matrix_gpu dense_matrix_gpuv1)')
    parser.add_argument('--qcvdv-shots', type=int, default=1,
                        help='qcvdv shots (default: 1)')
    parser.add_argument('--qcvdv-clear-cache-each-run', action='store_true', default=False,
                        help='Clear qcvdv converted-circuit gate cache before each run')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of timing runs (default: 10)')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup runs (default: 2)')
    args = parser.parse_args()
    
    run_comparison(
        dv_qubits=args.dv_qubits,
        cvdv_cv_qubits=args.cvdv_cv_qubits,
        bosonic_cv_qubits=args.bosonic_cv_qubits,
        qcvdv_cv_qubits=args.qcvdv_cv_qubits,
        qcvdv_methods=args.qcvdv_methods,
        qcvdv_shots=args.qcvdv_shots,
        qcvdv_clear_cache_each_run=args.qcvdv_clear_cache_each_run,
        n_runs=args.runs,
        warmup=args.warmup
    )
