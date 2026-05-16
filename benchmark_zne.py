import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit import qasm2
from qiskit_aer import AerSimulator
# Import the exact hardware-building blocks from your dataset generator
from generate_quantum_dataset import make_fake_backend, build_noise_model, noisy_expectation_value

def global_fold_circuit(qc, scale_factor):
    """
    Performs digital global circuit folding.
    Supports odd integer scale factors (1, 3, 5, ...).
    """
    if scale_factor == 1:
        return qc
    
    # Calculate how many fold iterations (C^\dagger C) are needed
    num_folds = int((scale_factor - 1) / 2)
    folded_qc = qc.copy()
    
    # Store the core instructions (excluding final measurements if present)
    core_qc = qc.remove_final_measurements(inplace=False)
    inverse_qc = core_qc.inverse()
    
    for _ in range(num_folds):
        folded_qc.compose(inverse_qc, inplace=True)
        folded_qc.compose(core_qc, inplace=True)
        
    return folded_qc

def extrapolate_polynomial(lambdas, values, degree=2):
    """Fits data points to a polynomial curve and extrapolates to lambda=0."""
    poly_coefficients = np.polyfit(lambdas, values, degree)
    return np.polyval(poly_coefficients, 0.0)

def run_zne_benchmark(dataset_path, global_seed=999):
    print("=" * 60)
    print("        Executing Zero-Noise Extrapolation (ZNE) Benchmark")
    print("=" * 60)
    
    # 1. Rebuild the exact noisy environment using the dataset parameters
    n_qubits = 5 
    rng = np.random.default_rng(global_seed)
    backend_seed = int(rng.integers(0, 1000))
    backend = make_fake_backend(n_qubits, seed=backend_seed)
    noise_model = build_noise_model(backend)
    
    # 2. Load the target test dataset
    with open(dataset_path, "r") as f:
        samples = json.load(f)
        
    ideal_list = []
    noisy_list = []
    zne_list = []
    
    lambdas = [1.0, 3.0, 5.0]
    
    print(f"Processing {len(samples)} circuits using ZNE (lambdas={lambdas})...")
    
    for idx, sample in enumerate(samples):
        qasm_content = sample["circuit"]

        # Add necessary headers for the parser
        if 'OPENQASM 2.0;' not in qasm_content:
            qasm_content = 'OPENQASM 2.0;\n' + qasm_content

        if 'include "qelib1.inc";' not in qasm_content:
            lines = qasm_content.split('\n')
            lines.insert(1, 'include "qelib1.inc";')
            qasm_content = '\n'.join(lines)

        # Read the raw QASM architecture
        try:
            qc = qasm2.loads(qasm_content)
        except Exception as e:
            print(f"Error loading circuit {idx}: {e}")
            continue
        
        obs_list = sample["observable"]
        
        # Extract base target values
        ideal_val = sample["ideal_exp_value"][0]
        baseline_noisy_val = sample["noisy_exp_values"][0][0] # first trial, first obs
        
        # Collect expectations across different scaled noise limits
        exp_at_lambdas = []
        for l in lambdas:
            folded_qc = global_fold_circuit(qc, scale_factor=l)
            # Simulate using the exact measurement methods built into your dataset code
            res = noisy_expectation_value(folded_qc, obs_list, noise_model, backend, shots=4096, n_trials=1)
            exp_at_lambdas.append(res[0][0])
            
        # Extrapolate back to the zero-noise point limit
        zne_mitigated_val = extrapolate_polynomial(lambdas, exp_at_lambdas, degree=2)
        
        ideal_list.append(ideal_val)
        noisy_list.append(baseline_noisy_val)
        zne_list.append(zne_mitigated_val)
        
    # 3. Compute final performance metrics
    ideal_arr = np.array(ideal_list)
    noisy_arr = np.array(noisy_list)
    zne_arr = np.array(zne_list)
    
    mse_noisy = np.mean((noisy_arr - ideal_arr) ** 2)
    mse_zne = np.mean((zne_arr - ideal_arr) ** 2)
    mae_zne = np.mean(np.abs(zne_arr - ideal_arr))
    
    r2_zne = 1.0 - (np.sum((zne_arr - ideal_arr) ** 2) / (np.sum((ideal_arr - ideal_arr.mean()) ** 2) + 1e-12))
    mitigation_ratio_zne = (1.0 - (mse_zne / (mse_noisy + 1e-12))) * 100
    
    print("\n" + "=" * 60)
    print("  ZNE Evaluation Results Summary")
    print("=" * 60)
    print(f"  Base Noisy MSE    : {mse_noisy:.6f}")
    print(f"  ZNE Mitigated MSE : {mse_zne:.6f}")
    print(f"  ZNE Mitigated MAE : {mae_zne:.6f}")
    print(f"  ZNE R² Score      : {r2_zne:.4f}")
    print(f"  Mitigation Ratio  : {mitigation_ratio_zne:.2f}%")
    print("=" * 60)
    
    # 4. Generate comparison verification plot
    plt.figure(figsize=(7, 7))
    lims = [min(ideal_list)-0.1, max(ideal_list)+0.1]
    plt.plot(lims, lims, 'k--', alpha=0.5, label="Perfect Ideal Line")
    plt.scatter(ideal_list, noisy_list, color='red', alpha=0.5, label='Raw Noisy Data', s=15)
    plt.scatter(ideal_list, zne_list, color='orange', alpha=0.6, label='ZNE Mitigated Data', s=15)
    plt.xlabel("Ideal Expectation Values")
    plt.ylabel("Mitigated/Measured Values")
    plt.title("ZNE Performance Validation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Point directly to your hard test data file
    run_zne_benchmark("hard_test.json", global_seed=999)