import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit import qasm2
from qiskit_aer import AerSimulator
# Ensure these imports match your local files
from generate_quantum_dataset import make_fake_backend, build_noise_model, noisy_expectation_value

# The "Magic Header" that defines gates explicitly so you don't need qelib1.inc
SELF_CONTAINED_HEADER = """OPENQASM 2.0;
gate rz(phi) q { U(0, 0, phi) q; }
gate sx q { U(pi/2, -pi/2, pi/2) q; }
gate x q { U(pi, 0, pi) q; }
gate cx c, t { CX c, t; }
"""

def global_fold_circuit(qc, scale_factor):
    if scale_factor == 1: return qc
    num_folds = int((scale_factor - 1) / 2)
    folded_qc = qc.copy()
    core_qc = qc.remove_final_measurements(inplace=False)
    inverse_qc = core_qc.inverse()
    for _ in range(num_folds):
        folded_qc.compose(inverse_qc, inplace=True)
        folded_qc.compose(core_qc, inplace=True)
    return folded_qc

def extrapolate_polynomial(lambdas, values, degree=1):
    poly_coefficients = np.polyfit(lambdas, values, degree)
    return np.polyval(poly_coefficients, 0.0)

def run_zne_benchmark(dataset_path, global_seed=999):
    print("=" * 60)
    print("        Executing Zero-Noise Extrapolation (ZNE) Benchmark")
    print("=" * 60)
    
    n_qubits = 5 
    rng = np.random.default_rng(global_seed)
    backend_seed = int(rng.integers(0, 1000))
    backend = make_fake_backend(n_qubits, seed=backend_seed)
    noise_model = build_noise_model(backend)
    
    with open(dataset_path, "r") as f:
        samples = json.load(f)
        
    ideal_list, noisy_list, zne_list = [], [], []
    lambdas = [1.0, 3.0, 5.0]
    
    print(f"Processing {len(samples)} circuits using ZNE...")
    
    for idx, sample in enumerate(samples):
        # --- ROBUST QASM LOADING ---
        raw_qasm = sample["circuit"]
        
        # Clean the string: remove original OPENQASM/include lines to avoid duplicates
        body_lines = []
        for line in raw_qasm.split('\n'):
            if "OPENQASM" in line or "include" in line or line.strip() == "":
                continue
            body_lines.append(line)
        
        # Combine the self-contained header with the circuit body
        fixed_qasm = SELF_CONTAINED_HEADER + "\n".join(body_lines)
        
        try:
            qc = qasm2.loads(fixed_qasm)
        except Exception as e:
            print(f"  [!] Skipping circuit {idx} due to load error: {e}")
            continue

        # --- ZNE EXECUTION ---
        obs_list = sample["observable"]
        ideal_val = sample["ideal_exp_value"][0]
        baseline_noisy_val = sample["noisy_exp_values"][0][0]
        
        exp_at_lambdas = []
        for l in lambdas:
            folded_qc = global_fold_circuit(qc, scale_factor=l)
            # shots=4096 to match your GNN training conditions
            res = noisy_expectation_value(folded_qc, obs_list, noise_model, backend, shots=4096, n_trials=1)
            exp_at_lambdas.append(res[0][0])
            
        zne_mitigated_val = extrapolate_polynomial(lambdas, exp_at_lambdas, degree=1)
        
        ideal_list.append(ideal_val)
        noisy_list.append(baseline_noisy_val)
        zne_list.append(zne_mitigated_val)
        
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(samples)} circuits completed")

    # --- METRICS CALCULATION ---
    if not ideal_list:
        print("Error: No circuits were processed successfully. Check your QASM format.")
        return

    ideal_arr, noisy_arr, zne_arr = np.array(ideal_list), np.array(noisy_list), np.array(zne_list)
    
    mse_noisy = np.mean((noisy_arr - ideal_arr) ** 2)
    mse_zne = np.mean((zne_arr - ideal_arr) ** 2)
    mae_zne = np.mean(np.abs(zne_arr - ideal_arr))
    r2_zne = 1.0 - (np.sum((zne_arr - ideal_arr) ** 2) / (np.sum((ideal_arr - ideal_arr.mean()) ** 2) + 1e-12))
    mit_ratio = (1.0 - (mse_zne / (mse_noisy + 1e-12))) * 100
    
    print("\n" + "=" * 60)
    print("  ZNE Evaluation Results Summary")
    print("=" * 60)
    print(f"  Base Noisy MSE    : {mse_noisy:.6f}")
    print(f"  ZNE Mitigated MSE : {mse_zne:.6f}")
    print(f"  ZNE Mitigated MAE : {mae_zne:.6f}")
    print(f"  ZNE R² Score      : {r2_zne:.4f}")
    print(f"  Mitigation Ratio  : {mit_ratio:.2f}%")
    print("=" * 60)
    
    # Parity Plot
    plt.figure(figsize=(7, 7))
    plt.plot([-1, 1], [-1, 1], 'k--', alpha=0.5, label="Ideal")
    plt.scatter(ideal_list, noisy_list, color='red', alpha=0.4, label='Raw Noisy', s=20)
    plt.scatter(ideal_list, zne_list, color='orange', alpha=0.7, label='ZNE Mitigated', s=20)
    plt.xlabel("Ideal Value"), plt.ylabel("Measured Value"), plt.title("ZNE Benchmark")
    plt.legend(), plt.grid(True, alpha=0.3), plt.show()

if __name__ == "__main__":
    run_zne_benchmark("hard_test.json", global_seed=999)