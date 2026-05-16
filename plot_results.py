import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_training_history(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot MSE
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE (Lower is better)', color='tab:blue')
    ax1.plot(df['train_mse'], label='Train MSE', color='tab:blue', linestyle='--')
    ax1.plot(df['val_mse'], label='Val MSE', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot Mitigation Ratio on second axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mitigation Ratio (Higher is better)', color='tab:green')
    ax2.plot(df['val_mitigation_ratio'], label='Mitigation Ratio', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title("GNN Training Progress")
    fig.tight_layout()
    plt.show()

def plot_parity(ideal, noisy, mitigated):
    """
    The 'Parity Plot' is the most important plot in Error Mitigation.
    It shows how close the values are to the y=x (perfect) line.
    """
    plt.figure(figsize=(8, 8))
    
    # Perfect mitigation line
    lims = [min(ideal)-0.1, max(ideal)+0.1]
    plt.plot(lims, lims, 'k--', alpha=0.5, label="Ideal (Perfect)")
    
    plt.scatter(ideal, noisy, alpha=0.5, label="Noisy Data", color='red', s=10)
    plt.scatter(ideal, mitigated, alpha=0.5, label="GNN Mitigated", color='blue', s=10)
    
    plt.xlabel("Ideal Expectation Value")
    plt.ylabel("Measured/Predicted Value")
    plt.title("Parity Plot: Noisy vs. Mitigated")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Run this after training
if __name__ == "__main__":
    plot_training_history("best_model.history.json")