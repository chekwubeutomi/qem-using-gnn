import matplotlib.pyplot as plt

def plot_noise_dilution(save_plot=True):
    # Data from Table 5.5
    qubits = [4, 5, 6, 7, 8]
    noisy_mse = [0.005695, 0.003871, 0.002781, 0.002147, 0.001100]

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(qubits, noisy_mse, marker='o', linestyle='-', color='#d62728', 
             linewidth=2.5, markersize=10, label='Baseline Noisy MSE')

    # Formatting
    plt.xlabel('Number of Qubits ($Q$)', fontsize=13)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=13)
    plt.title('Baseline Noisy MSE vs. Qubit Count (Noise Dilution Effect)', fontsize=14, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(qubits)
    
    # Add trend annotation
    # plt.annotate('Noise Dilution Trend', 
    #              xy=(7, 0.002147), xytext=(5.5, 0.005),
    #              arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
    #              fontsize=11, fontweight='bold', color='black')

    plt.tight_layout()

    # Save function
    if save_plot:
        # Saving as PDF for high-quality LaTeX inclusion
        plt.savefig('noisy_mse_dilution.png', format='pdf', bbox_inches='tight')
        # Saving as PNG for general use
        plt.savefig('noisy_mse_dilution.png', dpi=300, bbox_inches='tight')
        print("Figures saved as 'noisy_mse_dilution.pdf' and 'noisy_mse_dilution.png'")

    plt.show()

# Run the function
plot_noise_dilution(save_plot=True)