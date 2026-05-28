import matplotlib.pyplot as plt

def plot_mitigation_ratio(save_plot=True):
    # Data extracted from Table 5.3
    qubits = [4, 5, 6, 7, 8]
    
    # Mitigation Ratios (MR) in percentages
    gnn_mr = [82.48, 70.15, 64.39, 60.26, 52.93]
    mlp_mr = [81.17, 68.14, 63.12, 58.01, 49.04]

    # Initialize the plot
    plt.figure(figsize=(9, 6))
    
    # Plot GNN data
    plt.plot(qubits, gnn_mr, marker='o', markersize=9, linewidth=2.5, 
             label='GNN', color='#1f77b4', linestyle='-')
    
    # Plot MLP data
    plt.plot(qubits, mlp_mr, marker='s', markersize=9, linewidth=2.5, 
             label='MLP', color='#ff7f0e', linestyle='--')

    # Formatting axes and labels
    plt.xlabel('Number of Qubits ($Q$)', fontsize=13, fontweight='bold')
    plt.ylabel('Mitigation Ratio (%)', fontsize=13, fontweight='bold')
    plt.title('Mitigation Performance: GNN vs. MLP (Shallow Circuits)', fontsize=14, pad=15)
    
    # Set y-axis limits to show the differences clearly
    plt.ylim(40, 90)
    
    # Add grid and legend
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(qubits)
    plt.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)

    # Annotating the "Winner" trend (GNN consistently higher in MR)
    plt.text(5.5, 80, 'GNN maintains higher\nMitigation Ratios', 
             fontsize=10, color='#1f77b4', fontweight='bold', ha='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()

    # Save function
    if save_plot:
        plt.savefig('mitigation_ratio_comparison.pdf', format='pdf', bbox_inches='tight')
        plt.savefig('mitigation_ratio_comparison.png', dpi=300, bbox_inches='tight')
        print("Figures saved as 'mitigation_ratio_comparison.pdf' and 'png'")

    plt.show()

# Execute the plot
plot_mitigation_ratio()