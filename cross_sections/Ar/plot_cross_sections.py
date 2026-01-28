import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_lxcat_file(filename):
    """Parse LXCat cross-section data file and extract excitation processes."""
    processes = []
    current_process = None
    reading_data = False
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line == 'EXCITATION':
            current_process = {'type': 'EXCITATION', 'energies': [], 'cross_sections': []}
            reading_data = False
            i += 1
            
            # Get target and state info
            if i < len(lines):
                current_process['target'] = lines[i].strip()
                i += 1
            
            # Get threshold energy and statistical weight
            if i < len(lines):
                params = lines[i].strip().split()
                current_process['threshold'] = float(params[0])
                current_process['stat_weight'] = float(params[1])
                i += 1
            
            # Skip to data section
            while i < len(lines) and lines[i].strip() != '-----------------------------':
                i += 1
            reading_data = True
            i += 1
            
            # Read data
            while i < len(lines) and lines[i].strip() != '-----------------------------':
                parts = lines[i].strip().split()
                if len(parts) == 2:
                    try:
                        energy = float(parts[0])
                        cross_section = float(parts[1])
                        current_process['energies'].append(energy)
                        current_process['cross_sections'].append(cross_section)
                    except ValueError:
                        pass
                i += 1
            
            if current_process['energies']:
                processes.append(current_process)
            reading_data = False
        else:
            i += 1
    
    return processes

def plot_total_cross_section(filename):
    """Plot the total (sum) of all excitation cross-sections from the file."""
    processes = parse_lxcat_file(filename)
    
    if not processes:
        print("No excitation processes found!")
        return
    
    # Create a dictionary to sum cross-sections at each energy
    energy_cs_sum = {}
    
    for process in processes:
        for energy, cs in zip(process['energies'], process['cross_sections']):
            if energy not in energy_cs_sum:
                energy_cs_sum[energy] = 0
            energy_cs_sum[energy] += cs
    
    # Sort by energy
    sorted_energies = sorted(energy_cs_sum.keys())
    total_cs = [energy_cs_sum[e] for e in sorted_energies]
    
    # Convert to 10^-24 m^2
    total_cs = np.array(total_cs) * 1e24
    
    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(sorted_energies, total_cs, 'b-', linewidth=2.5, marker='o', markersize=4, label='Total Excitation')
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('Total Cross Section (10⁻²⁴ m²)', fontsize=12)
    ax.set_title('Total Argon Excitation Cross Section (Sum of all processes)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('argon_excitation_total.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as 'argon_excitation_total.png'")
    print(f"Total excitation processes summed: {len(processes)}")

if __name__ == '__main__':
    # Chemin absolu au fichier
    file_path = Path(__file__).resolve().parent / 'Testcross.dat'
    plot_total_cross_section(str(file_path))