import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_lxcat_cs_file(filename):
    """Parse LXCat cross-section data file and extract all processes."""
    processes = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for process type
        if line in ['ELASTIC', 'EXCITATION', 'IONIZATION', 'ATTACHMENT', 'EFFECTIVE']:
            process_type = line
            current_process = {'type': process_type, 'energies': [], 'cross_sections': []}
            i += 1
            
            # Get species/target info
            if i < len(lines):
                current_process['target'] = lines[i].strip()
                i += 1
            
            # Get threshold energy
            if i < len(lines):
                try:
                    params = lines[i].strip().split()
                    current_process['threshold'] = float(params[0])
                except (ValueError, IndexError):
                    current_process['threshold'] = 0
                i += 1
            
            # Skip to data section (find dashes)
            while i < len(lines) and lines[i].strip() != '-----------------------------':
                i += 1
            i += 1
            
            # Read data until next dashes
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
        else:
            i += 1
    
    return processes

def plot_all_processes(filename):
    """Plot all processes from the LXCat file."""
    processes = parse_lxcat_cs_file(filename)
    
    if not processes:
        print("No processes found!")
        return
    
    # Create figure with subplots
    num_processes = len(processes)
    cols = 2
    rows = (num_processes + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5*rows))
    axes = axes.flatten() if num_processes > 1 else [axes]
    
    colors = ['b', 'r', 'g', 'orange', 'purple']
    
    for idx, process in enumerate(processes):
        ax = axes[idx]
        energies = np.array(process['energies'])
        cross_sections = np.array(process['cross_sections']) * 1e24  # Convert to 10^-24 m^2
        
        color = colors[idx % len(colors)]
        ax.semilogy(energies, cross_sections, color=color, linewidth=2, marker='o', markersize=3, label=process['type'])
        ax.set_xlabel('Energy (eV)', fontsize=11)
        ax.set_ylabel('Cross Section (10⁻²⁴ m²)', fontsize=11)
        ax.set_title(f"{process['target']}\nThreshold: {process['threshold']:.3f} eV", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    # Hide unused subplots
    for idx in range(num_processes, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('lxcat_all_processes.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as 'lxcat_all_processes.png'")
    print(f"Total processes plotted: {num_processes}")
    for process in processes:
        print(f"  - {process['type']}: {process['target']} (Threshold: {process['threshold']:.3f} eV)")

if __name__ == '__main__':
    # Chemin absolu au fichier
    file_path = Path(__file__).resolve().parent / 'LXCat_CS.dat'
    plot_all_processes(str(file_path))