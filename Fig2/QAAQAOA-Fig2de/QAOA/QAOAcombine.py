import numpy as np

def synthesize_qaoa_data(total_layers=16, output_name="QAOAdraw_combined.npz"):
    """Merge QAOAdraw{t}.npz files (t=1..total_layers) into a single combined .npz dataset."""
    all_x = []
    all_energy = []
    all_possibility = []

    print(f"Merging QAOAdraw1 ... QAOAdraw{total_layers} into {output_name}...")

    for t in range(1, total_layers + 1):
        filename = f"QAOAdraw{t}.npz"
        try:
            data = np.load(filename)

            all_x.append(data["x"][0])
            all_energy.append(data["energylist"][0])
            all_possibility.append(data["possibility"][0])

            print(f"Loaded: {filename} | Energy: {all_energy[-1]:.6f}, Probability: {all_possibility[-1]:.6f}")
        except FileNotFoundError:
            print(f"Warning: missing file {filename}; skipped.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    combined_x = np.array(all_x)
    combined_energy = np.array(all_energy)
    combined_possibility = np.array(all_possibility)

    sort_idx = np.argsort(combined_x)
    combined_x = combined_x[sort_idx]
    combined_energy = combined_energy[sort_idx]
    combined_possibility = combined_possibility[sort_idx]

    np.savez(
        output_name,
        x=combined_x,
        energylist=combined_energy,
        possibility=combined_possibility,
    )

    print("-" * 30)
    print(f"Done. Total points: {len(combined_x)}")
    print(f"Saved: {output_name}")

if __name__ == "__main__":
    synthesize_qaoa_data()
