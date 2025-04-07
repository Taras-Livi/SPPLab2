import os
import re
import matplotlib.pyplot as plt
import numpy as np


def read_input_file(filepath="input.txt"):
    """Read seed, width, and height from input file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 3:
                seed = int(lines[0].strip())
                width = int(lines[1].strip())
                height = int(lines[2].strip())
                return seed, width, height
            else:
                raise ValueError("Input file must contain at least 3 lines: seed, width, height")
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None, None, None
    except ValueError as e:
        print(f"Error parsing input file: {e}")
        return None, None, None


def extract_performance_data(output_files):
    """Extract execution time and worker count from output files."""
    performance_data = []

    for file in output_files:
        try:
            with open(file, 'r') as f:
                content = f.read()

                # Extract execution time
                time_match = re.search(r'Execution time: (\d+\.\d+) seconds', content)
                if time_match:
                    execution_time = float(time_match.group(1))
                else:
                    print(f"Warning: Could not find execution time in {file}")
                    continue

                # Extract worker count
                workers_match = re.search(r'Workers: (\d+)', content)
                if workers_match:
                    workers = int(workers_match.group(1))
                else:
                    print(f"Warning: Could not find worker count in {file}")
                    continue

                performance_data.append((workers, execution_time))
        except FileNotFoundError:
            print(f"Warning: File {file} not found.")
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    return performance_data


def find_output_files():
    """Find all output files in the current directory."""
    output_files = []
    pattern = re.compile(r'output \((\d+)\)\.txt')

    for file in os.listdir():
        if pattern.match(file):
            output_files.append(file)

    return sorted(output_files)


def plot_performance(performance_data, width, height):
    """Plot the relationship between execution time and worker count."""
    if not performance_data:
        print("No performance data to plot.")
        return

    # Sort data by worker count
    performance_data.sort(key=lambda x: x[0])

    workers = [data[0] for data in performance_data]
    times = [data[1] for data in performance_data]

    plt.figure(figsize=(10, 6))
    plt.plot(workers, times, 'o-', linewidth=2, markersize=8)

    # Add polynomial trend line
    if len(workers) > 2:
        z = np.polyfit(workers, times, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(min(workers), max(workers), 100)
        plt.plot(x_trend, p(x_trend), '--', color='red', linewidth=1)

    plt.title(f'Execution Time vs. Number of Workers\n(Map size: {width}x{height})')
    plt.xlabel('Number of Workers')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, alpha=0.3)

    # Mark the minimum execution time point
    min_time_idx = times.index(min(times))
    min_workers = workers[min_time_idx]
    min_time = times[min_time_idx]
    plt.scatter([min_workers], [min_time], color='green', s=100, zorder=5)
    '''plt.annotate(f'Optimal: {min_workers} workers\n({min_time:.4f} sec)',
                 (min_workers, min_time),
                 xytext=(10, -30),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))
    '''
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Optimal performance: {min_time:.4f} seconds with {min_workers} workers")
    return min_workers


def analyze_speedup(performance_data):
    """Calculate and plot speedup and efficiency."""
    if not performance_data:
        return

    # Sort data by worker count
    performance_data.sort(key=lambda x: x[0])

    workers = [data[0] for data in performance_data]
    times = [data[1] for data in performance_data]

    # Calculate speedup (T1/Tp) and efficiency (Speedup/p)
    # Using the lowest worker count as baseline if no single worker data
    baseline_time = times[0]
    baseline_workers = workers[0]

    speedups = [baseline_time / time for time in times]
    efficiencies = [speedup / (worker / baseline_workers) for speedup, worker in zip(speedups, workers)]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot speedup
    ax1.plot(workers, speedups, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.plot(workers, [w / baseline_workers for w in workers], '--', color='gray', label='Ideal linear speedup')
    ax1.set_title('Speedup vs. Number of Workers')
    ax1.set_xlabel('Number of Workers')
    ax1.set_ylabel('Speedup (T1/Tp)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot efficiency
    ax2.plot(workers, efficiencies, 'o-', linewidth=2, markersize=8, color='green')
    ax2.axhline(y=1.0, linestyle='--', color='gray', label='Ideal efficiency')
    ax2.set_title('Efficiency vs. Number of Workers')
    ax2.set_xlabel('Number of Workers')
    ax2.set_ylabel('Efficiency (Speedup/p)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('speedup_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Read input parameters
    seed, width, height = read_input_file()
    if seed is None or width is None or height is None:
        return

    # Find output files
    output_files = find_output_files()
    if not output_files:
        print("No output files found. Please ensure output files follow the naming pattern: output(n).txt")
        return

    print(f"Found {len(output_files)} output files: {', '.join(output_files)}")

    # Extract performance data
    performance_data = extract_performance_data(output_files)
    if not performance_data:
        print("No valid performance data found.")
        return

    print("Performance Data (Workers, Time):")
    for workers, time in performance_data:
        print(f"  Workers: {workers}, Time: {time:.4f} seconds")

    # Plot performance
    optimal_workers = plot_performance(performance_data, width, height)

    # Analyze speedup and efficiency
    analyze_speedup(performance_data)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()