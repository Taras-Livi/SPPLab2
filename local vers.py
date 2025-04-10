import multiprocessing as mp
import random
import time
import math


class Solver:
    def __init__(self, num_workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
        print("Initialized with", self.num_workers, "workers")

    def solve(self):
        print("Job Started")

        start_time = time.time()

        # Read input parameters
        seed, width, height = self.read_input()

        # Generate the base noise map that will be shared between workers
        # This ensures terrain patterns connect properly across worker boundaries
        base_noise = self.generate_base_noise(seed, width, height)

        # Determine rows per worker
        rows_per_worker = height // self.num_workers

        # Prepare arguments for each worker
        tasks = []
        for i in range(self.num_workers):
            start_row = i * rows_per_worker
            end_row = (i + 1) * rows_per_worker if i < self.num_workers - 1 else height
            tasks.append((seed + i, width, start_row, end_row, base_noise))

        # Map phase - each worker generates a portion of the map
        with mp.Pool(processes=self.num_workers) as pool:
            mapped = pool.starmap(self.mymap, tasks)

        # Reduce phase - combine the map portions
        map_data = self.myreduce(mapped)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Add execution time to the map data
        map_data.append(f"Execution time: {execution_time:.4f} seconds")
        map_data.append(f"Workers: {self.num_workers}")

        # Write the map to output file
        self.write_output(map_data)

        print("Job Finished")

    @staticmethod
    def generate_base_noise(seed, width, height):
        """Generate a coherent noise map that will be used by all workers"""
        random.seed(seed)

        # Create a lower resolution grid for the base noise
        scale = 8  # Smaller scale = larger features
        grid_width = math.ceil(width / scale) + 3  # Add more padding to prevent boundary issues
        grid_height = math.ceil(height / scale) + 3

        # Generate random values at grid points
        grid = [[random.random() for _ in range(grid_width)] for _ in range(grid_height)]

        # Second layer of noise with larger features for more variation
        grid2_scale = 16  # Larger scale for second layer
        grid2_width = math.ceil(width / grid2_scale) + 3
        grid2_height = math.ceil(height / grid2_scale) + 3
        grid2 = [[random.random() for _ in range(grid2_width)] for _ in range(grid2_height)]

        return {
            'grid': grid,
            'grid2': grid2,
            'scale': scale,
            'grid2_scale': grid2_scale
        }

    @staticmethod
    def bilinear_interpolate(grid, x, y, scale):
        """Perform bilinear interpolation to get smooth noise with boundary checking"""
        # Find grid cell coordinates
        x1 = int(x / scale)
        y1 = int(y / scale)

        # Make sure we're not accessing beyond grid boundaries
        x2 = min(x1 + 1, len(grid[0]) - 1)
        y2 = min(y1 + 1, len(grid) - 1)

        # Safety check in case x1 or y1 are negative
        x1 = max(0, x1)
        y1 = max(0, y1)

        # Get normalized coordinates within the cell (0 to 1)
        fx = (x / scale) - x1
        fy = (y / scale) - y1

        # Interpolate between the four corners of the grid cell
        top = grid[y1][x1] * (1 - fx) + grid[y1][x2] * fx
        bottom = grid[y2][x1] * (1 - fx) + grid[y2][x2] * fx

        return top * (1 - fy) + bottom * fy

    @staticmethod
    def mymap(worker_seed, width, start_row, end_row, base_noise):
        print(f"Generating rows {start_row} to {end_row}")
        random.seed(worker_seed)

        # Terrain symbols
        symbols = {
            'mountain': '^',
            'hills': '.',
            'forest': '*',
            'plains': ' ',
            'water': '~',
            'desert': '_'
        }

        # Extract base noise parameters
        grid = base_noise['grid']
        grid2 = base_noise['grid2']
        scale = base_noise['scale']
        grid2_scale = base_noise['grid2_scale']

        map_rows = []
        for y in range(start_row, end_row):
            row = []
            for x in range(width):
                # Get interpolated noise values from both grids
                elevation = Solver.bilinear_interpolate(grid, x, y, scale)
                moisture = Solver.bilinear_interpolate(grid2, x, y, grid2_scale)

                # Add some random variation for micro features
                elevation += (random.random() * 0.1) - 0.05
                moisture += (random.random() * 0.1) - 0.05

                # Clamp values to 0-1 range
                elevation = max(0, min(1, elevation))
                moisture = max(0, min(1, moisture))

                # Determine terrain type based on elevation and moisture
                if elevation > 0.75:  # High elevation
                    if elevation > 0.85:
                        terrain = 'mountain'
                    else:
                        terrain = 'hills'
                elif elevation < 0.3:  # Low elevation
                    if moisture > 0.4:
                        terrain = 'water'
                    else:
                        terrain = 'desert'
                else:  # Mid elevation
                    if moisture > 0.65:
                        terrain = 'forest'
                    else:
                        terrain = 'plains'

                row.append(symbols[terrain])

            map_rows.append(''.join(row))

        return map_rows

    @staticmethod
    def myreduce(mapped):
        print("reduce")
        combined_map = []

        for map_portion in mapped:
            combined_map.extend(map_portion)

        print("reduce done")
        return combined_map

    def read_input(self):
        with open(self.input_file_name, 'r') as f:
            seed = int(f.readline())
            width = int(f.readline())
            height = int(f.readline())
        return seed, width, height

    def write_output(self, output):
        with open(self.output_file_name, 'w') as f:
            for line in output:
                f.write(line + '\n')
        print("output done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate a terrain map using parallel processing')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')

    args = parser.parse_args()

    solver = Solver(
        num_workers=args.workers,
        input_file_name=args.input,
        output_file_name=args.output
    )

    solver.solve()