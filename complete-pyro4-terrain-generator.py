from Pyro4 import expose
import random
import time
import math

class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        self.num_workers = len(workers) if workers is not None else 4
        print("Initialized with", self.num_workers, "workers")

    def solve(self):
        print("Job Started")
        print("Workers %d" % len(self.workers))

        start_time = time.time()

        # Read input parameters from file
        seed, width, height = self.read_input()

        # Generate the base noise that will ensure coherent patterns across worker boundaries
        # This is sent to all workers to ensure they use the same underlying terrain structure
        base_noise = self.generate_base_noise(seed, width, height)

        # Calculate rows per worker
        rows_per_worker = height // len(self.workers)

        # Prepare arguments and distribute work to each worker
        mapped = []
        for i in range(len(self.workers)):
            # Determine which rows this worker will process
            start_row = i * rows_per_worker
            # The last worker gets any remaining rows
            end_row = (i + 1) * rows_per_worker if i < len(self.workers) - 1 else height
            print("map %d" % i)

            # Each worker gets a unique seed derived from the master seed
            # This ensures different random variations while maintaining coherence
            worker_seed = seed + i * 1000

            # Send work to the worker with all necessary parameters
            mapped.append(self.workers[i].mymap(
                str(worker_seed),  # Worker-specific seed for deterministic but varied results
                str(width), 
                str(start_row), 
                str(end_row), 
                base_noise  # The shared noise grid for terrain coherence
            ))

        # Reduce phase - combine all the map portions from each worker
        map_data = self.myreduce(mapped)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Add execution time and worker count to the map data
        map_data.append("Execution time: %.4f seconds" % execution_time)
        map_data.append("Workers: %d" % self.num_workers)

        # Write the final map to output file
        self.write_output(map_data)

        print("Job Finished")

    @staticmethod
    @expose
    def mymap(worker_seed, width, start_row, end_row, base_noise):
        """
        Generate a portion of the terrain map.
        This function is executed on worker nodes.
        
        Parameters:
        - worker_seed: Unique seed for this worker to ensure varied but deterministic results
        - width: Width of the map
        - start_row, end_row: Range of rows this worker is responsible for
        - base_noise: Pre-generated noise grid for terrain coherence
        """
        # Convert string parameters to appropriate types
        worker_seed = int(worker_seed)
        width = int(width)
        start_row = int(start_row)
        end_row = int(end_row)

        print("Generating rows %d to %d with seed %d" % (start_row, end_row, worker_seed))
        
        # Initialize the random generator with worker seed
        # This ensures different workers produce different variations
        random.seed(worker_seed)

        # Terrain symbols (ASCII characters representing different terrain types)
        symbols = {
            'mountain': '^',
            'hills': '.',
            'forest': '*',
            'plains': ' ',
            'water': '~',
            'desert': '_'
        }

        # Extract noise grid parameters
        grid = base_noise['grid']        # Primary elevation grid
        grid2 = base_noise['grid2']      # Secondary moisture grid
        scale = base_noise['scale']      # Scaling factor for elevation grid
        grid2_scale = base_noise['grid2_scale']  # Scaling factor for moisture grid

        map_rows = []
        # Generate map data for each assigned row
        for y in range(start_row, end_row):
            row = []
            for x in range(width):
                # Set a position-specific seed to ensure consistent results
                # regardless of which worker processes this position
                point_seed = worker_seed + (y * width + x)
                random.seed(point_seed)
                
                # Get base elevation and moisture from the noise grids using interpolation
                elevation = Solver.bilinear_interpolate(grid, x, y, scale)
                moisture = Solver.bilinear_interpolate(grid2, x, y, grid2_scale)

                # Add small random variations to create micro-terrain features
                # Using a position-based seed ensures these are consistent
                elevation += (random.random() * 0.1) - 0.05
                moisture += (random.random() * 0.1) - 0.05

                # Ensure values stay within valid [0.0, 1.0] range
                elevation = max(0.0, min(1.0, elevation))
                moisture = max(0.0, min(1.0, moisture))

                # Determine terrain type based on elevation and moisture values
                # This creates varied but coherent terrain patterns
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

                # Add the terrain symbol to the current row
                row.append(symbols[terrain])

            # Add the completed row to our map data
            map_rows.append(''.join(row))

        # Return the generated map rows for this worker's section
        return map_rows

    @staticmethod
    @expose
    def myreduce(mapped):
        """
        Combine the map portions from all workers into a complete map.
        The map portions are combined in the order they were assigned to workers.
        """
        print("reduce")
        combined_map = []

        # Process each worker's results
        for map_portion in mapped:
            print("reduce loop")
            # In Pyro4, we need to access the .value property to get the actual data
            # Then extend our combined map with these rows
            combined_map.extend(map_portion.value)
            
        print("reduce done")
        return combined_map

    def read_input(self):
        """Read seed, width, and height parameters from the input file"""
        with open(self.input_file_name, 'r') as f:
            seed = int(f.readline())
            width = int(f.readline())
            height = int(f.readline())
        return seed, width, height

    def write_output(self, output):
        """Write the complete map to the output file"""
        with open(self.output_file_name, 'w') as f:
            for line in output:
                f.write(line + '\n')
        print("output done")

    @staticmethod
    @expose
    def generate_base_noise(seed, width, height):
        """
        Generate coherent noise grids for terrain elevation and moisture.
        These grids ensure terrain patterns remain consistent across worker boundaries.
        
        Parameters:
        - seed: Random seed for reproducible noise generation
        - width, height: Dimensions of the final map
        
        Returns:
        - A dictionary containing two noise grids and their scaling factors
        """
        # Initialize random generator with the provided seed
        random.seed(seed)
    
        # Use a smaller scale factor for more natural-looking terrain features
        # Scale=4 creates medium-sized terrain features (similar to multiprocessing version)
        scale = 4.0
        
        # Calculate grid dimensions with extra padding to prevent boundary issues
        grid_width = int(math.ceil(width / scale) + 3)
        grid_height = int(math.ceil(height / scale) + 3)
    
        # Generate random values at grid points for elevation
        # This creates a low-resolution grid that will be interpolated
        grid = [[random.random() for _ in range(grid_width)] for _ in range(grid_height)]
    
        # Second grid with larger features for moisture/biome variation
        grid2_scale = 16.0  # Larger scale = bigger moisture regions
        grid2_width = int(math.ceil(width / grid2_scale) + 3)
        grid2_height = int(math.ceil(height / grid2_scale) + 3)
        grid2 = [[random.random() for _ in range(grid2_width)] for _ in range(grid2_height)]
    
        # Return all noise components in a dictionary
        return {
            'grid': grid,           # Primary elevation grid
            'grid2': grid2,         # Secondary moisture grid
            'scale': scale,         # Scaling factor for elevation
            'grid2_scale': grid2_scale  # Scaling factor for moisture
        }

    @staticmethod
    @expose
    def bilinear_interpolate(grid, x, y, scale):
        """
        Perform bilinear interpolation to get smooth noise values between grid points.
        This creates continuous terrain rather than blocky patterns.
        
        Parameters:
        - grid: 2D grid of noise values
        - x, y: Position to sample
        - scale: Scaling factor (higher = more zoomed out)
        
        Returns:
        - Interpolated value at the requested position
        """
        # Find the grid cell containing this position
        x1 = int(x / scale)
        y1 = int(y / scale)

        # Find the second corner of the cell, with boundary checking
        x2 = min(x1 + 1, len(grid[0]) - 1)
        y2 = min(y1 + 1, len(grid) - 1)

        # Safety check for negative coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)

        # Calculate normalized position within the cell (0.0 to 1.0)
        fx = (x / scale) - x1
        fy = (y / scale) - y1

        # Perform bilinear interpolation between the four corner points
        # First interpolate along the top and bottom edges
        top = grid[y1][x1] * (1 - fx) + grid[y1][x2] * fx
        bottom = grid[y2][x1] * (1 - fx) + grid[y2][x2] * fx

        # Then interpolate between top and bottom
        return top * (1 - fy) + bottom * fy
