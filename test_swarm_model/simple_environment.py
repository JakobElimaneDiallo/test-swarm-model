import random
import numpy as np
import matplotlib.patches as patches

class SimpleEnvironment:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        # Create random grid where 30-40% of squares are vibrating
        num_squares = grid_size * grid_size
        num_vibrating = random.randint(int(0.6 * num_squares), int(0.8 * num_squares))
        
        # Initialize grid with all False
        self.grid = np.zeros((grid_size, grid_size), dtype=bool)
        
        # Randomly set squares to vibrating
        positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        vibrating_positions = random.sample(positions, num_vibrating)
        for x, y in vibrating_positions:
            self.grid[x, y] = True
            
        # Store rectangle patches for visualization
        self.patches = []
        
    def add_to_axes(self, ax):
        # Clear existing patches
        self.patches.clear()
        
        # Size of each square
        square_size = 1.0 / self.grid_size
        
        # Create and add patches
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = j * square_size
                y = i * square_size
                is_vibrating = self.grid[i, j]
                
                # Create square patch
                square = patches.Rectangle(
                    (x, y), square_size, square_size,
                    facecolor='gray' if is_vibrating else 'white',
                    edgecolor='black',
                    alpha=0.3 if is_vibrating else 0.1
                )
                ax.add_patch(square)
                self.patches.append(square)
                
    def update_patches(self, frame):
        # Make vibrating squares "vibrate" by slightly adjusting their alpha
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j]:
                    patch = self.patches[i * self.grid_size + j]
                    # Create subtle vibration effect with sine wave
                    alpha = 0.3 + 0.1 * np.sin(frame * 0.5)
                    patch.set_alpha(alpha)
                    
    def get_vibrating_ratio(self):
        return np.sum(self.grid) / (self.grid_size * self.grid_size) * 100
