import numpy as np

class Environment:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.total_squares = grid_size * grid_size
        self.vibrating_squares = set()
        self.setup_environment()
        
    def setup_environment(self):
        # Set between 40-60% of squares to vibrate
        num_vibrating = np.random.randint(
            int(self.total_squares * 0.4),
            int(self.total_squares * 0.6) + 1
        )
        self.vibrating_squares = set(np.random.choice(
            self.total_squares, 
            num_vibrating, 
            replace=False
        ))
        self.true_ratio = len(self.vibrating_squares) / self.total_squares
        
    def get_sample(self, x, y):
        # Convert continuous position to grid position
        grid_x = int(x * self.grid_size)
        grid_y = int(y * self.grid_size)
        grid_x = min(max(grid_x, 0), self.grid_size - 1)
        grid_y = min(max(grid_y, 0), self.grid_size - 1)
        
        # Convert to index
        idx = grid_y * self.grid_size + grid_x
        
        # Return 1 if square is vibrating, 0 otherwise
        return 1 if idx in self.vibrating_squares else 0
        
    def get_vibrating_squares(self):
        return self.vibrating_squares
