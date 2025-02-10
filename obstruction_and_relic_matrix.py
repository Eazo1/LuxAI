import numpy as np
import matplotlib.pyplot as plt

map_iteration_period = 20 # Can be other values (must determine in game)
grand_obstruction_matrix_size = 24 + (500 // map_iteration_period) # check if // is correct
obstruction_direction = 1 # Range {-1,1}, with 1 meaning the little matrix moves northeast, and -1 meaning it moves southwest

class SymmetricMatrix:
  def __init__(self, size):
    self.size = size
    self.data = np.zeros((size, size))
    
  def get_entangled_index(self, i, j):
    return self.size - j - 1, self.size - i - 1
  
  def set_index_value(self, index, value):
    i, j = index
    self.data[i, j] = value
    anti_i, anti_j = self.get_entangled_index(i, j)
    self.data[anti_i, anti_j] = value
  
  def get_index_value(self, index):
    return self.data[index]


###


class GrandObstructionMatrix():
  def __init__(self, size):
    self.size = size
    self.data = np.zeros((size, size))

  def get_entangled_iteration_index(self, iteration_i, iteration_j):
    return 24 - iteration_j - 1, 24 - iteration_i - 1
  
  def get_entangled_tiles_iteration_index(self, iteration_i, iteration_j):
    
    negative_side_n = (iteration_i//24)
    positive_side_n = (self.size - iteration_i)//24
    
    negative_side_m = (iteration_j//24)
    positive_side_m = (self.size - iteration_j)//24
    
    list_of_n = [n for n in range(-negative_side_n, positive_side_n + 1)]
    list_of_m = [m for m in range(-negative_side_m, positive_side_m + 1)]
        
    list_of_indices = [(iteration_i + n * 24, iteration_j + m * 24) for n in list_of_n for m in list_of_m if 0 <= (iteration_i + n * 24) < self.size and 0 <= (iteration_j + m * 24) < self.size]
    
    return list_of_indices
    
  def safe_update(self, i, j, value):
      if 0 <= i < self.size and 0 <= j < self.size:
          self.data[i, j] = value

  def set_index_value(self, map_iteration_period, time_iteration, iteration_index, value):
    obstruction_movement_step = time_iteration // map_iteration_period  # integer division
    iteration_i, iteration_j = iteration_index

    anti_iteration_i, anti_iteration_j = self.get_entangled_iteration_index(iteration_i, iteration_j)
    entangled_tile_indices = self.get_entangled_tiles_iteration_index(iteration_i, iteration_j)

    if obstruction_direction == 1:
        # Primary tile update with modulo
        target_i = (self.size - 24 - obstruction_movement_step + iteration_i) % self.size
        target_j = (obstruction_movement_step + iteration_j) % self.size
        self.safe_update(target_i, target_j, value)
        
        # Update the symmetric partner of the primary tile
        target_i = (self.size - 24 - obstruction_movement_step + anti_iteration_i) % self.size
        target_j = (obstruction_movement_step + anti_iteration_j) % self.size
        self.safe_update(target_i, target_j, value)
        
        # Update each of the entangled tiles
        for tile_i, tile_j in entangled_tile_indices:
            tile_anti_i, tile_anti_j = self.get_entangled_iteration_index(tile_i, tile_j)
            self.safe_update((self.size - 24 - obstruction_movement_step + tile_i) % self.size,
                             (obstruction_movement_step + tile_j) % self.size, value)
            self.safe_update((self.size - 24 - obstruction_movement_step + tile_anti_i) % self.size,
                             (obstruction_movement_step + tile_anti_j) % self.size, value)
    else:
        # Similar logic for obstruction_direction == -1, applying modulo to both row and col.
        self.safe_update((obstruction_movement_step + iteration_i) % self.size,
                         (self.size - 24 - obstruction_movement_step + iteration_j) % self.size, value)
        self.safe_update((obstruction_movement_step + anti_iteration_i) % self.size,
                         (self.size - 24 - obstruction_movement_step + anti_iteration_j) % self.size, value)
        
        for tile_i, tile_j in entangled_tile_indices:
            tile_anti_i, tile_anti_j = self.get_entangled_iteration_index(tile_i, tile_j)
            self.safe_update((obstruction_movement_step + tile_i) % self.size,
                             (self.size - 24 - obstruction_movement_step + tile_j) % self.size, value)
            self.safe_update((obstruction_movement_step + tile_anti_i) % self.size,
                             (self.size - 24 - obstruction_movement_step + tile_anti_j) % self.size, value)


  def get_index_value(self, index):
      i, j = index
      
      if not (0 <= i < self.size and 0 <= j < self.size):
          raise IndexError("Index out of bounds.")

      return self.data[i, j]
  
  def get_obstruction_matrix_iteration (self, map_iteration_period, time_iteration):
    total_size = self.size
    
    obstruction_movement_step = time_iteration // map_iteration_period # make sure // is correct here (for the game)
    
    if obstruction_direction == 1:
      obstruction_matrix_iteration = self.data[(total_size - 24 - obstruction_movement_step) : (total_size - obstruction_movement_step),
                                                              (obstruction_movement_step) : (24 + obstruction_movement_step)]
    else:
      obstruction_matrix_iteration = self.data[(obstruction_movement_step) : (24 + obstruction_movement_step),
                                                              (total_size - 24 - obstruction_movement_step) : (total_size - obstruction_movement_step)]
      
    return obstruction_matrix_iteration

# Trial simulation

# Initialize grand obstruction matrix
grand_obstruction_matrix = GrandObstructionMatrix(grand_obstruction_matrix_size)

# Set some values in the matrix (for the sake of this trial)
for time_iteration in range(0, 100, 5):  # Simulating 20 steps
    for i in range(24):  # Simulate a 24x24 section
        for j in range(24):
            value = np.random.rand()  # Random value for demonstration
            grand_obstruction_matrix.set_index_value(map_iteration_period, time_iteration, (i, j), value)

# Extract the 24×24 slice of the obstruction matrix at iteration 95
obstruction_matrix = grand_obstruction_matrix.get_obstruction_matrix_iteration(20, 0)

# Extract the full grand obstruction matrix
full_matrix = grand_obstruction_matrix.data  # The entire matrix

# Set up side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Normalize color scale across both plots
vmin, vmax = np.min(full_matrix), np.max(full_matrix)

# Plot the 24x24 slice
im1 = axes[0].imshow(obstruction_matrix, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
axes[0].set_title("24×24 Slice at Iteration 0")
fig.colorbar(im1, ax=axes[0])

# Plot the full grand obstruction matrix
im2 = axes[1].imshow(full_matrix, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
axes[1].set_title("Full Grand Obstruction Matrix")
fig.colorbar(im2, ax=axes[1])

# Save and show the plot
plt.tight_layout()
plt.savefig("obstruction_matrix_comparison.png")
plt.show()

