import numpy as np
import matplotlib.pyplot as plt

map_iteration_period = 20 # Can be other values (must determine in game)
grand_obstruction_matrix_size = 24 + (500 // map_iteration_period) # check if // is correct
#obstruction_direction = 1 # Range {-1,1}, with 1 meaning the little matrix moves northeast, and -1 meaning it moves southwest

class Unit_Ally:
  def __init__(self, unit_id, energy, index):
    self.unit_id = unit_id
    self.energy = energy
    self.index = index
  
  def set_energy(self, energy):
    self.energy = energy
    
  def set_index(self, index):
    self.index = index
  
  def get_energy(self):
    return self.energy
  
  def get_index(self):
    return self.index
  
class Unit_Enemy:
  def __init__(self, unit_id, energy, index):
    self.unit_id = unit_id
    self.energy = energy
    self.index = index
  
  def set_energy(self, energy):
    self.energy = energy
  
  def set_index(self, index):
    self.index = index
  
  def get_energy(self):
    return self.energy
  
  def get_index(self):
    return self.index

class UserMatrix:
  def __init__(self, size=24):
    self.size = size
    self.data = np.zeros((size, size), dtype=object)
    self.unit_positions = {}  # Dictionary to store unit_ids and their positions
  
  def get_entangled_index(self, i, j):
    return self.size - j - 1, self.size - i - 1
  
  def remove_unit_from_matrix(self, unit):
    # Look up the unit in the dictionary and remove it if found
    if unit.unit_id in self.unit_positions:
      i, j = self.unit_positions[unit.unit_id]
      # Remove unit from the ally or enemy list
      if unit in self.data[i, j][0]:
        self.data[i, j][0].remove(unit)  # This can be either ally or enemy
      elif unit in self.data[i, j][1]:
        self.data[i, j][1].remove(unit)
      # If both lists are empty, clear the matrix position
      if not self.data[i, j][0] and not self.data[i, j][1]:
        self.data[i, j] = [None, None, None]
      # Remove from the unit_positions dictionary
      del self.unit_positions[unit.unit_id]
      return True
    return False
  
  def set_index_value(self, index, ally_units, enemy_units, value):
    # Ensure a boolean
    assert isinstance(value, bool), "The third value must be a boolean."
    
    # Only remove units that already exist in the matrix
    for unit in ally_units + enemy_units:  # Check both ally and enemy units
        if unit.unit_id in self.unit_positions:
            self.remove_unit_from_matrix(unit)
    
    # Now place the new units into the matrix
    i, j = index
    self.data[i, j] = [ally_units, enemy_units, value]
    
    # Add the new units to the dictionary
    for unit in ally_units + enemy_units:  # Add both ally and enemy units
        self.unit_positions[unit.unit_id] = (i, j)
    
    anti_i, anti_j = self.get_entangled_index(i, j)
    self.data[anti_i, anti_j] = [None, None, value]

  
  def get_index_value(self, index):
    return self.data[index]

  
###

class GrandObstructionMatrix():
  def __init__(self, size, node_info_size=2, obstruction_direction=1):
    self.size = size
    self.node_info_size = node_info_size
    self.data = np.zeros((int(size), int(size)), dtype=object)
    self.obstruction_direction = 1  # Default value
    for i in range(size):
      for j in range(size):
        self.data[i, j] = np.zeros(self.node_info_size)  

  def set_obstruction_direction(self, direction):
    self.obstruction_direction = direction
  
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
    
  def safe_update(self, i, j, discrete_values):
    i = int(i)
    j = int(j)
    if 0 <= i < self.size and 0 <= j < self.size:
      self.data[i, j] = np.array(discrete_values)

  def set_index_value(self, map_iteration_period, time_iteration, iteration_index, discrete_values):
    obstruction_movement_step = time_iteration // map_iteration_period  # integer division
    iteration_i, iteration_j = iteration_index

    anti_iteration_i, anti_iteration_j = self.get_entangled_iteration_index(iteration_i, iteration_j)
    entangled_tile_indices = self.get_entangled_tiles_iteration_index(iteration_i, iteration_j)

    if self.obstruction_direction == 1:
      target_i = (self.size - 24 - obstruction_movement_step + iteration_i) % self.size
      target_j = (obstruction_movement_step + iteration_j) % self.size
      self.safe_update(target_i, target_j, discrete_values)
      
      target_i = (self.size - 24 - obstruction_movement_step + anti_iteration_i) % self.size
      target_j = (obstruction_movement_step + anti_iteration_j) % self.size
      self.safe_update(target_i, target_j, discrete_values)
        
      for tile_i, tile_j in entangled_tile_indices:
        tile_anti_i, tile_anti_j = self.get_entangled_iteration_index(tile_i, tile_j)
        self.safe_update((self.size - 24 - obstruction_movement_step + tile_i) % self.size,
                          (obstruction_movement_step + tile_j) % self.size, discrete_values)
        self.safe_update((self.size - 24 - obstruction_movement_step + tile_anti_i) % self.size,
                          (obstruction_movement_step + tile_anti_j) % self.size, discrete_values)
    else:
      self.safe_update((obstruction_movement_step + iteration_i) % self.size,
                        (self.size - 24 - obstruction_movement_step + iteration_j) % self.size, discrete_values)
      self.safe_update((obstruction_movement_step + anti_iteration_i) % self.size,
                        (self.size - 24 - obstruction_movement_step + anti_iteration_j) % self.size, discrete_values)
        
      for tile_i, tile_j in entangled_tile_indices:
        tile_anti_i, tile_anti_j = self.get_entangled_iteration_index(tile_i, tile_j)
        self.safe_update((obstruction_movement_step + tile_i) % self.size,
                          (self.size - 24 - obstruction_movement_step + tile_j) % self.size, discrete_values)
        self.safe_update((obstruction_movement_step + tile_anti_i) % self.size,
                          (self.size - 24 - obstruction_movement_step + tile_anti_j) % self.size, discrete_values)
    
  
  def get_index_value(self, index):
      i, j = index
      
      if not (0 <= i < self.size and 0 <= j < self.size):
          raise IndexError("Index out of bounds.")

      return self.data[i, j]
  
  def get_obstruction_matrix_iteration (self, map_iteration_period, time_iteration):
    total_size = self.size
    
    obstruction_movement_step = time_iteration // map_iteration_period # make sure // is correct here (for the game)
    
    if self.obstruction_direction == 1:
      obstruction_matrix_iteration = self.data[(total_size - 24 - obstruction_movement_step) : (total_size - obstruction_movement_step),
                                                              (obstruction_movement_step) : (24 + obstruction_movement_step)]
    else:
      obstruction_matrix_iteration = self.data[(obstruction_movement_step) : (24 + obstruction_movement_step),
                                                              (total_size - 24 - obstruction_movement_step) : (total_size - obstruction_movement_step)]
      
    return obstruction_matrix_iteration

'''# Trial simulatizon

# Initialize grand obstruction matrix
grand_obstruction_matrix = GrandObstructionMatrix(grand_obstruction_matrix_size)

# Set some values in the matrix (for the sake of this trial)
for time_iteration in range(0, 100, 5):  # Simulating 20 steps
    for i in range(24):  # Simulate a 24x24 section
        for j in range(24):
            # Generate random discrete values
            discrete_values = np.random.randint(0, 10, size=grand_obstruction_matrix.node_info_size)  # Random integers for discrete values
            grand_obstruction_matrix.set_index_value(map_iteration_period, time_iteration, (i, j), discrete_values)

# Extract the 24×24 slice of the obstruction matrix at iteration 0 (start of the simulation)
obstruction_matrix = grand_obstruction_matrix.get_obstruction_matrix_iteration(map_iteration_period, 0)

# Convert the matrix to a numeric form for visualization (sum of the discrete values for each element)
numeric_obstruction_matrix = np.array([[np.sum(grand_obstruction_matrix.get_index_value((i, j))) for j in range(24)] for i in range(24)])

# Extract the full grand obstruction matrix (convert similarly)
full_matrix = np.array([[np.sum(grand_obstruction_matrix.get_index_value((i, j))) for j in range(grand_obstruction_matrix.size)] for i in range(grand_obstruction_matrix.size)])

# Set up side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Normalize color scale across both plots
vmin, vmax = np.min(full_matrix), np.max(full_matrix)

# Plot the 24x24 slice
im1 = axes[0].imshow(numeric_obstruction_matrix, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
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
'''
# Create a UserMatrix instance
matrix = UserMatrix(size=24)

# Create 1 unit for ally and enemy (you can modify this to test with more units)
ally_units = [Unit_Ally(unit_id=1, energy=100, index=(0, 0))]
enemy_units = [Unit_Enemy(unit_id=2, energy=50, index=(0, 0))]

# Test: Set one unit in the matrix
index = (0, 0)
value = True  # Arbitrary boolean value for testing
matrix.set_index_value(index, ally_units, enemy_units, value)

# Test: Check if units were added to the matrix correctly
matrix_value = matrix.get_index_value(index)
print(f"Matrix value at {index}: {matrix_value}")

# Test: Ensure ally and enemy units are correctly stored
assert matrix_value[0] == ally_units, "Ally units not added correctly!"
assert matrix_value[1] == enemy_units, "Enemy units not added correctly!"
assert matrix_value[2] == value, "Matrix value should be the boolean value"

print("All tests passed!")

