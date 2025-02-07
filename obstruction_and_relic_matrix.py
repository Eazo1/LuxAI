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

#grand_obstruction_matrix = SymmetricMatrix(grand_obstruction_matrix_size)
#obstruction_matrix_i = SymmetricMatrix(24)

class GrandObstructionMatrix(SymmetricMatrix):
  def __init__(self, size):
    super().__init__(size)

  def get_entangled_iteration_index(self, iteration_i, iteration_j):
    return 24 - iteration_j - 1, 24 - iteration_i - 1
  
  def set_index_value(self, map_iteration_period, time_iteration, iteration_index, value):
    
    obstruction_movement_step = time_iteration // map_iteration_period # make sure // is correct here (for the game)
    
    iteration_i, iteration_j = iteration_index
    
    if obstruction_direction == 1:
      self.data[(self.size - 24 - obstruction_movement_step + iteration_i), (obstruction_movement_step + iteration_j)] = value
      anti_iteration_i, anti_iteration_j = self.get_entangled_iteration_index(iteration_i, iteration_j)
      
      #print( 'obstruction movement step:', obstruction_movement_step)
      #print( 'anti-iteration i:', anti_iteration_i)
      #print('self size:', self.size)
      #print('total:', self.size - 24 - obstruction_movement_step + anti_iteration_i)
      
      self.data[(self.size - 24 - obstruction_movement_step + anti_iteration_i), (obstruction_movement_step + anti_iteration_j)] = value
    else:
      self.data[(obstruction_movement_step + iteration_i), (self.size - 24 - obstruction_movement_step + iteration_j)] = value
      anti_iteration_i, anti_iteration_j = self.get_entangled_iteration_index(iteration_i, iteration_j)
      self.data[(obstruction_movement_step + anti_iteration_i), (self.size - 24 - obstruction_movement_step + anti_iteration_j)] = value

  def get_index_value(self, index):
      """
      Retrieves the value at a given index.
      """
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
        for j in range(24 - i):
            value = np.random.rand()  # Random value for demonstration
            grand_obstruction_matrix.set_index_value(map_iteration_period, time_iteration, (i, j), value)

# Create plot of the final iteration's obstruction matrix
obstruction_matrix = grand_obstruction_matrix.get_obstruction_matrix_iteration(20, 95)

# Plot the matrix
plt.imshow(obstruction_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("Grand Obstruction Matrix at Time Iteration 95")
plt.savefig('obstruction_matrix.png')
