import numpy as np
import matplotlib.pyplot as plt
import json
import os
import obstruction_and_relic_matrix
import gen_map_from_game
import gc
import jax
import re


#map_iteration_period = 20 # Can be other values (must determine in game)
#grand_obstruction_matrix_size = 24 + (500 // map_iteration_period) # check if // is correct
#obstruction_direction = 1 # Range {-1,1}, with 1 meaning the little matrix moves northeast, and -1 meaning it moves southwest

'''random_seed = np.random.randint(0, 2**10)
map, relic_nodes, params = gen_map_from_game.gen_game_map(random_seed)
params = json.loads(params)

# Take nebula_tile_drift_speed value from params
nebula_tile_drift_speed = params['nebula_tile_drift_speed']
map_iteration_period = 1/np.absolute(nebula_tile_drift_speed)
grand_obstruction_matrix_size = 24 + (500 // int(map_iteration_period))
obstruction_direction = np.sign(nebula_tile_drift_speed)

obstruction_matrix = obstruction_and_relic_matrix.GrandObstructionMatrix(grand_obstruction_matrix_size)

#print(map[0][1][1])

for i in range(len(map[0][0])-1):
    for j in range(len(map[0][0])-1):
        obstruction_matrix.set_index_value(map_iteration_period, time_iteration=0, iteration_index=[i,j], discrete_values=map[0][i][j])

# Visualize the MAP and the generated obstruction matrix
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(map[0], cmap='viridis')
axes[0].set_title('Map at t=0, Seed = ' + str(random_seed))
axes[0].axis('off')
obstruction_matrix_data = obstruction_matrix.data
obstruction_matrix_data = np.array([[np.mean(cell) for cell in row] for row in obstruction_matrix_data], dtype=float)
axes[1].imshow(obstruction_matrix_data, cmap='viridis')
axes[1].set_title('Grand Obstruction Matrix')

plt.show()
'''

# Define function to create and save as many GOMs as needed
def create_and_save_grand_obstruction_matrix(num_matrices):
    file_location = 'C:/Users/ahmad/Desktop/lux/GOM_seed_dataset/'
    os.makedirs(file_location, exist_ok=True)

    # Find the highest existing GOM number
    existing_files = [f for f in os.listdir(file_location) if f.startswith('grand_obstruction_matrix_') and f.endswith('.npy')]
    existing_numbers = [int(re.search(r'(\d+)', f).group(0)) for f in existing_files if re.search(r'(\d+)', f)]

    start_index = max(existing_numbers) + 1 if existing_numbers else 0  # Start from next available index

    for m in range(start_index, start_index + num_matrices):
        random_seed = np.random.randint(0, 2**10)
        map, relic_nodes, params = gen_map_from_game.gen_game_map_time_start(random_seed)
        params = json.loads(params)
        nebula_tile_drift_speed = params['nebula_tile_drift_speed']
        map_iteration_period = 1/np.absolute(nebula_tile_drift_speed)
        grand_obstruction_matrix_size = 24 + (500 // int(map_iteration_period))
        obstruction_direction = np.sign(nebula_tile_drift_speed)
        obstruction_matrix = obstruction_and_relic_matrix.GrandObstructionMatrixTest(grand_obstruction_matrix_size)
        obstruction_matrix.set_obstruction_direction(obstruction_direction)
        
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                obstruction_matrix.set_index_value(map_iteration_period, time_iteration=0, iteration_index=[i, j], discrete_values=map[i, j])

        #obstruction_matrix_data = obstruction_matrix.data.astype(int)
        obstruction_matrix_data = obstruction_matrix.data

        # Save the obstruction matrix as a numpy file
        #print(obstruction_matrix_data)
        np.save(os.path.join(file_location, f'grand_obstruction_matrix_{m}.npy'), obstruction_matrix_data)
        
        del obstruction_matrix, obstruction_matrix_data, map, relic_nodes, params
        gc.collect()  # Run garbage collection to free up memory
        jax.clear_caches()  # Clear JAX backend caches
        
        if (m - start_index) % 50 == 0:
            print(f'{m - start_index} Grand Obstruction Matrices saved!')

    print(f'Completed saving {num_matrices} new Grand Obstruction Matrices starting from index {start_index}.')

# Check if there are already GOMs saved in the directory
# If there are, delete them
file_location = 'C:/Users/ahmad/Desktop/lux/GOM_seed_dataset/'
for f in os.listdir(file_location):
    if 'grand_obstruction_matrix' in f:
        os.remove(file_location + f)
print('All existing Grand Obstruction Matrices deleted!')

# Call the function to create and save #N GOMs
number_of_matrices = 500
create_and_save_grand_obstruction_matrix(number_of_matrices)
print('All ' + str(number_of_matrices) + ' Grand Obstruction Matrices saved successfully')