import numpy as np
import matplotlib.pyplot as plt
import json

import obstruction_and_relic_matrix
import gen_map_from_game

#map_iteration_period = 20 # Can be other values (must determine in game)
#grand_obstruction_matrix_size = 24 + (500 // map_iteration_period) # check if // is correct
#obstruction_direction = 1 # Range {-1,1}, with 1 meaning the little matrix moves northeast, and -1 meaning it moves southwest

random_seed = np.random.randint(0, 2**10)
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


