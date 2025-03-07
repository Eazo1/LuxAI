import random
import sys
import pickle
from pathlib import Path
import jax.numpy as jnp
from multiprocessing import Pool
from neat import gene, genome, specie, population, savegenome, neatconstants
from agent.agent import Agent
from obstruction_and_relic_matrix import GrandObstructionMatrixTest
import numpy as np
import torch
from GOM_autoencoder import Autoencoder
import wandb
import torch.nn.functional as F


# Import LuxAI classes from your game runner.
from luxai_s3.wrappers import LuxAIS3GymEnv
from luxai_runner.episode import EpisodeConfig


wandb.init(project="GOM_AE", mode="disabled")
# Define model parameters (must match those used during saving)
latent_dim = 10
num_hiddens = 256
num_residual_layers = 2
num_residual_hiddens = 32

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = Autoencoder(num_hiddens, num_residual_layers, num_residual_hiddens, latent_dim).to(device)
model_save_path = 'C:/Users/ahmad/Desktop/lux/GOM_autoencoder_model.pth'
autoencoder.load_state_dict(torch.load(model_save_path, map_location=device))
autoencoder.eval()
autoencoder.train(False)  
autoencoder.eval()  # Set the model to evaluation mode
for param in autoencoder.parameters():
    param.requires_grad = False  # Disable gradients to prevent training
wandb.finish()


def flatten_structure(structure):
    """Recursively flatten nested structures into a single 1D NumPy array."""
    flat_list = []
    
    if isinstance(structure, dict):
        for value in structure.values():
            flat_list.extend(flatten_structure(value))
    elif isinstance(structure, (list, tuple)):
        for item in structure:
            flat_list.extend(flatten_structure(item))
    elif isinstance(structure, np.ndarray):
        flat_list.extend(structure.ravel())  # Flatten arrays
    elif isinstance(structure, (int, float, bool)):
        flat_list.append(structure)
    else:
        raise TypeError(f"Unsupported type: {type(structure)}")
    
    return flat_list

def temperature_softmax(x, temperature=1.0):
    """Softmax-like probability distribution from a single output using temperature scaling."""
    x = np.clip(x, -10, 10)  # Clip for numerical stability
    logits = np.array([x + i for i in range(-3, 3)])  # Create 6 pseudo-logits from the single score
    exp_logits = np.exp((logits - np.max(logits)) / temperature)
    probabilities = exp_logits / np.sum(exp_logits)
    return probabilities

class NeatBot:
    def __init__(self, player: str, env_cfg, init_obs, genome) -> None:
        self.genome = genome
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.GOM = GrandObstructionMatrixTest(size=49, obs=init_obs)
        self.ally_score = 0
        self.enemy_score = 0

    def get_ally_score(self):
        return self.ally_score
    
    def get_enemy_score(self):
        return self.enemy_score
    
    
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Take the different information, craft a set of usable input parameters and find output
        Make actions based on the output of the genome
        """
        inputs = self.extract_features(step, obs)
        outputs = self.genome.evaluate(inputs)
        action = self.interpret_outputs(outputs)
        return action
    
    def extract_features(self, step, obs):
        # NEED TO CODE
        # Following is DUMMY CODE
        # Current problem, I am not sure where I am pulling the observation dictionary from, but I will
        # just put a placeholder name

        #updating the status of each user is a bit problematic because of Eazo's code, I need to add more
        #functions to it first

        unit_information = np.concatenate((obs["units"]["position"], np.expand_dims(obs["units"]["energy"], axis=-1)), axis=-1)
        unit_information = unit_information.flatten() # Unit information extracted and flattend for NN

        #updating GOM
        time_iteration = obs["steps"]
        map_iteration_period = 6.67

        ally_index = 0
        enemy_index = 0

        if self.player == "player_0":
            ally_index = 0
            enemy_index = 1
        else:
            ally_index = 1
            enemy_index = 0

        for i in range(24):
            for j in range(24):
                iteration_index = (i,j)
                discrete_values = np.array([obs["map_features"]["tile_type"][i][j]])
                iteration_i, iteration_j = self.GOM.get_entangled_iteration_index(i, j)
                current_cell_state = self.GOM.get_index_value((iteration_i, iteration_j))
                if current_cell_state != -1:
                    discrete_values[0] = current_cell_state
                # if current_cell_state[1] != -1:
                #     discrete_values[1] = current_cell_state[1]

                self.GOM.set_index_value(map_iteration_period, time_iteration, iteration_index, discrete_values)

        # At this point, the GOM has been updated I need the AE to transform it into better inputs
        # Then I would flatten this input (just keeping spatial data) and combaine this with the unit information

        # Get the GOM data as a NumPy array
        gom_data = self.GOM.get_data()  # Get the current GOM matrix

        # Convert GOM data to a format suitable for the encoder (HWC -> CHW format)
        gom_data = np.expand_dims(gom_data, axis=0)  # Add batch dimension
        gom_tensor = torch.tensor(gom_data, dtype=torch.float32)
        # gom_tensor = gom_tensor + 1
        # gom_tensor = F.one_hot(gom_tensor, num_classes=3).permute(2, 0, 1).unsqueeze(0).float().to(device)
        # gom_tensor = F.one_hot(gom_tensor, num_classes=4).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
        # print(gom_tensor.shape)
        # gom_tensor = F.one_hot(torch.tensor(gom_data, dtype=torch.long), num_classes=4).permute(2,0,1).float()  # Now (4,49,49)
        #gom_tensor = F.one_hot(torch.from_numpy(gom_data).long(), num_classes=3).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

        # Run the GOM data through the Encoder directly to extract latent features
        with torch.no_grad():
            #gom_tensor = torch.from_numpy(gom_data).to(device)  # Convert NumPy array to PyTorch tensor
            latent_features = autoencoder.encoder(gom_tensor).cpu().numpy().flatten()  # Extract and flatten latent features
        flattened_GOM = np.array(latent_features).flatten()
        points_gained = np.array([obs["team_points"][ally_index]-self.ally_score, obs["team_points"][enemy_index]-self.enemy_score]) #
        relic_info = np.array(obs["relic_nodes"]).flatten()
        final_inputs = np.concatenate([unit_information, flattened_GOM, relic_info, points_gained]) # All these objects need to be a 1D arrays
        # print(final_inputs.size)
        # Final number of inputs = 48 + 12 + 2 + 360  
        self.ally_score = obs["team_points"][ally_index]
        self.enemy_score = obs["team_points"][enemy_index] # The code for team points relies on the assumption that ally score is always first in the list

        return final_inputs

    def interpret_outputs(self, outputs):
        '''Takes the outputs of genome and applies softmax to decide moves'''
        num_units = 16  # Assume max 16 units to control
        num_actions = 6  # Each unit can take 6 possible actions

        # Convert neural network outputs into a discrete action for each unit
        actions = np.zeros((num_units, 3), dtype=int)

        for i, output in enumerate(outputs):
            action_probs = temperature_softmax(output, temperature=0.8)
            chosen_action = np.random.choice(num_actions, p=action_probs)  # Sample based on probability
            actions[i] = [chosen_action, 0, 0]

        # Convert to JAX-compatible format
        action_array = jnp.array(actions)     

        return action_array

# ------------------------------
# Evaluation Function
# ------------------------------

def evaluate_genome(genome):
    """
    Runs one episode in the LuxAI environment using the genome-controlled NEATBot.
    Returns the total reward for the episode.
    """
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset()
    env_cfg = info["params"]
    random_number = random.randint(0,1)
    ally_index = 0
    enemy_index = 0
    if random_number == 0:
        ally_index = 0
        enemy_index = 1
        player_0 = NeatBot("player_0",env_cfg, obs["player_0"], genome=genome)
        player_1 = Agent("player_1", env_cfg)
    else:
        enemy_index = 0
        ally_index = 1
        player_1 = NeatBot("player_1",env_cfg, obs["player_1"], genome=genome)
        player_0 = Agent("player_0", env_cfg)

    # Initialise agents
    # player_0 = NeatBot("player_0",env_cfg, obs, genome=genome)
    # player_1 = Agent("player_1", env_cfg)

    done = False
    total_fitness = 0.0
    #total_fitness2 = 0.0
    step = 0

    ally_tally = 0
    enemy_tally = 0
    while not done:
        actions = {}
        for agent in [player_0, player_1]:
            actions[agent.player] = agent.act(step=step, obs=obs[agent.player])

        obs, reward ,terminated, truncated, info = env.step(actions)
        doneDict = {k: terminated[k] | truncated[k] for k in terminated}
        step += 1
        if step // 20 == 0:
            if (ally_index == 0):
                total_fitness += (player_0.get_ally_score()-ally_tally)-(player_0.get_enemy_score()-enemy_tally)

                ally_tally = player_0.get_ally_score()
                enemy_tally = player_0.get_enemy_score()
            else:
                total_fitness += (player_1.get_ally_score()-ally_tally)-(player_1.get_enemy_score()-enemy_tally)

                ally_tally = player_1.get_ally_score()
                enemy_tally = player_1.get_enemy_score()
                
        done = (doneDict["player_0"] and doneDict["player_1"])
        if ally_index == 0:
            if obs["player_0"]["team_wins"][ally_index] == 3:
                total_fitness += 10
            if obs["player_0"]["team_wins"][enemy_index] == 3:
                total_fitness -= 10
        else:
            if obs["player_1"]["team_wins"][ally_index] == 3:
                total_fitness += 10
            if obs["player_1"]["team_wins"][enemy_index] == 3:
                total_fitness -= 10
    genome.fitness = total_fitness
    return total_fitness

def evaluate_population(population):
    with Pool() as pool:
        fitness_scores = pool.map(evaluate_genome, population.genomes)
    
    # Assign fitness values correctly
    for i in range(len(population.genomes)):
        population.genomes[i].fitness = fitness_scores[i]

# ------------------------------
# Training Loop
# ------------------------------

def main():
    num_inputs = 470 # SUBSTITUTE WITH ACTUAL NUMBER OF INPUTS
    num_outputs =  16 # Number of outputs
    population_size = 2
    generations = 1
    percent_saved = 0.6

    pop = population.Population(size=population_size, num_inputs=num_inputs, num_outputs=num_outputs)

    best_genome = None
    best_fitness = -float('inf')

    for gen in range(generations):
        print(f"--- Generation {gen} ---")
        evaluate_population(pop)
        current_best = max(pop.genomes, key=lambda g: g.fitness)
        print(f"Generation {gen} best fitness: {current_best.fitness}")
        if current_best.fitness > best_fitness:
            best_fitness = current_best.fitness
            best_genome = current_best.copy()
        pop.evolve()

    print("Training complete. Best fitness:", best_fitness)
    savegenome.save_top_genomes(pop.genomes, percent_saved)

if __name__ == "__main__":
    main()


