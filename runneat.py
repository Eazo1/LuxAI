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

# Import LuxAI classes from your game runner.
from luxai_s3.wrappers import LuxAIS3GymEnv
from luxai_runner.episode import EpisodeConfig

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
        self.GOM = GrandObstructionMatrixTest(size=24, obs=init_obs)
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

        unit_information = np.concatenate((obs["units"]["position"], obs["units"]["energy"]), axis=2)
        unit_information = np.flatten(unit_information) # Unit information extracted and flattend for NN

        #updating GOM
        time_iteration = obs["steps"]
        map_iteration_period = 6.67
        for i in range(24):
            for j in range(24):
                iteration_index = (i,j)
                discrete_values = np.array([obs["map_features"]["tile_type"][i][j], obs["map_features"]["energy"][i][j]],
                                            obs["sensor_mask"])
                iteration_i, iteration_j = self.GOM.get_entangled_iteration_index(i, j)
                current_cell_state = self.GOM.get_index_value((iteration_i, iteration_j))
                if current_cell_state[0] != -1:
                    discrete_values[0] = current_cell_state[0]
                if current_cell_state[1] != -1:
                    discrete_values[1] = current_cell_state[1]

                self.GOM.set_index_values(map_iteration_period, time_iteration, iteration_index, discrete_values)

        # At this point, the GOM has been updated I need the AE to transform it into better inputs
        # Then I would flatten this input (just keeping spatial data) and combaine this with the unit information

        flattened_GOM = np.flatten(self.GOM.get_data)
        points_gained = np.array([obs["team_points"][0]-self.ally_score, obs["team_points"][1]-self.enemy_score]) #
        relic_info = np.flatten(obs["relic_nodes"])
        final_inputs = np.concatente(unit_information, flattened_GOM, relic_info, points_gained) # All these objects need to be a 1D arrays

        # Final number of inputs = 48 + 12 + 2 + flattened GOM  
        self.ally_score = obs["team_points"][0]
        self.ally_score = obs["team_points"][1] # The code for team points relies on the assumption that ally score is always first in the list

        return flatten_structure(list(obs.values()))

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

    # Initialise agents
    player_0 = NeatBot("player_0",env_cfg, obs, genome=genome)
    player_1 = Agent("player_1", env_cfg)

    done = False
    total_fitness = 0.0
    total_fitness2 = 0.0
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
        if step // 30 == 0:
            total_fitness += player_0.get_ally_score()-ally_tally-player_0.get_ally_score()-enemy_tally

            ally_tally = player_0.get_ally_score()
            enemy_tally = player_0.get_enemy_score()
        done = (doneDict["player_0"] and doneDict["player_1"])
        total_fitness += reward['player_0'] #bot is player 0
        total_fitness2 += reward['player_1'] #agent is player 1
        
   
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
    num_inputs = 1880 # SUBSTITUTE WITH ACTUAL NUMBER OF INPUTS
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


