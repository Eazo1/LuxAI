import random
import sys
import pickle
from pathlib import Path
import jax.numpy as jnp
from multiprocessing import Pool
from neat import gene, genome, specie, population, savegenome, neatconstants
from agent.agent import Agent

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
    def __init__(self, player: str, env_cfg,genome) -> None:
        self.genome = genome
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
    
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
        return flatten_structure(list(obs.values()))

    def interpret_outputs(self, outputs):
        '''Takes the outputs of genome and applies softmax to decide moves'''
        num_units = 16  # Assume max 16 units to control
        num_actions = 6  # Each unit can take 3 possible actions

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
    player_0 = NeatBot("player_0",env_cfg,genome=genome)
    player_1 = Agent("player_1", env_cfg)

    done = False
    total_fitness = 0.0
    total_fitness2 = 0.0
    step = 0
    while not done:
        actions = {}
        for agent in [player_0, player_1]:
            actions[agent.player] = agent.act(step=step, obs=obs[agent.player])

        obs, reward ,terminated, truncated, info = env.step(actions)
        doneDict = {k: terminated[k] | truncated[k] for k in terminated}
        step += 1
        done = (doneDict["player_0"] and doneDict["player_1"])
        if (step == 505):
            print(f'Step: {step}, Reward0: {total_fitness},Reward1: {total_fitness2}')
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


