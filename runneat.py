import random
import sys
import pickle
from pathlib import Path
import jax.numpy as jnp
from multiprocessing import Pool
from neat import gene, genome, specie, population, savegenome, neatconstants

import numpy as np

# Import LuxAI classes from your game runner.
from luxai_s3.wrappers import LuxAIS3GymEnv
from luxai_runner.episode import EpisodeConfig

class NeatBot:
    def __init__(self, genome):
        self.genome = genome
    
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
        return np.random.randn(20)

    def interpret_outputs(self, outputs):
        # NEED TO CODE
        # Following is DUMMY CODE
        
        num_units = 16  # Assume max 16 units to control
        num_actions = 3  # Each unit can take 3 possible actions

        # Convert neural network outputs into a discrete action for each unit
        actions = np.zeros((num_units, num_actions), dtype=int)

        for i, output in enumerate(outputs[:num_units]):  # Ensure we don't exceed the number of units
            if output < 0.33:
                actions[i] = [1, 0, 0]  # Example: Move Left
            elif output < 0.66:
                actions[i] = [0, 1, 0]  # Example: Move Right
            else:
                actions[i] = [0, 0, 1]  # Example: Collect Resource

        # Convert to JAX-compatible format
        action_array = jnp.array(actions)
        
        # Ensure action dict includes both players
        action_dict = {
            "player_0": action_array,  # Controlled by NEAT agent
            "player_1": jnp.zeros_like(action_array)  # Assume player_1 takes no action (optional)
        }

        return action_dict

# ------------------------------
# Evaluation Function
# ------------------------------

def evaluate_genome(genome):
    """
    Runs one episode in the LuxAI environment using the genome-controlled NEATBot.
    Returns the total reward for the episode.
    """
    env = LuxAIS3GymEnv(numpy_output=True)
    obs = env.reset()
    bot = NeatBot(genome)
    total_fitness = 0.0
    done = False
    step = 0
    # Run episode till completion
    while not done:
        action = bot.act(step, obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        # MAKE FITNESS FUNCTION BETTER
        if (step % 20):
            total_fitness += reward
        done = terminated or truncated
        print(step)
        step += 1

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
    num_inputs = 20 # SUBSTITUTE WITH ACTUAL NUMBER OF INPUTS
    num_outputs = 3 # SUBSTITUE WITH ACTUAL NUMBER OF OUTPUTS
    population_size = 25
    generations = 2

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
    savegenome.save_best_genome(best_genome)

if __name__ == "__main__":
    main()


