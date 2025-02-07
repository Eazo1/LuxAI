import random
import sys
import pickle
from pathlib import Path
from neat import population, savegenome

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
        pass

    def interpret_outputs(self, outputs):
        # NEED TO CODE
        pass

# ------------------------------
# Evaluation Function
# ------------------------------

def evaluate_genome(genome, episode_cfg: EpisodeConfig):
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
        obs, reward, done, _ = env.step(action)
        # MAKE FITNESS FUNCTION BETTER
        total_fitness += reward
        step += 1
    
    return total_fitness

def evaluate_population(population):
    for genome in population.genomes:
        genome.fitness = evaluate_genome(genome)

# ------------------------------
# Training Loop
# ------------------------------

def main():
    num_inputs = 20 # SUBSTITUTE WITH ACTUAL NUMBER OF INPUTS
    num_outputs = 3 # SUBSTITUE WITH ACTUAL NUMBER OF OUTPUTS
    population_size = 50
    generations = 50

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


