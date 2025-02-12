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

        #Agent(player, configurations["env_cfg"])
        # Ensure action dict includes both players
        

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
    #player_0 = Agent("player_0", env_cfg)
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
    
      print(f'Step: {step}, Reward0: {total_fitness},Reward1: {total_fitness2}')
      if (step % 20):
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


