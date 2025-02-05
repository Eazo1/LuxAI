# Hyperparameters

POPULATION_SIZE = 150       # Population size (adjust for your problem)
CROSSOVER_RATE = 0.75       # Probability of crossover rather than cloning
WEIGHT_MUTATION_RATE = 0.8
WEIGHT_PERTURBATION_RATE = 0.9
WEIGHT_PERTURBATION_STRENGTH = 0.5
NEW_WEIGHT_RANGE = (-1.0, 1.0)
BIAS_MUTATION_RATE = 0.1
BIAS_PERTURBATION_STRENGTH = 0.5
ADD_LINK_RATE = 0.05
ADD_NODE_RATE = 0.03

# Speciation parameters (these coefficients are as in the original NEAT paper)
C1 = 1.0      # Coefficient for excess genes
C2 = 1.0      # Coefficient for disjoint genes
C3 = 0.4      # Coefficient for average weight difference
COMPATIBILITY_THRESHOLD = 3.0
TARGET_SPECIES = 10
DELTA_THRESHOLD = 0.3

# Staleness: how many generations a species is allowed to go without improvement
STALE_SPECIES = 15