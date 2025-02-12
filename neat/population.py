import random
import math
import pickle
from . import neatconstants
from . import gene
from . import genome
from . import specie

# =====================
# Population
# =====================

class Population:
    def __init__(self, size, num_inputs, num_outputs, genome_files = None):
        self.size = size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.innovation_tracker = gene.InnovationTracker(starting_node_id=num_inputs + num_outputs)
        self.genomes = []
        self.species = []
        self.generation = 0
        self.compatibility_threshold = neatconstants.COMPATIBILITY_THRESHOLD

        # Load genomes from files if provided
        loaded_genomes = []
        if genome_files:
            for file in genome_files:
                try:
                    with open(file, "rb") as f:
                        loaded_genomes.append(pickle.load(f))
                    print(f"Loaded genome from {file}.")
                except Exception as e:
                    print(f"Error loading genome from {file}: {e}")

        # Initialize population with loaded genomes and random genomes
        num_loaded = min(len(loaded_genomes), size)
        for i in range(num_loaded):
            self.genomes.append(loaded_genomes[i].copy())
        
        # Initialize population with minimal networks.
        for _ in range(num_loaded, size):
            gnm = genome.Genome(num_inputs, num_outputs, self.innovation_tracker)
            # Create input neurons.
            for i in range(num_inputs):
                gnm.neurons[i] = gene.NeuronGene(i, bias=0.0, layer=0.0, neuron_type="input")
            # Create output neurons.
            for i in range(num_inputs, num_inputs + num_outputs):
                gnm.neurons[i] = gene.NeuronGene(i, bias=0.0, layer=1.0, neuron_type="output")
            # Fully connect inputs to outputs.
            for i in range(num_inputs):
                for j in range(num_inputs, num_inputs + num_outputs):
                    innov = self.innovation_tracker.get_innovation_number(i, j)
                    gnm.links[innov] = gene.LinkGene(i, j, random.uniform(*neatconstants.NEW_WEIGHT_RANGE), True, innov)
            gnm.max_neuron = num_inputs + num_outputs - 1
            self.genomes.append(gnm)

    def speciate(self):
        self.species = []
        for genome in self.genomes:
            found_species = False
            for species in self.species:
                if specie.compatibility_distance(genome, species.representative) < self.compatibility_threshold:
                    species.add_genome(genome)
                    found_species = True
                    break
            if not found_species:
                new_species = specie.Species(genome)
                new_species.add_genome(genome)
                self.species.append(new_species)

    def remove_stale_species(self):
        survived = []
        for species in self.species:
            species.sort_genomes()
            if species.staleness < neatconstants.STALE_SPECIES or species.best_fitness >= self.get_top_fitness():
                survived.append(species)
        self.species = survived

    def get_top_fitness(self):
        return max(genome.fitness for genome in self.genomes) if self.genomes else 0

    def adjust_compatibility_threshold(self):
        # Dynamically adjust to keep number of species near TARGET_SPECIES.
        if len(self.species) < neatconstants.TARGET_SPECIES:
            self.compatibility_threshold -= neatconstants.DELTA_THRESHOLD
        elif len(self.species) > neatconstants.TARGET_SPECIES:
            self.compatibility_threshold += neatconstants.DELTA_THRESHOLD
        if self.compatibility_threshold < 0.5:
            self.compatibility_threshold = 0.5

    def total_average_fitness(self):
        total = 0.0
        for species in self.species:
            species.calculate_average_fitness()
            total += species.average_fitness
        return total

    def breed(self, species):
        fitness_values = [genome.fitness for genome in species.genomes]
        min_fitness = min(fitness_values)
        adjusted_fitness = [(f - min_fitness + 1) for f in fitness_values]  # Shift to ensure non-negative values
        total_fitness = sum(adjusted_fitness)

        if total_fitness == 0:
            parent1, parent2 = random.sample(species.genomes, 2)
        else:
            parent1 = random.choices(species.genomes, weights=adjusted_fitness, k=1)[0]
            parent2 = random.choices(species.genomes, weights=adjusted_fitness, k=1)[0]
            if parent1.fitness < parent2.fitness:
                parent1, parent2 = parent2, parent1
        
        child = genome.Genome.crossover(parent1, parent2) if random.random() < neatconstants.CROSSOVER_RATE else random.choice(species.genomes).copy()
        child.mutate()
        return child

    def reproduce(self):
        new_genomes = []
        # Elitism: keep the best genome (champion) from each species.
        for species in self.species:
            species.sort_genomes()
            new_genomes.append(species.genomes[0].copy())
        total_average = self.total_average_fitness()
        for species in self.species:
            if total_average == 0:
                share = 0
            else:
                share = species.average_fitness / total_average
            # Allocate offspring to species proportionally.
            offspring = int(round(share * self.size)) - 1
            for _ in range(offspring):
                new_genomes.append(self.breed(species))
        # Fill in any remaining genomes randomly.
        while len(new_genomes) < self.size:
            species = random.choice(self.species)
            new_genomes.append(self.breed(species))
        self.genomes = new_genomes
        self.generation += 1

    def evolve(self):
        # Evaluate fitness of all genomes.
        self.speciate()
        self.adjust_compatibility_threshold()
        self.remove_stale_species()
        self.reproduce()