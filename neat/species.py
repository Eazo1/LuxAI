import neatconstants
import genome

# =====================================
# Compatibility Distance for Speciation
# =====================================

def compatibility_distance(genome1: genome.Genome, genome2: genome.Genome):
    """
    Computes the compatibility distance between two genomes.
    Genes (links) are aligned by their innovation numbers.
    """
    genes1 = genome1.links
    genes2 = genome2.links

    innov_numbers1 = set(genes1.keys())
    innov_numbers2 = set(genes2.keys())
    max_innov1 = max(innov_numbers1) if innov_numbers1 else 0
    max_innov2 = max(innov_numbers2) if innov_numbers2 else 0
    max_innov = max(max_innov1, max_innov2)

    matching = []
    disjoint = 0
    excess = 0

    # Iterate over innovation numbers from 0 to max_innov.
    for innov in range(max_innov + 1):
        gene1 = genes1.get(innov, None)
        gene2 = genes2.get(innov, None)
        if gene1 is not None and gene2 is not None:
            matching.append((gene1, gene2))
        elif gene1 is not None or gene2 is not None:
            # Determine whether the gene is disjoint or excess.
            if innov > max_innov1 or innov > max_innov2:
                excess += 1
            else:
                disjoint += 1

    # Normalize by N (set to 1 if the genome is small).
    N = max(len(genes1), len(genes2))
    if N < 20:
        N = 1
    weight_diff = sum(abs(g1.weight - g2.weight) for g1, g2 in matching)
    avg_weight_diff = weight_diff / len(matching) if matching else 0.0

    distance = (neatconstants.C1 * excess / N) + (neatconstants.C2 * disjoint / N) + (neatconstants.C3 * avg_weight_diff)
    return distance


# =====================
# Species
# =====================

class Species:
    def __init__(self, representative):
        self.representative = representative.copy()
        self.genomes = []
        self.best_fitness = -float("inf")
        self.staleness = 0
        self.average_fitness = 0.0

    def add_genome(self, genome: genome.Genome):
        self.genomes.append(genome)

    def calculate_average_fitness(self):
        total = sum(g.fitness for g in self.genomes)
        self.average_fitness = total / len(self.genomes) if self.genomes else 0

    def sort_genomes(self):
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)
        if self.genomes[0].fitness > self.best_fitness:
            self.best_fitness = self.genomes[0].fitness
            self.staleness = 0
            self.representative = self.genomes[0].copy()
        else:
            self.staleness += 1