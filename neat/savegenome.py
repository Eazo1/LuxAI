from . import genome
import pickle

def save_best_genome(genome:genome, filename="best_genome.pkl"):
    """Saves the given genome (best model) to a file."""
    with open(filename, "wb") as f:
        pickle.dump(genome, f)

def save_top_genomes(genomes, percent_saved, filename_prefix="best_genome"):
    """Saves the top percentage of genomes to files."""
    num_to_save = max(1, int(len(genomes) * percent_saved))
    sorted_genomes = sorted(genomes, key=lambda g: g.fitness, reverse=True)
    for i in range(num_to_save):
        filename = f"{filename_prefix}_{i+1}.pkl"
        save_best_genome(sorted_genomes[i], filename=filename)
    print(f"Saved top {num_to_save} genomes.")

def load_best_genome(filename="best_genome.pkl"):
    """Loads the genome (best model) from a file."""
    with open(filename, "rb") as f:
        genome = pickle.load(f)
    return genome