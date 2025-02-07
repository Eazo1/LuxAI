from . import genome
import pickle

def save_best_genome(genome:genome, filename="best_genome.pkl"):
    """Saves the given genome (best model) to a file."""
    with open(filename, "wb") as f:
        pickle.dump(genome, f)

def load_best_genome(filename="best_genome.pkl"):
    """Loads the genome (best model) from a file."""
    with open(filename, "rb") as f:
        genome = pickle.load(f)
    return genome