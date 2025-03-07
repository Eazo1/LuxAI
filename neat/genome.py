from . import gene
from . import neatconstants
import math
import random

# ============================
# Genome (Neural Network)
# ============================

def normalize(values):
    # if not values:
    #     return []
    
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return [0] * len(values)
    
    scale = max_val - min_val
    return [(x - min_val) / scale for x in values]

class Genome:
    def __init__(self, num_inputs, num_outputs, innovation_tracker, neurons=None, links=None):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.innovation_tracker = innovation_tracker
        self.neurons = neurons if neurons is not None else {}   # Dictionary: id -> NeuronGene
        self.links = links if links is not None else {}         # Dictionary: innovation -> LinkGene
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.max_neuron = 0  # tracks the highest neuron id used

    def copy(self):
        new_neurons = {nid: gene.NeuronGene(n.id, n.bias, n.layer, n.neuron_type)
                       for nid, n in self.neurons.items()}
        new_links = {}
        for innov, link in self.links.items():
            new_links[innov] = gene.LinkGene(link.in_id, link.out_id, link.weight, link.enabled, link.innovation)
        g = Genome(self.num_inputs, self.num_outputs, self.innovation_tracker, new_neurons, new_links)
        g.fitness = self.fitness
        g.max_neuron = self.max_neuron
        return g

    # ------------------------
    # Mutation Operations
    # ------------------------

    def mutate_weights(self):
        for link in self.links.values():
            if random.random() < neatconstants.WEIGHT_MUTATION_RATE:
                if random.random() < neatconstants.WEIGHT_PERTURBATION_RATE:
                    link.weight += random.uniform(-neatconstants.WEIGHT_PERTURBATION_STRENGTH, neatconstants.WEIGHT_PERTURBATION_STRENGTH)
                else:
                    link.weight = random.uniform(*neatconstants.NEW_WEIGHT_RANGE)
        for neuron in self.neurons.values():
            if random.random() < neatconstants.BIAS_MUTATION_RATE:
                neuron.bias += random.uniform(-neatconstants.BIAS_PERTURBATION_STRENGTH, neatconstants.BIAS_PERTURBATION_STRENGTH)

    def mutate_add_link(self):
        # Attempt to add a new feed-forward link.
        possible_pairs = []
        neuron_ids = list(self.neurons.keys())
        for a in neuron_ids:
            for b in neuron_ids:
                if a == b:
                    continue
                # Enforce feedforward: only allow if source layer < target layer.
                if self.neurons[a].layer >= self.neurons[b].layer:
                    continue
                # Check whether a link already exists between these neurons.
                exists = any(link.in_id == a and link.out_id == b for link in self.links.values())
                if exists:
                    continue
                possible_pairs.append((a, b))
        if not possible_pairs:
            return
        in_id, out_id = random.choice(possible_pairs)
        innov = self.innovation_tracker.get_innovation_number(in_id, out_id)
        new_link = gene.LinkGene(in_id, out_id, random.uniform(*neatconstants.NEW_WEIGHT_RANGE), True, innov)
        self.links[innov] = new_link

    def mutate_add_node(self):
        # Choose a random enabled link to split.
        enabled_links = [link for link in self.links.values() if link.enabled]
        if not enabled_links:
            return
        link_to_split = random.choice(enabled_links)
        link_to_split.enabled = False
        new_node_id = self.innovation_tracker.get_new_node_id()
        # Determine new node layer. If there is a gap, use an integer; otherwise, assign fractional layer.
        in_layer = self.neurons[link_to_split.in_id].layer
        out_layer = self.neurons[link_to_split.out_id].layer
        if out_layer - in_layer > 1:
            new_layer = in_layer + 1
        else:
            new_layer = (in_layer + out_layer) / 2.0
        new_neuron = gene.NeuronGene(new_node_id, bias=0.0, layer=new_layer, neuron_type="hidden")
        self.neurons[new_node_id] = new_neuron

        # Create new links: one from the original input to the new node, one from the new node to the original output.
        innov1 = self.innovation_tracker.get_innovation_number(link_to_split.in_id, new_node_id)
        link1 = gene.LinkGene(link_to_split.in_id, new_node_id, 1.0, True, innov1)
        self.links[innov1] = link1

        innov2 = self.innovation_tracker.get_innovation_number(new_node_id, link_to_split.out_id)
        link2 = gene.LinkGene(new_node_id, link_to_split.out_id, link_to_split.weight, True, innov2)
        self.links[innov2] = link2

    def mutate(self):
        self.mutate_weights()
        if random.random() < neatconstants.ADD_LINK_RATE:
            self.mutate_add_link()
        if random.random() < neatconstants.ADD_NODE_RATE:
            self.mutate_add_node()

    # ------------------------
    # Crossover Operation
    # ------------------------

    @staticmethod
    def crossover(parent1, parent2):
        # Assumes parent1 has equal or higher fitness than parent2.
        child = Genome(parent1.num_inputs, parent1.num_outputs, parent1.innovation_tracker)
        # Copy neurons from the more fit parent.
        for nid, neuron in parent1.neurons.items():
            child.neurons[nid] = gene.NeuronGene(neuron.id, neuron.bias, neuron.layer, neuron.neuron_type)
        # Align links by innovation number.
        for innov in parent1.links:
            if innov in parent2.links:
                l1 = parent1.links[innov]
                l2 = parent2.links[innov]
                chosen_link = random.choice([l1, l2])
                # If either gene is disabled, disable in child with 75% chance.
                if (not l1.enabled or not l2.enabled) and random.random() < 0.75:
                    chosen_link.enabled = False
                child.links[innov] = gene.LinkGene(chosen_link.in_id, chosen_link.out_id,
                                              chosen_link.weight, chosen_link.enabled,
                                              chosen_link.innovation)
            else:
                # Gene only in parent1 (the more fit parent).
                child.links[innov] = gene.LinkGene(parent1.links[innov].in_id,
                                              parent1.links[innov].out_id,
                                              parent1.links[innov].weight,
                                              parent1.links[innov].enabled,
                                              parent1.links[innov].innovation)
        return child

    # ------------------------
    # Network Evaluation (Feedforward)
    # ------------------------

    def evaluate(self, input_values):
        """
        Evaluates the network on the given input values.
        Assumes that the number of input_values equals the number of input neurons.
        """
        input_values = normalize(input_values)
        activations = {}
        # Set input neuron activations (sorted by neuron id for consistency).
        input_neurons = sorted([n for n in self.neurons.values() if n.neuron_type == "input"], key=lambda n: n.id)
        if len(input_values) != len(input_neurons):
            raise ValueError("Incorrect number of inputs provided.")
        for i, neuron in enumerate(input_neurons):
            activations[neuron.id] = input_values[i]
        # Process neurons in order of increasing layer.
        sorted_neurons = sorted(self.neurons.values(), key=lambda n: n.layer)
        for neuron in sorted_neurons:
            if neuron.id in activations:
                continue  # Skip input neurons (already set)
            total = neuron.bias
            for link in self.links.values():
                if link.enabled and link.out_id == neuron.id:
                    if link.in_id not in activations:
                        continue  # Should not happen if layers are correctly assigned.
                    total += activations[link.in_id] * link.weight
            activations[neuron.id] = neuron.activation(total)
        # Collect output neuron activations (sorted by id).
        output_neurons = sorted([n for n in self.neurons.values() if n.neuron_type == "output"], key=lambda n: n.id)
        outputs = [activations[n.id] for n in output_neurons]
        outputs = normalize(outputs)
        return outputs
