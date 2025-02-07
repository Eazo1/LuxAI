import math
import random
from . import neatconstants

# ===================================
# Gene Representations (Nodes/Links)
# ===================================

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class NeuronGene:
    def __init__(self, id, bias=0.0, layer=0.0, neuron_type="hidden"):
        """
        id: Unique identifier.
        bias: Bias term.
        layer: A number representing its “depth” (inputs at 0, outputs at highest).
        neuron_type: "input", "output", or "hidden".
        """
        self.id = id
        self.bias = bias
        self.layer = layer
        self.neuron_type = neuron_type
        self.activation = sigmoid

    @staticmethod
    def crossover(neuron1, neuron2):
        # Assume the neurons represent the same node.
        chosen_bias = random.choice([neuron1.bias, neuron2.bias])
        return NeuronGene(neuron1.id, chosen_bias, neuron1.layer, neuron1.neuron_type)


class LinkGene:
    def __init__(self, in_id, out_id, weight, enabled=True, innovation=0):
        """
        in_id: ID of the source neuron.
        out_id: ID of the target neuron.
        weight: Connection weight.
        enabled: Whether the link is active.
        innovation: Unique innovation number.
        """
        self.in_id = in_id
        self.out_id = out_id
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

    @staticmethod
    def crossover(link1, link2):
        # Assumes both links represent the same innovation.
        weight = random.choice([link1.weight, link2.weight])
        # If either gene is disabled, disable the gene in the child with 75% chance.
        enabled = True
        if (not link1.enabled or not link2.enabled) and random.random() < 0.75:
            enabled = False
        return LinkGene(link1.in_id, link1.out_id, weight, enabled, link1.innovation)


# =====================================
# Innovation Tracker for Nodes & Links
# =====================================

class InnovationTracker:
    def __init__(self, starting_node_id):
        self.next_node_id = starting_node_id
        self.next_innovation = 0
        self.connection_history = {}  # Maps (in_id, out_id) -> innovation number

    def get_new_node_id(self):
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

    def get_innovation_number(self, in_id, out_id):
        key = (in_id, out_id)
        if key in self.connection_history:
            return self.connection_history[key]
        else:
            self.connection_history[key] = self.next_innovation
            self.next_innovation += 1
            return self.connection_history[key]
