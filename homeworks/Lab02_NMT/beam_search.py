import numpy as np
import torch


class Node:
    def __init__(self, parent, state, value, cost):
        self.value = value
        self.parent = parent
        self.state = state
        self.cum_cost = parent.cum_cost + cost if parent else cost
        self.length = 1 if parent is None else parent.length + 1
        self._sequence = None

    def to_sequence(self):
        # Return sequence of nodes from root to current node.
        if not self._sequence:
            self._sequence = []
            current_node = self
            while current_node:
                self._sequence.insert(0, current_node)
                current_node = current_node.parent
        return self._sequence

    def to_sequence_of_values(self):
        return [int(s.value.cpu().numpy()) for s in self.to_sequence()]
