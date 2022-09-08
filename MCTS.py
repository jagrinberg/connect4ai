import math
import torch

class MCTS():
    def __init__(self, state, prior_probabilities, c):
        self.visit_count = 0
        self.total_action_values = None
        self.mean_action_values = None
        self.prior_probabilities = prior_probabilities
        self.state = state
        self.child_visit_counts = None
        self.children = []
        self.c = c
    
    def prob(self, temperature):
        a = torch.pow(self.child_visit_counts, temperature)
        return a/torch.sum(a)

    def explore(self, update):
        # increment visit count everytime we pass the node
        self.visit_count += 1

        # check to see if this is a leaf node
        return_flag = False
        if self.child_visit_counts == None:
            # if it is, populate its children and return the value of the chosen state
            self.populate(update)
            return_flag = True

        # perform an upper confidence bound to choose a node to explore
        ucb = self.mean_action_values + self.c * self.prior_probabilities * math.sqrt(self.visit_count) / (self.child_visit_counts + 1)
        ind = torch.argmax(ucb)

        # check to see if it was a leaf node
        if return_flag:
            # return the chosen child's value
            return -self.mean_action_values[ind]
        else:
            # explore the chosen child and add the eventual value to its own
            val = self.children[ind].explore(update)
            self.child_visit_counts[ind] += 1
            self.total_action_values[ind] += val
            self.mean_action_values[ind] = self.total_action_values[ind]/self.child_visit_counts[ind]
            return -val

    def populate(self, update):
        action_values, next_priors, next_states = update(self.state)
        self.total_action_values = -torch.clone(action_values)
        self.mean_action_values = -torch.clone(action_values)
        self.child_visit_counts = torch.zeros_like(action_values)
        for prior, state in zip(torch.split(next_priors, 1)):
            child = MCTS(state.squeeze(), prior.squeeze(), self.c)
            self.children.append(child)



