import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, discrete=True):
        self.mem_size = max_size
        self.mem_counter = 0
        # Input shape of the environment
        self.input_shape = input_shape
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        # One hot encoding
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal


class QNetwork(nn.Module):
    """
    Adopt Deep Set architecture to ensure permutation invariance among neighbours states and actions.
    Input shape:
    - Relative position of current satellite to destination satellite (3,)
    - For each neighbour:
        - Relative position of neighbour to destination satellite (3,)
        - Processing rate (1,)
        - Queue length (1,)
    """

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super().__init__()

    def forward(self, state):
        pass


class DeepSetNetwork(nn.Module):
    """
    Deep Set Network for processing satellite and neighbour information.
    Input to the Deep Set:
    - Current satellite relative position to destination (3,) (as contextual input)
    - Each neighbour is represented as a vector of shape (5,)
    Output:
    - Embedding vector of shape (fc2_dims,)
    """
    def __init__(self, input_dims, fc1_dims, fc2_dims):
