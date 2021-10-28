import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.buffer_expert = []
        self.position = 0
        self.position_expert = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_expert(self, state, action):
        if len(self.buffer_expert) < self.capacity:
            self.buffer_expert.append(None)
        self.buffer_expert[self.position_expert] = (state, action)
        self.position_expert = (self.position_expert + 1) % self.capacity

    def sample(self, batch_size, use_expert=False):
        if not use_expert:
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done
        else:
            if len(self.buffer) != len(self.buffer_expert):
                raise ValueError("Mismatch size of buffers {} not equal to {}".format(len(self.buffer), len(self.buffer_expert)))
            batch, batch_expert = zip(*random.sample(list(zip(self.buffer, self.buffer_expert)), batch_size))
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            state_e, action_e = map(np.stack, zip(*batch_expert))
            return state, action, reward, next_state, done,\
                   state_e, action_e

    def __len__(self):
        return len(self.buffer)

    def expert_len(self):
        return len(self.buffer_expert)
