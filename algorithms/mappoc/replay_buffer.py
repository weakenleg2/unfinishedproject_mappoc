import random
import numpy as np
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def store(self, obs, action, reward, next_obs, done, vpred,ac_log):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, action, reward, next_obs, done, vpred,ac_log)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done, vpred, ac_log = map(np.stack, zip(*batch))
        return obs, action, reward, next_obs, done, vpred, ac_log

    def clear(self):
        self.buffer.clear()
        self.position = 0

    def __len__(self):
        return len(self.buffer)
