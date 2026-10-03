"""V12: one-step Double DQN + PER across independent vector environments.

Each tree leaf identifies one (time slot, environment) transition. Raw rewards
stay in replay; the current VecNormalize statistics are applied when sampling.
"""
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from prioritized_double_dqn import PriorityTree, PrioritizedReplayBuffer, PrioritizedDoubleDQN


class VectorPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, *args, alpha=0.6, priority_epsilon=1e-6, **kwargs):
        ReplayBuffer.__init__(self, *args, **kwargs)
        if self.optimize_memory_usage:
            raise ValueError('Vector PER requires optimize_memory_usage=False')
        if not 0 <= alpha <= 1 or not np.isfinite(priority_epsilon) or priority_epsilon <= 0:
            raise ValueError('Invalid PER alpha or epsilon')
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.max_priority = 1.0
        self.tree = PriorityTree(self.buffer_size * self.n_envs)

    def reset(self):
        ReplayBuffer.reset(self)
        self.tree = PriorityTree(self.buffer_size * self.n_envs)
        self.max_priority = 1.0

    def add(self, *args, **kwargs):
        indices = self.pos * self.n_envs + np.arange(self.n_envs)
        ReplayBuffer.add(self, *args, **kwargs)
        self.tree.update(indices, np.full(self.n_envs, self.max_priority ** self.alpha))

    def _get_samples(self, batch_inds, env=None):
        rows, envs = np.divmod(np.asarray(batch_inds, dtype=np.int64), self.n_envs)
        data = (
            self._normalize_obs(self.observations[rows, envs], env),
            self.actions[rows, envs],
            self._normalize_obs(self.next_observations[rows, envs], env),
            (self.dones[rows, envs] * (1 - self.timeouts[rows, envs])).reshape(-1, 1),
            self._normalize_reward(self.rewards[rows, envs].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def update_priorities(self, indices, td_errors):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        errors = np.asarray(td_errors, dtype=np.float64).reshape(-1)
        if (len(indices) == 0 or len(indices) != len(errors)
                or not np.isfinite(errors).all() or np.any(indices < 0)
                or np.any(indices >= self.size() * self.n_envs)):
            raise ValueError('Invalid vector PER priority update')
        priorities = np.abs(errors) + self.priority_epsilon
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self.tree.update(indices, priorities ** self.alpha)


class VectorPrioritizedDoubleDQN(PrioritizedDoubleDQN):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('replay_buffer_class', VectorPrioritizedReplayBuffer)
        super().__init__(*args, **kwargs)
