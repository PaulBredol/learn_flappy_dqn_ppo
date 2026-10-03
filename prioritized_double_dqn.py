"""Proportional PER for single-environment, one-step Double DQN (SB3 2.9).

Sampling changes; FIFO replacement and timeout handling remain unchanged.
Reference: https://arxiv.org/abs/1511.05952
"""
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from double_dqn import DoubleDQN


class PriorityTree:
    """Sum/min segment trees with O(log capacity) updates and draws."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.leaves = 1 << (capacity - 1).bit_length()
        self.sums = np.zeros(2 * self.leaves, dtype=np.float64)
        self.mins = np.full(2 * self.leaves, np.inf, dtype=np.float64)

    def update(self, indices, values):
        # Duplicates may occur in a minibatch; keep the largest priority.
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        unique, inverse = np.unique(indices, return_inverse=True)
        merged = np.zeros(len(unique), dtype=np.float64)
        np.maximum.at(merged, inverse, values)
        nodes = unique + self.leaves
        self.sums[nodes] = merged
        self.mins[nodes] = merged
        while nodes[0] > 1:
            nodes = np.unique(nodes // 2)
            self.sums[nodes] = self.sums[2 * nodes] + self.sums[2 * nodes + 1]
            self.mins[nodes] = np.minimum(self.mins[2 * nodes], self.mins[2 * nodes + 1])

    def draw(self, masses):
        masses = np.asarray(masses, dtype=np.float64).copy()
        nodes = np.ones(len(masses), dtype=np.int64)
        while nodes[0] < self.leaves:
            left = nodes * 2
            go_right = masses >= self.sums[left]
            masses -= np.where(go_right, self.sums[left], 0.0)
            nodes = left + go_right
        return nodes - self.leaves


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, *args, alpha=0.6, priority_epsilon=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1 or self.optimize_memory_usage:
            raise ValueError("PER requires n_envs=1 and optimize_memory_usage=False")
        if not 0 <= alpha <= 1 or not np.isfinite(priority_epsilon) or priority_epsilon <= 0:
            raise ValueError("Invalid PER alpha or epsilon")
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.max_priority = 1.0
        self.tree = PriorityTree(self.buffer_size)

    def reset(self):
        super().reset()
        self.tree = PriorityTree(self.buffer_size)
        self.max_priority = 1.0

    def add(self, *args, **kwargs):
        index = self.pos
        super().add(*args, **kwargs)
        self.tree.update([index], [self.max_priority ** self.alpha])

    def sample_with_info(self, batch_size, beta, env=None):
        if self.size() == 0 or batch_size <= 0 or not 0 <= beta <= 1:
            raise ValueError("PER needs a nonempty buffer, positive batch and beta in [0,1]")
        total = self.tree.sums[1]
        # Stratified draws, with replacement. Empty/padding leaves have zero mass.
        masses = (np.arange(batch_size) + np.random.random(batch_size)) * (total / batch_size)
        masses = np.minimum(masses, np.nextafter(total, 0.0))
        indices = self.tree.draw(masses)
        priorities = self.tree.sums[indices + self.tree.leaves]
        # (N P(i))^-beta normalized by the maximum weight across the whole buffer.
        weights = (priorities / self.tree.mins[1]) ** (-beta)
        return self._get_samples(indices, env=env), indices, self.to_torch(weights.astype(np.float32).reshape(-1, 1))

    def sample(self, batch_size, env=None):
        return self.sample_with_info(batch_size, beta=1.0, env=env)[0]

    def update_priorities(self, indices, td_errors):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        errors = np.asarray(td_errors, dtype=np.float64).reshape(-1)
        if (len(indices) == 0 or len(indices) != len(errors)
                or not np.isfinite(errors).all()
                or np.any(indices < 0) or np.any(indices >= self.size())):
            raise ValueError("Invalid PER priority update")
        priorities = np.abs(errors) + self.priority_epsilon
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self.tree.update(indices, priorities ** self.alpha)


class PrioritizedDoubleDQN(DoubleDQN):
    def __init__(self, *args, per_beta_start=0.4, **kwargs):
        if not 0 <= per_beta_start <= 1:
            raise ValueError("per_beta_start must be in [0,1]")
        self.per_beta_start = per_beta_start
        kwargs.setdefault("replay_buffer_class", PrioritizedReplayBuffer)
        super().__init__(*args, **kwargs)
        if self.n_steps != 1:
            raise ValueError("This PER implementation supports one-step targets only")

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if not isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            raise TypeError("PrioritizedDoubleDQN requires PrioritizedReplayBuffer")
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        progress = np.clip(1 - self._current_progress_remaining, 0., 1.)
        beta = self.per_beta_start + (1 - self.per_beta_start) * progress
        losses, errors = [], []
        for _ in range(gradient_steps):
            data, indices, weights = self.replay_buffer.sample_with_info(
                batch_size, beta, env=self._vec_normalize_env)
            discounts = data.discounts if data.discounts is not None else self.gamma
            with th.no_grad():
                target = data.rewards + (1 - data.dones) * discounts * self._next_q_values(data.next_observations)
            current = self.q_net(data.observations).gather(1, data.actions.long())
            td_errors = (target - current).detach().cpu().numpy().reshape(-1)
            loss = (weights * F.smooth_l1_loss(current, target, reduction="none")).mean()
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            self.replay_buffer.update_priorities(indices, td_errors)
            losses.append(loss.item())
            errors.append(float(np.abs(td_errors).mean()))
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/per_beta", float(beta))
        self.logger.record("train/mean_abs_td_error", np.mean(errors))
