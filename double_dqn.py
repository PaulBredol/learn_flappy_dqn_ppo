"""Double DQN extension for the project's Stable-Baselines3 2.9.0 DQN.

Training loop follows SB3 DQN; only selection of the next-state value changes.
Reference: van Hasselt, Guez & Silver, https://arxiv.org/abs/1509.06461
"""
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import DQN


class DoubleDQN(DQN):
    def _next_q_values(self, next_observations):
        """Online network selects; target network evaluates. Called without grad."""
        next_actions = self.q_net(next_observations).argmax(dim=1, keepdim=True)
        return self.q_net_target(next_observations).gather(1, next_actions)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            # Preserve SB3's support for n-step discounts and terminal masking.
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            with th.no_grad():
                next_q_values = self._next_q_values(replay_data.next_observations)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
