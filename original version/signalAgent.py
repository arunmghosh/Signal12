"""
signalAgent.py

Two independent DQN models for Signal12:

  - a SIGNAL model: observes the game state and chooses HIGH/LOW
  - a PLAY model:   observes the game state and chooses a card to play

Keeping them as two separate networks (rather than two heads sharing a
trunk, and rather than one network controlling every seat) means each
model's gradient updates are driven only by its own decisions. Combined
with the fixed roles in train.py (each model is always paired against a
*deterministic* scripted bot handling the other decision type - see
bots.py), this keeps credit assignment clean: a play model's outcomes
aren't confounded by a partner exploring random signals, and a signal
model's outcomes aren't confounded by a partner exploring random card
plays.

`DQNAgent` is a generic single-head Double-DQN agent (network + replay
buffer + training) parameterized by its action space size. `train.py`
instantiates one with action_size=2 for signaling and one with
action_size=3 for card play.
"""

from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from signal12 import OBS_SIZE

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "next_legal_actions", "done"],
)


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
class QNetwork(nn.Module):
    """Simple MLP producing one Q-value per action."""

    def __init__(self, obs_size: int = OBS_SIZE, action_size: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
class DQNAgent:
    """A single-head Double-DQN agent for one decision type (signal OR
    play). `action_size` should be 2 for a signal agent, or 3 (the max
    hand size) for a play agent - illegal/out-of-range actions are masked
    out both when acting and when bootstrapping the training target."""

    def __init__(
        self,
        action_size: int,
        obs_size: int = OBS_SIZE,
        hidden: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: Optional[str] = None,
    ):
        self.action_size = action_size
        self.obs_size = obs_size
        self.gamma = gamma
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_net = QNetwork(obs_size, action_size, hidden).to(self.device)
        self.target_net = QNetwork(obs_size, action_size, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, legal_actions: Sequence[int], epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.choice(list(legal_actions))
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(x).squeeze(0).clone()
        mask = torch.full_like(q, float("-inf"))
        mask[list(legal_actions)] = 0.0
        q = q + mask
        return int(torch.argmax(q).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def train_step(self, batch: List[Transition]) -> float:
        states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        zero_state = np.zeros(self.obs_size, dtype=np.float32)
        next_states = torch.as_tensor(
            np.stack([t.next_state if t.next_state is not None else zero_state for t in batch]),
            dtype=torch.float32, device=self.device,
        )

        q_values = self.policy_net(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)
            next_q_target = self.target_net(next_states)
            next_q = torch.zeros(len(batch), device=self.device)
            for i, t in enumerate(batch):
                if t.done:
                    continue
                mask = torch.full((self.action_size,), float("-inf"), device=self.device)
                mask[list(t.next_legal_actions)] = 0.0
                best_a = int(torch.argmax(next_q_policy[i] + mask).item())
                next_q[i] = next_q_target[i, best_a]

        target_q = rewards + (1.0 - dones) * self.gamma * next_q
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "obs_size": self.obs_size,
            "action_size": self.action_size,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])


# Convenience constructors so callers don't need to remember action sizes.
def make_signal_agent(**kwargs) -> DQNAgent:
    return DQNAgent(action_size=2, **kwargs)


def make_play_agent(**kwargs) -> DQNAgent:
    from signal12 import MAX_CARD_ACTIONS
    return DQNAgent(action_size=MAX_CARD_ACTIONS, **kwargs)
