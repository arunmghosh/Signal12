"""
signal12.py

Rules engine for Signal12: a 4-player, 2-team trick-taking card game with
public HIGH/LOW signaling.

Design notes for RL usage
--------------------------
The game is turn-based at the level of *atomic decisions*. Each of the four
"turns" in a round is actually made of two atomic decisions:

    1. SIGNAL decision - the acting player's teammate publicly announces
       HIGH (1) or LOW (0) before the acting player plays a card.
    2. PLAY decision   - the acting player plays one card from their hand.

So a full round consists of 8 atomic decisions (4 signal + 4 play), and a
full game is 2-3 rounds (game ends the instant a team reaches 2 round wins).

The engine exposes a single-decision-at-a-time API (`current_decision_maker`,
`phase`, `legal_actions()`, `step(action)`), which is convenient for
self-play training: at every point in the game there is exactly one player
who must act next, regardless of whether that action is a signal or a card
play.

Observations are *egocentric*: `get_observation(player)` always returns a
feature vector ordered as [self, teammate, opponent, opponent], regardless
of the player's absolute seat index. This lets a single shared network
(used for all 4 seats via self-play) generalize across seats.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

NUM_PLAYERS = 4
DECK = list(range(1, 13))  # cards 1..12
HAND_SIZE = 3
POINTS_TO_WIN = 2

TEAM_OF = {0: 0, 2: 0, 1: 1, 3: 1}          # Team A = {0,2}, Team B = {1,3}
TEAMMATE = {0: 2, 2: 0, 1: 3, 3: 1}

SIGNAL_LOW = 0
SIGNAL_HIGH = 1

PHASE_SIGNAL = "signal"
PHASE_PLAY = "play"

MAX_CARD_ACTIONS = 3    # largest hand size (round 1)
NUM_SIGNAL_ACTIONS = 2  # LOW / HIGH

# Observation layout (see get_observation for details):
#   hand one-hot            : 12
#   played flags (4 seats)  : 4
#   played values (4 seats) : 4
#   signal flags (4 seats)  : 4
#   signal values (4 seats) : 4
#   own/opp score           : 2
#   phase flag              : 1
#   round number (norm.)    : 1
OBS_SIZE = 12 + 4 + 4 + 4 + 4 + 2 + 1 + 1  # = 32


class Signal12Error(RuntimeError):
    pass


class Signal12Game:
    """Rules engine for Signal12."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def reset(self) -> None:
        deck = list(DECK)
        self.rng.shuffle(deck)
        self.hands: List[List[int]] = [
            sorted(deck[i * HAND_SIZE:(i + 1) * HAND_SIZE]) for i in range(NUM_PLAYERS)
        ]
        self.scores: List[int] = [0, 0]  # indexed by team id
        self.round_num = 0
        self.round_history: List[dict] = []  # for rendering / debugging
        self.start_player = self.rng.randrange(NUM_PLAYERS)
        self.done = False
        self.winner_team: Optional[int] = None
        self.last_round_winner: Optional[int] = None
        self.cards_left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # for signaling bot 
        self._start_round()

    def _start_round(self) -> None:
        self.round_num += 1
        self.turn_order = [(self.start_player + i) % NUM_PLAYERS for i in range(NUM_PLAYERS)]
        self.turn_index = 0
        self.played: Dict[int, int] = {}   # player -> card value
        self.signals: Dict[int, int] = {}  # player -> signal given FOR that player's turn
        self.phase = PHASE_SIGNAL

    # ------------------------------------------------------------------
    # Decision-point properties
    # ------------------------------------------------------------------
    @property
    def current_actor(self) -> int:
        """The player whose card-play turn this is (regardless of phase)."""
        return self.turn_order[self.turn_index]

    @property
    def current_decision_maker(self) -> int:
        """The player who must supply the *next* action."""
        if self.done:
            raise Signal12Error("Game is over; no decision maker.")
        if self.phase == PHASE_SIGNAL:
            return TEAMMATE[self.current_actor]
        return self.current_actor

    def legal_actions(self) -> List[int]:
        if self.done:
            return []
        if self.phase == PHASE_SIGNAL:
            return [SIGNAL_LOW, SIGNAL_HIGH]
        return list(range(len(self.hands[self.current_actor])))

    # ------------------------------------------------------------------
    # Stepping the game
    # ------------------------------------------------------------------
    def step(self, action: int) -> None:
        if self.done:
            raise Signal12Error("Cannot step a finished game.")
        legal = self.legal_actions()
        if action not in legal:
            raise Signal12Error(f"Illegal action {action}; legal={legal}, phase={self.phase}")

        if self.phase == PHASE_SIGNAL:
            actor = self.current_actor
            self.signals[actor] = action
            self.phase = PHASE_PLAY
        else:
            actor = self.current_actor
            card = self.hands[actor].pop(action)
            self.cards_left.remove(card)
            self.played[actor] = card
            self.turn_index += 1
            if self.turn_index == NUM_PLAYERS:
                self._resolve_round()
            else:
                self.phase = PHASE_SIGNAL

    def _resolve_round(self) -> None:
        winner_player = max(self.played, key=lambda p: self.played[p])
        winner_team = TEAM_OF[winner_player]
        self.scores[winner_team] += 1
        self.last_round_winner = winner_player
        self.round_history.append({
            "round": self.round_num,
            "played": dict(self.played),
            "signals": dict(self.signals),
            "winner_player": winner_player,
            "winner_team": winner_team,
        })
        if self.scores[winner_team] >= POINTS_TO_WIN:
            self.done = True
            self.winner_team = winner_team
        else:
            self.start_player = winner_player
            self._start_round()

    # ------------------------------------------------------------------
    # Observations (egocentric: [self, teammate, opp, opp])
    # ------------------------------------------------------------------
    def _relative_order(self, player: int) -> List[int]:
        teammate = TEAMMATE[player]
        others = sorted(p for p in range(NUM_PLAYERS) if TEAM_OF[p] != TEAM_OF[player])
        return [player, teammate] + others

    def get_observation(self, player: int) -> np.ndarray:
        order = self._relative_order(player)

        hand_vec = [0.0] * 12
        for c in self.hands[player]:
            hand_vec[c - 1] = 1.0

        played_flags, played_vals = [], []
        for p in order:
            if p in self.played:
                played_flags.append(1.0)
                played_vals.append(self.played[p] / 12.0)
            else:
                played_flags.append(0.0)
                played_vals.append(0.0)

        signal_flags, signal_vals = [], []
        for p in order:
            if p in self.signals:
                signal_flags.append(1.0)
                signal_vals.append(float(self.signals[p]))
            else:
                signal_flags.append(0.0)
                signal_vals.append(0.0)

        own_team = TEAM_OF[player]
        opp_team = 1 - own_team
        score_vec = [self.scores[own_team] / POINTS_TO_WIN, self.scores[opp_team] / POINTS_TO_WIN]

        phase_flag = 0.0 if self.phase == PHASE_SIGNAL else 1.0
        round_norm = self.round_num / 3.0

        obs = hand_vec + played_flags + played_vals + signal_flags + signal_vals + score_vec + [phase_flag, round_norm]
        return np.asarray(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward helper
    # ------------------------------------------------------------------
    def team_result(self, player: int) -> float:
        """+1 if player's team won the (finished) game, -1 if lost."""
        if not self.done:
            raise Signal12Error("Game is not finished.")
        return 1.0 if TEAM_OF[player] == self.winner_team else -1.0

    # ------------------------------------------------------------------
    # Rendering helpers (used by play.py)
    # ------------------------------------------------------------------
    def render_state(self, viewer: Optional[int] = None) -> str:
        lines = []
        lines.append(f"Round {self.round_num} | Score  Team A: {self.scores[0]}  Team B: {self.scores[1]}")
        for p in range(NUM_PLAYERS):
            hand_str = str(self.hands[p]) if viewer is None or p == viewer else f"[{len(self.hands[p])} hidden cards]"
            played_str = str(self.played[p]) if p in self.played else "-"
            signal_str = {0: "LOW", 1: "HIGH"}.get(self.signals.get(p), "-")
            marker = " <-- current actor" if p == self.current_actor else ""
            lines.append(
                f"  Player {p} (Team {'A' if TEAM_OF[p] == 0 else 'B'}): "
                f"hand={hand_str} played={played_str} signal_for_this_turn={signal_str}{marker}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Quick self-test / sanity check when run directly
# --------------------------------------------------------------------------
if __name__ == "__main__":
    g = Signal12Game()
    step_count = 0
    while not g.done:
        dm = g.current_decision_maker
        legal = g.legal_actions()
        action = random.choice(legal)
        g.step(action)
        step_count += 1
        if step_count > 200:
            raise RuntimeError("Game did not terminate - possible bug.")
    print(f"Game finished in {step_count} atomic decisions.")
    print(f"Final score: Team A {g.scores[0]} - Team B {g.scores[1]}, winner team {g.winner_team}")
    for r in g.round_history:
        print(r)
    obs = g.get_observation(0)
    print("Observation size:", obs.shape, "expected", OBS_SIZE)
    assert obs.shape[0] == OBS_SIZE
    print("Self-test passed.")
