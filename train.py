"""
train.py

Training for Signal12 with two INDEPENDENT DQN models:

  - signal_agent: learns to choose HIGH/LOW signals
  - play_agent:   learns to choose which card to play

Why split them, and why bots?
------------------------------
If a single model (or one model controlling all four seats) makes both
kinds of decisions, a bad/exploratory signal and a bad/exploratory play can
both occur in the same game, and the shared win/loss reward can't cleanly
tell either one which of its choices actually mattered.

Why *separate games*, not just separate models?
-------------------------------------------------
An earlier version of this file still ran both models in the *same* game
(one team = SignalBot + play_agent, the other = signal_agent + PlayBot).
That still let bad data leak into each model's inputs: play_agent's
observation includes the opponent's live signals, which came from a
still-learning (early on, near-random) signal_agent - so play_agent was
partly learning to react to noise. Symmetrically, signal_agent's round
outcomes depended on the opponent's play_agent, itself still learning -
so signal_agent's reward signal was contaminated by a non-stationary,
noisy opponent too.

To fix this, training now runs two entirely separate kinds of games, and
every episode is randomly one or the other:

  PLAY-training games: every seat's signal comes from the scripted
      SignalBot (deterministic heuristic), and every seat's card play
      comes from the (single, shared) play_agent - i.e. play_agent plays
      itself, self-play style, with all signals guaranteed to come from a
      consistent, non-learning source. No signal_agent involved at all.

  SIGNAL-training games: every seat's card play comes from the scripted
      PlayBot (deterministic: HIGH -> highest card, LOW -> lowest card),
      and every seat's signal comes from the (single, shared) signal_agent
      - i.e. signal_agent plays itself, with every signal guaranteed to be
      converted into a play by a consistent, non-learning source. No
      play_agent involved at all.

Each episode type produces transitions only for its own model/buffer, so
play_agent only ever sees SignalBot's honest signals, and signal_agent's
outcomes only ever depend on PlayBot's honest, deterministic response -
never on the other model's in-progress, noisy behavior. See bots.py for
the heuristics, and evaluate() below for a separate scenario where the two
independently-trained models are finally put on the same team together.

Observations are egocentric (see signal12.get_observation) and never
reveal absolute seat identity, so having all four seats share one learning
model within an episode (self-play) costs no generality.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from signal12 import (
    Signal12Game, TEAM_OF, PHASE_SIGNAL, PHASE_PLAY, SIGNAL_HIGH, SIGNAL_LOW,
)
from signalAgent import DQNAgent, ReplayBuffer, make_signal_agent, make_play_agent
from bots import SignalBot, PlayBot

GROUP_P_TEAM = 0  # Team A (players 0, 2) - used only by evaluate()'s scenarios
GROUP_Q_TEAM = 1  # Team B (players 1, 3) - used only by evaluate()'s scenarios

signal_bot = SignalBot()
play_bot = PlayBot()


# --------------------------------------------------------------------------
# Episode generation: two fully separate training regimes
# --------------------------------------------------------------------------
def run_play_training_episode(
    play_agent: DQNAgent,
    play_buffer: ReplayBuffer,
    epsilon: float,
    shaping_weight: float = 0.05,
    seed: Optional[int] = None,
) -> Signal12Game:
    """All four seats signal via SignalBot; all four seats play via the
    shared play_agent (self-play). Produces transitions only for
    play_agent - signal_agent is not involved in this episode at all."""
    game = Signal12Game(seed=seed)

    pending: Dict[int, Tuple[np.ndarray, int]] = {}
    bonus: Dict[int, float] = {p: 0.0 for p in range(4)}
    rounds_seen = 0

    while not game.done:
        dm = game.current_decision_maker
        phase = game.phase
        legal = game.legal_actions()
        obs = game.get_observation(dm)

        if phase == PHASE_SIGNAL:
            action = signal_bot.act(game, dm)
        else:
            if dm in pending:
                p_state, p_action = pending.pop(dm)
                reward = bonus[dm]
                bonus[dm] = 0.0
                play_buffer.push(p_state, p_action, reward, obs, legal, False)
            action = play_agent.act(obs, legal, epsilon=epsilon)
            pending[dm] = (obs, action)

        game.step(action)

        if len(game.round_history) > rounds_seen:
            rounds_seen = len(game.round_history)
            winner_team = game.round_history[-1]["winner_team"]
            for p in range(4):
                bonus[p] += shaping_weight if TEAM_OF[p] == winner_team else -shaping_weight

    for p, (s, a) in pending.items():
        reward = game.team_result(p) + bonus[p]
        play_buffer.push(s, a, reward, None, None, True)

    return game


def run_signal_training_episode(
    signal_agent: DQNAgent,
    signal_buffer: ReplayBuffer,
    epsilon: float,
    shaping_weight: float = 0.05,
    seed: Optional[int] = None,
) -> Signal12Game:
    """All four seats play via PlayBot; all four seats signal via the
    shared signal_agent (self-play). Produces transitions only for
    signal_agent - play_agent is not involved in this episode at all."""
    game = Signal12Game(seed=seed)

    pending: Dict[int, Tuple[np.ndarray, int]] = {}
    bonus: Dict[int, float] = {p: 0.0 for p in range(4)}
    rounds_seen = 0

    while not game.done:
        dm = game.current_decision_maker
        phase = game.phase
        legal = game.legal_actions()
        obs = game.get_observation(dm)

        if phase == PHASE_SIGNAL:
            if dm in pending:
                p_state, p_action = pending.pop(dm)
                reward = bonus[dm]
                bonus[dm] = 0.0
                signal_buffer.push(p_state, p_action, reward, obs, legal, False)
            action = signal_agent.act(obs, legal, epsilon=epsilon)
            pending[dm] = (obs, action)
        else:
            action = play_bot.act(game.hands[dm], game.signals.get(dm))

        game.step(action)

        if len(game.round_history) > rounds_seen:
            rounds_seen = len(game.round_history)
            winner_team = game.round_history[-1]["winner_team"]
            for p in range(4):
                bonus[p] += shaping_weight if TEAM_OF[p] == winner_team else -shaping_weight

    for p, (s, a) in pending.items():
        reward = game.team_result(p) + bonus[p]
        signal_buffer.push(s, a, reward, None, None, True)

    return game


# --------------------------------------------------------------------------
# Evaluation helpers - generic actor-function based simulator
# --------------------------------------------------------------------------
ActorFn = Callable[[Signal12Game, int, str, list], int]


def simulate_game(seed: int, actor_fn: ActorFn) -> Signal12Game:
    game = Signal12Game(seed=seed)
    while not game.done:
        dm = game.current_decision_maker
        phase = game.phase
        legal = game.legal_actions()
        action = actor_fn(game, dm, phase, legal)
        game.step(action)
    return game


def bot_signal_fn(game, player, phase, legal):
    return signal_bot.act(game, player)


def bot_play_fn(game, player, phase, legal):
    return play_bot.act(game.hands[player], game.signals.get(player))


def random_fn(game, player, phase, legal):
    return random.choice(legal)


def model_fn(agent: DQNAgent):
    def f(game, player, phase, legal):
        obs = game.get_observation(player)
        return agent.act(obs, legal, epsilon=0.0)
    return f


def make_controller(team_p_signal, team_p_play, team_q_signal, team_q_play) -> ActorFn:
    def actor_fn(game, player, phase, legal):
        team = TEAM_OF[player]
        if phase == PHASE_SIGNAL:
            fn = team_p_signal if team == GROUP_P_TEAM else team_q_signal
        else:
            fn = team_p_play if team == GROUP_P_TEAM else team_q_play
        return fn(game, player, phase, legal)
    return actor_fn


def evaluate(signal_agent: DQNAgent, play_agent: DQNAgent, num_games: int = 300, seed: int = 0) -> dict:
    """Runs three evaluation scenarios and one behavioral-correlation probe:

    1. combined_model_vs_random: a single team where BOTH decision types are
       handled by the trained models (signal_agent signals, play_agent
       plays) vs. a fully random opposing team. This is the cleanest
       end-to-end measure of "do the two learned models work together
       to beat chance", since neither scripted bot is propping them up.
    2. play_model_vs_random: isolates play_agent's skill - Team A
       (SignalBot + play_agent) vs. a fully random Team B.
    3. signal_model_vs_random: isolates signal_agent's skill - Team B
       (signal_agent + PlayBot) vs. a fully random Team A.
    4. signal/hand-strength correlation and play/signal correlation, as a
       communication-quality probe (see docstring in run this module).
    """
    rng = random.Random(seed)

    combined_vs_random_ctrl = make_controller(model_fn(signal_agent), model_fn(play_agent), random_fn, random_fn)
    play_vs_random_ctrl = make_controller(bot_signal_fn, model_fn(play_agent), random_fn, random_fn)
    signal_vs_random_ctrl = make_controller(random_fn, random_fn, model_fn(signal_agent), bot_play_fn)

    wins_combined_vs_random = 0
    wins_play_vs_random = 0
    wins_signal_vs_random = 0

    hand_strengths, signals_sent = [], []       # for signal_agent
    received_signals, played_ranks = [], []     # for play_agent

    for _ in range(num_games):
        s = rng.randrange(1_000_000)

        g = simulate_game(s, combined_vs_random_ctrl)
        if g.winner_team == GROUP_P_TEAM:
            wins_combined_vs_random += 1

        g2 = simulate_game(s, play_vs_random_ctrl)
        if g2.winner_team == GROUP_P_TEAM:
            wins_play_vs_random += 1

        g3 = simulate_game(s, signal_vs_random_ctrl)
        if g3.winner_team == GROUP_Q_TEAM:
            wins_signal_vs_random += 1

    # Behavioral probes: run additional greedy specialist games while
    # recording signal_agent's and play_agent's decisions directly.
    for _ in range(num_games):
        s = rng.randrange(1_000_000)
        game = Signal12Game(seed=s)
        while not game.done:
            dm = game.current_decision_maker
            phase = game.phase
            legal = game.legal_actions()
            obs = game.get_observation(dm)
            if phase == PHASE_SIGNAL and TEAM_OF[dm] == GROUP_Q_TEAM:
                hand = game.hands[dm]
                if hand:
                    hand_strengths.append(max(hand) / 12.0)
                    action = signal_agent.act(obs, legal, epsilon=0.0)
                    signals_sent.append(action)
                else:
                    action = signal_agent.act(obs, legal, epsilon=0.0)
                game.step(action)
                continue
            if phase == PHASE_PLAY and TEAM_OF[dm] == GROUP_P_TEAM:
                signal = game.signals.get(dm, SIGNAL_LOW)
                action = play_agent.act(obs, legal, epsilon=0.0)
                played_value = game.hands[dm][action]
                received_signals.append(signal)
                played_ranks.append(played_value / 12.0)
                game.step(action)
                continue
            # everyone else this round follows their scripted role
            if phase == PHASE_SIGNAL:
                action = bot_signal_fn(game, dm, phase, legal)
            else:
                action = bot_play_fn(game, dm, phase, legal)
            game.step(action)

    def corr(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if len(a) < 2 or a.std() == 0 or b.std() == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "combined_model_vs_random_winrate": wins_combined_vs_random / num_games,
        "play_model_vs_random_winrate": wins_play_vs_random / num_games,
        "signal_model_vs_random_winrate": wins_signal_vs_random / num_games,
        "signal_hand_correlation": corr(hand_strengths, signals_sent),
        "play_signal_correlation": corr(received_signals, played_ranks),
    }


# --------------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    signal_agent = make_signal_agent(hidden=args.hidden, lr=args.lr, gamma=args.gamma, device=args.device)
    play_agent = make_play_agent(hidden=args.hidden, lr=args.lr, gamma=args.gamma, device=args.device)
    signal_buffer = ReplayBuffer(capacity=args.buffer_size)
    play_buffer = ReplayBuffer(capacity=args.buffer_size)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    eps_start, eps_end, eps_decay_episodes = args.eps_start, args.eps_end, args.eps_decay_episodes
    play_train_steps = 0
    signal_train_steps = 0
    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        frac = min(1.0, episode / eps_decay_episodes)
        epsilon = eps_start + frac * (eps_end - eps_start)

        # Each episode is entirely a play-training game OR a signal-training
        # game (never both) - see module docstring for why.
        if random.random() < args.play_episode_prob:
            run_play_training_episode(play_agent, play_buffer, epsilon=epsilon,
                                       shaping_weight=args.shaping_weight)
        else:
            run_signal_training_episode(signal_agent, signal_buffer, epsilon=epsilon,
                                         shaping_weight=args.shaping_weight)

        if len(play_buffer) >= args.warmup_steps:
            for _ in range(args.train_updates_per_episode):
                play_agent.train_step(play_buffer.sample(args.batch_size))
                play_train_steps += 1
                if play_train_steps % args.target_update_interval == 0:
                    play_agent.update_target()

        if len(signal_buffer) >= args.warmup_steps:
            for _ in range(args.train_updates_per_episode):
                signal_agent.train_step(signal_buffer.sample(args.batch_size))
                signal_train_steps += 1
                if signal_train_steps % args.target_update_interval == 0:
                    signal_agent.update_target()

        if episode % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"[episode {episode:6d}] epsilon={epsilon:.3f} "
                  f"signal_buffer={len(signal_buffer):6d} play_buffer={len(play_buffer):6d} "
                  f"signal_train_steps={signal_train_steps:6d} play_train_steps={play_train_steps:6d} "
                  f"elapsed={elapsed:.1f}s")

        if episode % args.eval_interval == 0 or episode == args.episodes:
            stats = evaluate(signal_agent, play_agent, num_games=args.eval_games)

            def fmt(x):
                return f"{x:+.3f}" if x is not None else "n/a"

            print(f"  -> EVAL @ episode {episode}:")
            print(f"       combined team (signal_agent + play_agent) vs fully random team: "
                  f"win rate = {stats['combined_model_vs_random_winrate']:.2%}")
            print(f"       play_agent   vs random opponent: win rate = {stats['play_model_vs_random_winrate']:.2%}")
            print(f"       signal_agent vs random opponent: win rate = {stats['signal_model_vs_random_winrate']:.2%}")
            print(f"       signal_agent: correlation(own best card, HIGH signal) = {fmt(stats['signal_hand_correlation'])}")
            print(f"       play_agent:   correlation(received signal, card played) = {fmt(stats['play_signal_correlation'])}")

            signal_agent.save(os.path.join(args.checkpoint_dir, f"signal_agent_ep{episode}.pt"))
            play_agent.save(os.path.join(args.checkpoint_dir, f"play_agent_ep{episode}.pt"))
            print(f"  -> saved checkpoints for episode {episode}")

    signal_agent.save(os.path.join(args.checkpoint_dir, "signal_agent_final.pt"))
    play_agent.save(os.path.join(args.checkpoint_dir, "play_agent_final.pt"))
    print("Training complete. Final models saved to "
          f"{args.checkpoint_dir}/signal_agent_final.pt and {args.checkpoint_dir}/play_agent_final.pt")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train separate signal/play DQN models for Signal12.")
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--warmup-steps", type=int, default=500,
                   help="Minimum replay buffer size (each buffer) before training starts.")
    p.add_argument("--train-updates-per-episode", type=int, default=4)
    p.add_argument("--target-update-interval", type=int, default=200)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-episodes", type=int, default=8000)
    p.add_argument("--shaping-weight", type=float, default=0.05)
    p.add_argument("--play-episode-prob", type=float, default=0.5,
                   help="Probability that a given training episode is a play-training game "
                        "(SignalBot + self-play play_agent) rather than a signal-training game "
                        "(self-play signal_agent + PlayBot).")
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--eval-games", type=int, default=300)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    train(parser.parse_args())
