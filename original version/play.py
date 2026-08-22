"""
play.py

Lets a human play Signal12 as Player 0 (Team A, teammate Player 2).

Every other seat (Players 1, 2, and 3) is controlled by the two trained
models used together, the same way evaluate()'s `combined_model_vs_random`
scenario in train.py does: whenever it's a SIGNAL decision, signal_agent
chooses; whenever it's a PLAY decision, play_agent chooses. The scripted
SignalBot/PlayBot from training aren't used here at all - they exist only
to give each model a clean, isolated training signal (see train.py's
docstring); once trained, the two models are simply used together.

So: you make both decisions for your own seat (the signal you send for
Player 2's turn, and the card you play on your own turn), and the two
trained models jointly control Players 1, 2, and 3.

Usage:
    python play.py --signal-checkpoint checkpoints/signal_agent_final.pt \
                    --play-checkpoint checkpoints/play_agent_final.pt
"""

from __future__ import annotations

import argparse
import os

from signal12 import (
    Signal12Game, TEAM_OF, TEAMMATE, PHASE_SIGNAL, PHASE_PLAY,
    SIGNAL_LOW, SIGNAL_HIGH,
)
from signalAgent import make_signal_agent, make_play_agent

HUMAN = 0


def prompt_choice(prompt: str, options: dict) -> int:
    while True:
        raw = input(prompt).strip().lower()
        if raw in options:
            return options[raw]
        print(f"  Invalid input. Options: {', '.join(options.keys())}")


def human_signal_action(game: Signal12Game) -> int:
    actor = game.current_actor
    print(f"\nYou (Player {HUMAN}) must send a signal for Player {actor}'s upcoming turn.")
    print(f"Your hand: {game.hands[HUMAN]}")
    options = {"h": SIGNAL_HIGH, "high": SIGNAL_HIGH, "l": SIGNAL_LOW, "low": SIGNAL_LOW}
    return prompt_choice("Send signal - (H)IGH or (L)OW? ", options)


def human_play_action(game: Signal12Game) -> int:
    hand = game.hands[HUMAN]
    signal = game.signals.get(HUMAN)
    signal_str = {SIGNAL_LOW: "LOW", SIGNAL_HIGH: "HIGH"}.get(signal, "none")
    print(f"\nYour turn to play. Your teammate (Player {TEAMMATE[HUMAN]}) signaled: {signal_str}")
    print(f"Your hand: {list(enumerate(hand))}  (index: card value)")
    options = {str(i): i for i in range(len(hand))}
    return prompt_choice(f"Choose a card index to play {list(options.keys())}: ", options)


def render_round_result(game: Signal12Game) -> None:
    last = game.round_history[-1]
    team_name = "A (you!)" if last["winner_team"] == 0 else "B"
    print(f"\n=== Round {last['round']} result ===")
    for p in range(4):
        sig = {SIGNAL_LOW: "LOW", SIGNAL_HIGH: "HIGH"}.get(last["signals"].get(p), "-")
        print(f"  Player {p}: played {last['played'][p]:>2}  (was signaled: {sig})")
    print(f"  Winner: Player {last['winner_player']} - Team {team_name}")
    print(f"  Score now -> Team A: {game.scores[0]}   Team B: {game.scores[1]}")


def play(signal_checkpoint: str, play_checkpoint: str, seed=None) -> None:
    signal_agent = make_signal_agent()
    play_agent = make_play_agent()

    if signal_checkpoint and os.path.exists(signal_checkpoint):
        signal_agent.load(signal_checkpoint)
        print(f"Loaded signal model from {signal_checkpoint}")
    else:
        print(f"WARNING: signal checkpoint '{signal_checkpoint}' not found - "
              f"all AI signaling will use an UNTRAINED model.")

    if play_checkpoint and os.path.exists(play_checkpoint):
        play_agent.load(play_checkpoint)
        print(f"Loaded play model from {play_checkpoint}")
    else:
        print(f"WARNING: play checkpoint '{play_checkpoint}' not found - "
              f"all AI card play will use an UNTRAINED model.")

    game = Signal12Game(seed=seed)
    print("\nWelcome to Signal12! You are Player 0 on Team A (teammate: Player 2).")
    print("Players 1, 2, and 3 are all controlled by the trained models: "
          "signal_agent signals, play_agent plays.")
    print("First team to win 2 rounds wins the game.\n")

    rounds_reported = 0
    while not game.done:
        dm = game.current_decision_maker
        phase = game.phase
        legal = game.legal_actions()

        if dm == HUMAN:
            action = human_signal_action(game) if phase == PHASE_SIGNAL else human_play_action(game)
        elif phase == PHASE_SIGNAL:
            obs = game.get_observation(dm)
            action = signal_agent.act(obs, legal, epsilon=0.0)
            actor = game.current_actor
            label = {SIGNAL_LOW: "LOW", SIGNAL_HIGH: "HIGH"}[action]
            print(f"\n[Player {dm} (signal_agent) signals {label} for Player {actor}'s turn]")
        else:
            obs = game.get_observation(dm)
            action = play_agent.act(obs, legal, epsilon=0.0)
            print(f"[Player {dm} (play_agent) plays {game.hands[dm][action]}]")

        game.step(action)

        if len(game.round_history) > rounds_reported:
            rounds_reported = len(game.round_history)
            render_round_result(game)

    print("\n================ GAME OVER ================")
    if game.winner_team == TEAM_OF[HUMAN]:
        print(f"Your team (A) WINS! Final score: A {game.scores[0]} - B {game.scores[1]}")
    else:
        print(f"Your team (A) loses. Final score: A {game.scores[0]} - B {game.scores[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Signal12 against the trained signal/play models.")
    parser.add_argument("--signal-checkpoint", type=str, default="checkpoints/signal_agent_final.pt")
    parser.add_argument("--play-checkpoint", type=str, default="checkpoints/play_agent_final.pt")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    play(args.signal_checkpoint, args.play_checkpoint, seed=args.seed)
