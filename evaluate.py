"""
evaluate.py

Runs standardized evaluation benchmarks on the trained Signal12 AI models:

Tests (200 games each, 100 on Evens team + 100 on Odds team):
  1. AI team (signal_agent + play_agent) vs Bot team (SignalBot + PlayBot)
  2. AI team (signal_agent + play_agent) vs Random team
  3. Play agent (SignalBot + play_agent) vs Bot team
  4. Signal agent (signal_agent + PlayBot) vs Bot team
  5. Play agent (SignalBot + play_agent) vs Random team
  6. Signal agent (signal_agent + PlayBot) vs Random team

Reports overall, evens, and odds win rates for every scenario.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

from signal12 import (
    Signal12Game, TEAM_OF, PHASE_SIGNAL, PHASE_PLAY,
)
from signalAgent import DQNAgent, make_signal_agent, make_play_agent
from bots import SignalBot, PlayBot

# Team 0 = AI / agent under test, Team 1 = Opponent
TEST_TEAM = 0
OPP_TEAM = 1

signal_bot = SignalBot()
play_bot = PlayBot()


# --------------------------------------------------------------------------
# Decision Controllers
# --------------------------------------------------------------------------
ActorFn = Callable[[Signal12Game, int, str, list], int]


def bot_signal_fn(game: Signal12Game, player: int, phase: str, legal: list) -> int:
    return signal_bot.act(game, player)


def bot_play_fn(game: Signal12Game, player: int, phase: str, legal: list) -> int:
    return play_bot.act(game.hands[player], game.signals.get(player))


def random_fn(game: Signal12Game, player: int, phase: str, legal: list) -> int:
    return random.choice(legal)


def model_fn(agent: DQNAgent) -> ActorFn:
    def f(game: Signal12Game, player: int, phase: str, legal: list) -> int:
        obs = game.get_observation(player)
        return agent.act(obs, legal, epsilon=0.0)
    return f


def make_controller(team0_signal: ActorFn, team0_play: ActorFn,
                    team1_signal: ActorFn, team1_play: ActorFn) -> ActorFn:
    def actor_fn(game: Signal12Game, player: int, phase: str, legal: list) -> int:
        team = TEAM_OF[player]
        if phase == PHASE_SIGNAL:
            fn = team0_signal if team == TEST_TEAM else team1_signal
        else:
            fn = team0_play if team == TEST_TEAM else team1_play
        return fn(game, player, phase, legal)
    return actor_fn


def simulate_game(seed: int, actor_fn: ActorFn, team0_evens: Optional[bool] = None) -> Signal12Game:
    game = Signal12Game(seed=seed, team0_evens=team0_evens)
    while not game.done:
        dm = game.current_decision_maker
        phase = game.phase
        legal = game.legal_actions()
        action = actor_fn(game, dm, phase, legal)
        game.step(action)
    return game


# --------------------------------------------------------------------------
# Evaluation Runner
# --------------------------------------------------------------------------
def evaluate_scenario(
    name: str,
    controller: ActorFn,
    games_per_parity: int,
    base_seed: int,
) -> Dict[str, float]:
    """Runs games_per_parity with AI on Evens and games_per_parity with AI on Odds."""
    wins_evens = 0
    wins_odds = 0

    rng = random.Random(base_seed)

    # 1. AI on Evens team
    for _ in range(games_per_parity):
        s = rng.randrange(10_000_000)
        g = simulate_game(seed=s, actor_fn=controller, team0_evens=True)
        if g.winner_team == TEST_TEAM:
            wins_evens += 1

    # 2. AI on Odds team
    for _ in range(games_per_parity):
        s = rng.randrange(10_000_000)
        g = simulate_game(seed=s, actor_fn=controller, team0_evens=False)
        if g.winner_team == TEST_TEAM:
            wins_odds += 1

    total_games = 2 * games_per_parity
    total_wins = wins_evens + wins_odds

    return {
        "name": name,
        "total_games": total_games,
        "total_wins": total_wins,
        "overall_winrate": total_wins / total_games,
        "evens_games": games_per_parity,
        "evens_wins": wins_evens,
        "evens_winrate": wins_evens / games_per_parity,
        "odds_games": games_per_parity,
        "odds_wins": wins_odds,
        "odds_winrate": wins_odds / games_per_parity,
    }


def run_all_evaluations(
    signal_checkpoint: str,
    play_checkpoint: str,
    games_per_parity: int = 100,
    seed: int = 42,
) -> List[Dict[str, float]]:
    signal_agent = make_signal_agent()
    play_agent = make_play_agent()

    if os.path.exists(signal_checkpoint):
        signal_agent.load(signal_checkpoint)
        print(f"Loaded signal model: {signal_checkpoint}")
    else:
        print(f"WARNING: '{signal_checkpoint}' not found; using untrained signal agent.")

    if os.path.exists(play_checkpoint):
        play_agent.load(play_checkpoint)
        print(f"Loaded play model:   {play_checkpoint}")
    else:
        print(f"WARNING: '{play_checkpoint}' not found; using untrained play agent.")

    ai_sig = model_fn(signal_agent)
    ai_play = model_fn(play_agent)

    scenarios = [
        ("AI Team vs Bot Team",
         make_controller(ai_sig, ai_play, bot_signal_fn, bot_play_fn)),

        ("AI Team vs Random Team",
         make_controller(ai_sig, ai_play, random_fn, random_fn)),

        ("Play Agent vs Bot",
         make_controller(bot_signal_fn, ai_play, bot_signal_fn, bot_play_fn)),

        ("Signal Agent vs Bot",
         make_controller(ai_sig, bot_play_fn, bot_signal_fn, bot_play_fn)),

        ("Play Agent vs Random",
         make_controller(bot_signal_fn, ai_play, random_fn, random_fn)),

        ("Signal Agent vs Random",
         make_controller(ai_sig, bot_play_fn, random_fn, random_fn)),
    ]

    results = []
    print("\n" + "=" * 80)
    print(f"RUNNING BENCHMARKS ({games_per_parity * 2} games per scenario: {games_per_parity} Evens, {games_per_parity} Odds)")
    print("=" * 80)

    for idx, (name, controller) in enumerate(scenarios, 1):
        res = evaluate_scenario(name, controller, games_per_parity, seed + idx * 1000)
        results.append(res)
        print(f"\n[{idx}/6] {name}:")
        print(f"      Overall Win Rate : {res['overall_winrate']:6.2%}  ({res['total_wins']}/{res['total_games']})")
        print(f"      Evens Win Rate   : {res['evens_winrate']:6.2%}  ({res['evens_wins']}/{res['evens_games']})")
        print(f"      Odds Win Rate    : {res['odds_winrate']:6.2%}  ({res['odds_wins']}/{res['odds_games']})")

    # Summary Table
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Scenario':<26} | {'Total':^9} | {'Overall Win%':^14} | {'Evens Win%':^12} | {'Odds Win%':^12}"
    print(header)
    print("-" * len(header))
    for res in results:
        print(f"{res['name']:<26} | {res['total_games']:^9} | {res['overall_winrate']:^14.2%} | {res['evens_winrate']:^12.2%} | {res['odds_winrate']:^12.2%}")
    print("=" * 80 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Signal12 trained models across parity and opponent conditions.")
    parser.add_argument("--signal-checkpoint", type=str, default="checkpoints/signal_agent_final.pt")
    parser.add_argument("--play-checkpoint", type=str, default="checkpoints/play_agent_final.pt")
    parser.add_argument("--games-per-parity", type=int, default=100,
                        help="Number of games played as Evens team and as Odds team (total = 2 * games_per_parity).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all_evaluations(
        signal_checkpoint=args.signal_checkpoint,
        play_checkpoint=args.play_checkpoint,
        games_per_parity=args.games_per_parity,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
