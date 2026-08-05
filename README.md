# Signal12 
I trained a DQN to play a card game that I developed called *Signal12*. The game has four players divided into two teams, and features public signals between teammates to test communication. The DQN achieves this with **two independently-trained models** — one for signaling, one for card play — each trained in its *own* self-play games against a deterministic scripted bot. This experiment showed that Signal12 has a large component of luck, which created too much noise for the signaling mechanism to significantly improve the DQN's performance. Nonetheless, the win rate was still consistently over 50% (sometimes over 60%), which is mild evidence of learning. 

[See "Signal12 Project" Powerpoint for more information]

## Overview
We have already reached a point in development of AI technologies where AI algorithms can learn and surpass human skill level in deterministic games such as Chess and Go. I have also shown in the "Stay Positive" project that RL models can learn new games and invent strategies, even gauging the overall impact of luck components on the outcome. However, I wanted to test if AI can play games a step beyond purely deterministic "move -> reap reward" schemas. I added a signaling component to test communication skills. To win Signal12, a team must be able to coordinate who should play high or low cards without giving their cards away to the other team. I searched for evidence of trust (following signals), and thought of this basic game as a stepping stone to much larger game theory type problems. The result was somewhat underwhelming: due to the large luck component in the game, even good signals had a noisy correlation to winning. Nonetheless, the performance of the DQN team (trained separately then tested together) was consistently better than a random team. This shows signs of communication and trust because in a random team, signals are essentially worthless, so the DQN performance is gauging the value of signaling. 

## Results
In the early experiment, before separating the training of the signaling and card choice models, the DQN win rate hovered around 60%. The second trial, addressing the possible data contamination, actually lowered that rate to 54%, leading me to believe that the game has a very large component of luck, diminishing the benefit of strategy in the first place. However, this is still consistently better than random, so there is reason to believe that in a game like Signal20 (see Powerpoint for more details), the benefit would be larger (more possible actions, less likely one team gets a better hand in the beginning). 

## Approach

### 1. Data Formulation
An input observation consists of the player hand, played flags, cards played, signal flags, signals given, team scores, phase flag, and the round number. These are used in two separate models to make signal and play card choice outputs.   

### 2. Model Architecture
I used two separate DQN models: one for signaling, and one for play card choice. If one model makes both the signal decisions and the play decisions, a bad signal and a bad play can both happen in the same game, and a shared win/loss reward can't cleanly tell either decision which of the two actually mattered.

Splitting into two models isn't enough by itself, though: an earlier version of this project still ran both models in the *same* game (one
team = SignalBot + play_agent, the other = signal_agent + PlayBot). That still let bad data leak in both directions — `play_agent`'s observation included the opponent's live signals, which came from a still-learning (often near-random early on) `signal_agent`, so `play_agent` was partly learning to react to noise. Symmetrically, `signal_agent`'s round outcomes depended on the opponent's `play_agent`, itself still learning, so `signal_agent`'s reward signal was contaminated by a moving-target opponent too.  

### 3. Training & Evaluation
I trained the models in isolation for 20000 training episodes and measured their collaborative performance at 1000 episode checkpoints. Both models had the same basic architecture, except for the output dimension.  

I used a policy and target DQN structure with a replay buffer capacity of 50000, and epsilon-greedy approach with constant decay over 8000 episodes, fixed learning rate of 0.001, batch size of 64, reward shaping rate of 0.05 (how much winning or losing in one game impacts gradients), and gamma = 0.99 (future reward multiplier, so model could think ahead). I also randomly selected a play or signal training episode (probability 0.5 each) at every step. Evaluation was run every 1000 episodes with 300 games of random v.s. DQN + bot teams to isolate the models. I also conducted collaborative tests (fully DQN team v.s. random) to check for the emergence of cooperation. 

## What "Signal12" Looks Like

## Usage

```bash
pip install torch numpy

# Train (defaults to 20,000 episodes; adjust as needed)
python train.py --episodes 20000 --checkpoint-dir checkpoints

# Play against the trained models (they jointly control Players 1, 2, 3)
python play.py --signal-checkpoint checkpoints/signal_agent_final.pt \
                --play-checkpoint checkpoints/play_agent_final.pt
```

Checkpoints from a short (6,000-episode) validation run are included in
`checkpoints/`. In that run, `play_agent` reached ~55-59% against a fully
random opponent, while `signal_agent` stayed close to chance (~48-52%) —
signaling is the harder credit-assignment problem here, since its effect
on the outcome is one step removed (signal → bot's deterministic play →
round result) and it has to learn a convention essentially from scratch.
Expect `signal_agent` to need noticeably more training episodes than
`play_agent`; if it's still flat after tens of thousands of episodes,
that itself is an interesting finding about how learnable this signaling
mechanic is.

## Notes on the design

- Reward is mostly sparse (+1/-1 for winning/losing the game), plus a
  small optional per-round-win shaping bonus (`--shaping-weight`, default
  0.05).
- Observations never reveal absolute seat identity (they're always
  `[self, teammate, opponent, opponent]`-ordered), so having all four
  seats share one learning model within a training episode (self-play)
  costs no generality.
- `--play-episode-prob` controls the mix of the two training-game types
  (default 0.5/0.5). If one model seems to be lagging, you can bias
  training toward it, e.g. `--play-episode-prob 0.3` spends more episodes
  training `signal_agent`.
