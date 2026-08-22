# Signal12 DQN (separate signal / play models, trained in separate games)

A DQN project for the *Signal12* card game (4 players, 2 teams, public
HIGH/LOW signaling before each play), with **two independently-trained
models** — one for signaling, one for card play — each trained in its
*own* self-play games against a deterministic scripted bot, so neither
model's training data is ever contaminated by the other model's
in-progress, noisy behavior.

## Why two models + two bots, in separate games

If one model (or one model controlling all four seats) makes both the
signal decisions and the play decisions, a bad signal and a bad play can
both happen in the same game, and a shared win/loss reward can't cleanly
tell either decision which of the two actually mattered.

Splitting into two models isn't enough by itself, though: an earlier
version of this project still ran both models in the *same* game (one
team = SignalBot + play_agent, the other = signal_agent + PlayBot). That
still let bad data leak in both directions — `play_agent`'s observation
included the opponent's live signals, which came from a still-learning
(often near-random early on) `signal_agent`, so `play_agent` was partly
learning to react to noise. Symmetrically, `signal_agent`'s round outcomes
depended on the opponent's `play_agent`, itself still learning, so
`signal_agent`'s reward signal was contaminated by a moving-target
opponent too.

So training now runs two **entirely separate kinds of games**, and every
episode is randomly one or the other (`--play-episode-prob`, default 0.5):

- **Play-training games:** every seat signals via the scripted
  **SignalBot** (deterministic heuristic — see below), and every seat
  plays via the *same, shared* `play_agent` (self-play). `signal_agent`
  is not involved at all.
- **Signal-training games:** every seat plays via the scripted
  **PlayBot** (deterministic: HIGH → highest card, LOW → lowest card),
  and every seat signals via the *same, shared* `signal_agent`
  (self-play). `play_agent` is not involved at all.

So `play_agent` only ever sees SignalBot's honest, consistent signals,
and `signal_agent`'s outcomes only ever depend on PlayBot's honest,
deterministic response — never on the other model's noisy behavior. The
two models only ever meet each other in `evaluate()`'s
`combined_model_vs_random` scenario, which puts them on the same team
together (against a fully random opponent) purely to measure how well
they cooperate once trained.

## Files

- **signal12.py** — rules engine (unchanged interface: `current_decision_maker`,
  `phase`, `legal_actions()`, `step(action)`, egocentric `get_observation(player)`).
- **bots.py** — `SignalBot` (signals HIGH if your best remaining card beats
  the strongest card played so far this round, else LOW) and `PlayBot`
  (always plays the highest/lowest remaining card per the signal it's given).
- **signalAgent.py** — `QNetwork` (single-head MLP) + `DQNAgent` (Double-DQN
  agent: acting, training, save/load) + `make_signal_agent()` /
  `make_play_agent()` convenience constructors.
- **train.py** — training loop. Each episode is either a play-training game
  or a signal-training game (see above), chosen randomly each time. Every
  `--eval-interval` episodes it reports:
  - **combined team (signal_agent + play_agent) vs. a fully random team**
    — the two models working together, the cleanest end-to-end measure
  - **play_agent vs. a fully random opponent** (isolates play skill, aided
    by SignalBot)
  - **signal_agent vs. a fully random opponent** (isolates signaling
    skill, aided by PlayBot)
  - **signal/hand-strength correlation** — does `signal_agent`'s HIGH/LOW
    track its own hand strength?
  - **play/signal correlation** — does `play_agent` actually respond to
    the signal it receives (playing higher cards on HIGH, lower on LOW)?
- **play.py** — you play Player 0 (Team A), handling both your own
  decisions (signaling for your teammate, and playing your own cards).
  Players 1, 2, and 3 are all controlled by the two trained models used
  together (the same combination `evaluate()`'s `combined_model_vs_random`
  scenario measures): `signal_agent` makes every AI signaling decision,
  `play_agent` makes every AI play decision. The SignalBot/PlayBot bots
  aren't used here — they exist only to give each model a clean training
  signal (see above); at play time, the two trained models are simply
  used together.

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
