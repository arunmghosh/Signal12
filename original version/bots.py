"""
bots.py

Scripted (non-learning) heuristic players used to give each learned model a
clean, low-variance partner/opponent to train against.

  SignalBot   - decides HIGH/LOW purely from observation: the player's own
                remaining hand and the cards already played this round.
                Deterministic and stateless (no learning).

  PlayBot     - given a HIGH/LOW signal, always plays deterministically:
                HIGH -> play the highest card remaining in hand
                LOW  -> play the lowest card remaining in hand
                Never deviates from the signal, by design - this is what
                makes it a clean "receiver" for training the signal model,
                and a clean, unsurprising "signal source" for training the
                play model when used as a signaler proxy in reverse.
"""

from __future__ import annotations

from typing import List, Optional

from signal12 import Signal12Game, SIGNAL_HIGH, SIGNAL_LOW, TEAMMATE


class SignalBot:
    """Heuristic signaler: HIGH if the player can win the round, LOW if signaling player
    can win or knows opponent is going to win."""

    def act(self, game: Signal12Game, player: int) -> int:
        cards_left = game.cards_left
        high_card = max(cards_left)
        hand = game.hands[player]

        if game.turn_index == 0:
            # no cards played yet
            best = max(hand)
            if best == high_card:  # signaling player can catch the round on their turn
                return SIGNAL_LOW
            return SIGNAL_HIGH
        
        played_values = list(game.played.values())  # non-empty if game index isn't 0
        threshold = max(played_values)
        
        if game.turn_index < 3:
            # don't know what second opponent player will do
            # signal high if there is a chance to win
            if high_card > threshold:
                return SIGNAL_HIGH
            return SIGNAL_LOW

        # last turn for the round, player should play high if they can win
        teammate_hand = game.hands[TEAMMATE[player]]
        if max(teammate_hand) > threshold:
            return SIGNAL_HIGH
        return SIGNAL_LOW

class PlayBot:
    """Deterministic play-chooser: always obeys the given signal by playing
    the highest (HIGH) or lowest (LOW) card remaining in hand."""

    def act(self, hand: List[int], signal: Optional[int]) -> int:
        if not hand:
            raise ValueError("PlayBot called with an empty hand.")
        if signal == SIGNAL_HIGH:
            target = max(hand)
        else:
            # Treat LOW, or a missing/unknown signal, the same way: LOW.
            target = min(hand)
        return hand.index(target)
