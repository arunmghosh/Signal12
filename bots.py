"""
bots.py

Scripted (non-learning) heuristic players used to give each learned model a
clean, low-variance partner/opponent to train against.

  SignalBot   - decides HIGH/LOW purely from observation: the player's own
                remaining hand and the cards already played this round.
                Deterministic and stateless (no learning).

  PlayBot     - given a HIGH/LOW signal, always plays deterministically:
                HIGH -> play the highest card unless there is no chance of winning
                LOW  -> play the lowest card remaining in hand
                Never deviates from the signal, by design - this is what
                makes it a clean "receiver" for training the signal model,
                and a clean, unsurprising "signal source" for training the
                play model when used as a signaler proxy in reverse.
"""

from __future__ import annotations
from typing import List, Optional, Union, Dict
from signal12 import SIGNAL_HIGH, SIGNAL_LOW, TEAMMATE
import random


def get_playable_opp(hand: List[int], center: Union[List[int], Dict[int, int]], unplayed_opp: List[int], 
                     player_num: int) -> List[int]:
    playable_opp = list()  # cards the bot needs to beat to win round
    if isinstance(center, dict):
        center_cards = list(center.values())
    else:
        center_cards = list(center)
    for c in center_cards:
        playable_opp.append(c)  # need to beat every card in the center
    
    if player_num == 0:  
        # opponents could play anything they haven't already played
        for u in unplayed_opp:
            playable_opp.append(u)
    elif player_num < 3:  
        # we know only one opponent can still play in the round
        # guess what cards that opponent has
        guess_pool = list(unplayed_opp)
        for i in range(min(len(hand), len(guess_pool))):
            guess = random.choice(guess_pool)
            playable_opp.append(guess)
            guess_pool.remove(guess)
    return playable_opp

class SignalBot:
    """Heuristic signaler: HIGH if the player can win the round, LOW if signaling player
    can win or knows opponent is going to win."""

    def act(self, hand: List[int], teammate_hand: List[int], center: Union[List[int], Dict[int, int]], 
            unplayed_opp: List[int], player_num: int) -> int:
        # hand is cards this bot can play
        # teammate hand is cards their teammate can play
        # center is the cards already played in the round
        # unplayed_opp is the cards the opposing team has yet to play
        # player_num is the index of this bot in this round's play order (0-3)

        teammate_ind = TEAMMATE[player_num] if isinstance(TEAMMATE, dict) else (player_num + 2) % 4
        teammate_high = max(teammate_hand) if teammate_hand else 0
        your_high = max(hand) if hand else 0
        playable_opp = get_playable_opp(hand, center, unplayed_opp, teammate_ind)
        opp_high = max(playable_opp) if playable_opp else 0

        if teammate_ind < 2:  # teammate is first to play for the team
            # Check if you can win
            if your_high > opp_high:
                # If so, tell your teammate to play low
                return SIGNAL_LOW

        # Either your teammate is second to play on the team or you can't win anyway
        # Since you can't win or can't play, check if your teammate can win
        if teammate_high > opp_high:
            # If so, tell them to win the round
            return SIGNAL_HIGH

        # If we know opponent will win, play low
        return SIGNAL_LOW

class PlayBot:
    """Deterministic play-chooser: always obeys the given signal by playing
    the highest (HIGH) or lowest (LOW) card remaining in hand."""

    def act(self, hand: List[int], center: Union[List[int], Dict[int, int]], unplayed_opp: List[int], 
            player_num: int, signal: Optional[int]) -> int:
        # hand is cards this bot can play
        # center is the cards already played in the round
        # unplayed_opp is the cards the opposing team has yet to play
        # player_num is the index of this bot in this round's play order (0-3)
        # signal is 0 or 1 (SIGNAL_LOW or SIGNAL_HIGH)

        if not hand:
            raise ValueError("PlayBot called with an empty hand.")
        if signal == SIGNAL_LOW:
            target = min(hand)
        else:
            # Treat HIGH, or a missing/unknown signal, the same way: HIGH.
            highest = max(hand)
            playable_opp = get_playable_opp(hand, center, unplayed_opp, player_num)

            # Check 1: no chance that the bot can win this round
            benchmark = max(playable_opp) if playable_opp else 0
            if highest < benchmark:
                target = min(hand)
            else:
                # Check 2: multiple cards are guaranteed to win
                clone_hand = list(hand)
                target = highest
                while clone_hand:
                    test_target = min(clone_hand)
                    if test_target < benchmark:
                        clone_hand.remove(test_target)
                    else:
                        target = test_target
                        break

        return hand.index(target)
