# -*- coding: utf-8 -*-
import numpy as np
import random
import logging
from copy import deepcopy
from itertools import combinations
from ..game.game_rules import RANK_VALUES

# Sử dụng logger chung
logger = logging.getLogger('xam_loc_solo')

class Node:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

class MCTS:
    def __init__(self, env, ai, simulations=200, c=1.5):
        self.env = env
        self.ai = ai
        self.simulations = simulations
        self.c = c
        logger.info(f"Initialized MCTS: simulations={simulations}, c={c}")

    def search(self, state, player_id, valid_actions=None):
        logger.debug("DEBUG - MCTS search: Starting with money-optimized approach")
        hand = self.env.ai_hand.copy() if player_id == 1 else self.env.player_hand.copy()
        opp_cards = self.env.player_hand.copy() if player_id == 1 else self.env.ai_hand.copy()
        opp_count = len(opp_cards)
        logger.debug(f"DEBUG - MCTS search: hand={hand}, opponent_cards_count={opp_count}")
        if valid_actions is None:
            state_copy = deepcopy(state)
            state_copy["hand"] = hand
            valid_actions = self.ai.get_valid_actions(state_copy, self.env)
        filtered_actions = []
        for action in valid_actions:
            if action == "pass" or action == "declare_xam":
                filtered_actions.append(action)
                logger.debug(f"DEBUG - MCTS search: Added {action} to filtered_actions")
            elif isinstance(action, list) and all(card in hand for card in action):
                filtered_actions.append(action)
                logger.debug(f"DEBUG - MCTS search: Added {action} to filtered_actions")
        if not filtered_actions:
            logger.warning("No valid actions after filtering, defaulting to pass")
            return "pass"
        if len(filtered_actions) == 1:
            logger.debug(f"DEBUG - MCTS search: Only one valid action: {filtered_actions[0]}")
            return filtered_actions[0]
        if "declare_xam" in filtered_actions and self._evaluate_hand_strength(hand):
            logger.debug("DEBUG - MCTS search: Strong hand for Xam, returning declare_xam")
            return "declare_xam"
        if not state["current_play"] and player_id == 1:
            straights = [a for a in filtered_actions if isinstance(a, list) and len(a) >= 3 and self._is_straight(a)]
            if straights:
                longest_straight = max(straights, key=len)
                if len(longest_straight) >= 4:
                    logger.debug(f"DEBUG - MCTS search: Found long straight, returning {longest_straight}")
                    return longest_straight
        action_stats = {str(action): {"visits": 0, "value": 0} for action in filtered_actions}
        for _ in range(self.simulations):
            action = self._select_action(action_stats, filtered_actions, opp_count)
            reward = self._evaluate_money_potential(action, hand, opp_count)
            action_key = str(action)
            action_stats[action_key]["visits"] += 1
            action_stats[action_key]["value"] += reward
        best_action = max(filtered_actions,
                         key=lambda a: action_stats[str(a)]["value"] / max(action_stats[str(a)]["visits"], 1))
        logger.debug(f"DEBUG - MCTS search: Selected best action {best_action}")
        return best_action

    def _select_action(self, action_stats, actions, opp_count):
        total_visits = sum(stats["visits"] for stats in action_stats.values())
        if total_visits < len(actions):
            unvisited = [a for a in actions if action_stats[str(a)]["visits"] == 0]
            return random.choice(unvisited) if unvisited else random.choice(actions)
        def calculate_ucb(action):
            stats = action_stats[str(action)]
            if stats["visits"] == 0:
                return float("inf")
            exploitation = stats["value"] / stats["visits"]
            exploration = self.c * np.sqrt(np.log(total_visits) / stats["visits"])
            bias = 0
            if action == "declare_xam":
                bias = 5.0
            elif isinstance(action, list):
                if opp_count <= 3 and any(card[0] in ['K', 'A', '2'] for card in action):
                    bias = 2.0
                if len(action) >= 4 and self._is_straight(action):
                    bias = 3.0
            return exploitation + exploration + bias
        return max(actions, key=calculate_ucb)

    def _evaluate_money_potential(self, action, hand, opp_count):
        if action == "pass":
            return -5
        if action == "declare_xam":
            if self._evaluate_hand_strength(hand):
                return 30
            return -10
        if isinstance(action, list):
            score = len(action) * 2
            new_hand_size = len(hand) - len(action)
            if new_hand_size <= 3:
                score += 10
            if opp_count <= 3:
                high_cards = sum(1 for c in action if c[0] in ['K', 'A', '2'])
                score += high_cards * 5
            if len(action) >= 2 and len(set(c[0] for c in action)) == 1:
                score += len(action) * 3
            if len(action) >= 3 and self._is_straight(action):
                score += 10 + len(action)
            if any(c[0] == '2' for c in action):
                if len(action) == 1:
                    score += 5
                elif len(action) >= 3 and self._is_straight(action):
                    score += 8
            return score
        return 0

    def _evaluate_hand_strength(self, hand):
        twos = sum(1 for c in hand if c[0] == '2')
        ranks = [c[0] for c in hand]
        rank_counts = {r: ranks.count(r) for r in ranks}
        trios = sum(1 for r, count in rank_counts.items() if count >= 3)
        strong_trio = any(RANK_VALUES.get(r, -1) >= RANK_VALUES['J'] for r, count in rank_counts.items() if count >= 3)
        pairs = sum(1 for r, count in rank_counts.items() if count == 2)
        high_cards = sum(1 for c in hand if RANK_VALUES.get(c[0], -1) >= RANK_VALUES['K'])
        straights = self._find_straights(hand)
        has_long_straight = any(len(straight) >= 4 for straight in straights)
        strength = (twos >= 2 or
                   (strong_trio and high_cards >= 2) or
                   (pairs >= 2 and high_cards >= 2) or
                   has_long_straight or
                   (twos >= 1 and strong_trio))
        return strength

    def _is_straight(self, cards):
        if len(cards) < 3:
            return False
        ranks_nums = sorted([RANK_VALUES.get(c[0], -1) for c in cards])
        if -1 in ranks_nums:
            return False
        has_two = any(c[0] == '2' for c in cards)
        if has_two and 13 in ranks_nums and 12 in ranks_nums:
            return False
        for i in range(len(ranks_nums) - 1):
            if ranks_nums[i + 1] != ranks_nums[i] + 1:
                return False
        return True

def find_straights(hand):
    """Tìm tất cả các sảnh có thể có trong một bộ bài."""
    straights = []
    non_two_cards = [c for c in hand if c[0] != '2']
    if len(non_two_cards) < 3:
        return []
    sorted_cards = sorted(non_two_cards, key=lambda c: RANK_VALUES.get(c[0], -1))
    for length in range(3, len(sorted_cards) + 1):
        for i in range(len(sorted_cards) - length + 1):
            candidate = sorted_cards[i:i+length]
            is_sequential = True
            for j in range(length - 1):
                curr_rank = RANK_VALUES.get(candidate[j][0], -1)
                next_rank = RANK_VALUES.get(candidate[j+1][0], -1)
                if next_rank != curr_rank + 1:
                    is_sequential = False
                    break
            if is_sequential:
                straights.append(candidate)
    return straights

def would_end_with_two(hand, play):
    """Kiểm tra xem lượt chơi này có làm hết bài và bài cuối là lá 2 không."""
    if set(play) == set(hand):
        return any(c[0] == '2' for c in play)
    return False