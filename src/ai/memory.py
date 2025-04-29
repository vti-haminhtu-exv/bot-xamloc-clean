# -*- coding: utf-8 -*-
import torch
import logging
from ..game.game_rules import RANK_VALUES

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def remember(self, state, action, reward, next_state, done):
    """Lưu trải nghiệm vào memory để replay."""
    self.memory.append((self.encode_state(state), self.encode_action(action), reward,
                       self.encode_state(next_state), done))

def encode_state(self, state):
    """Mã hóa state thành tensor."""
    hand_encoding = torch.zeros(13)
    for card in state["hand"]:
        rank_index = RANK_VALUES.get(card[0], -1)
        if rank_index >= 0 and rank_index < 13:
            hand_encoding[rank_index] += 1
    hand_ranks = torch.zeros(13)
    for card in state["hand"]:
        rank_index = RANK_VALUES.get(card[0], -1)
        if rank_index >= 0 and rank_index < 13:
            hand_ranks[rank_index] = 1
    current_play_encoding = torch.zeros(13)
    for card in state["current_play"]:
        rank_index = RANK_VALUES.get(card[0], -1)
        if rank_index >= 0 and rank_index < 13:
            current_play_encoding[rank_index] += 1
    game_features = torch.tensor([
        state["opponent_cards_count"] / 10.0,
        state["player_turn"],
        state["consecutive_passes"] / 2.0,
        1.0 if state["xam_declared"] == 0 else (2.0 if state["xam_declared"] == 1 else 0.0),
        0.0 if state["last_player"] is None else state["last_player"],
        state["turn_count"] / 20.0
    ])
    has_two = torch.tensor([1.0 if any(c[0] == '2' for c in state["hand"]) else 0.0])
    hand_diff = torch.tensor([(len(state["hand"]) - state["opponent_cards_count"]) / 10.0])
    state_tensor = torch.cat((hand_encoding, hand_ranks, current_play_encoding, game_features, has_two, hand_diff))
    return state_tensor.float()

def encode_action(self, action):
    """Mã hóa action thành index."""
    if action == "pass":
        return 13
    elif action == "declare_xam":
        return 14
    else:
        if isinstance(action, list) and action:
            ranks = [RANK_VALUES.get(card[0], -1) for card in action]
            return min(ranks) if min(ranks) >= 0 and min(ranks) < 13 else 0
        return 0

def decode_action(self, action_index, valid_actions):
    """Giải mã action index thành hành động hợp lệ."""
    if action_index == 13 and "pass" in valid_actions:
        return "pass"
    elif action_index == 14 and "declare_xam" in valid_actions:
        return "declare_xam"
    for action in valid_actions:
        if isinstance(action, list) and action:
            ranks = [RANK_VALUES.get(card[0], -1) for card in action]
            if min(ranks) == action_index:
                return action
    return valid_actions[0] if valid_actions else "pass"