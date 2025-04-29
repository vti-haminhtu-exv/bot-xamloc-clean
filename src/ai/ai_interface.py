# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import logging
import os
from itertools import combinations
from ..game.game_rules import RANK_VALUES
from .ai_utils import find_straights, would_end_with_two

# Sử dụng logger chung
logger = logging.getLogger('xam_loc_solo')

def get_valid_actions(self, state, env):
    """Lấy các hành động hợp lệ từ state hiện tại."""
    hand = state["hand"]
    current_play = state["current_play"]
    valid_actions = []
    can_pass = True
    if not current_play and hand:
        can_pass = False
    elif state["last_player"] == state["player_turn"] and state["consecutive_passes"] >= 1:
        can_pass = False
    if can_pass:
        valid_actions.append("pass")
    if state["turn_count"] == 0 and state["xam_declared"] is None and not current_play:
        twos = sum(1 for c in hand if c[0] == '2')
        pairs = {}
        for card in hand:
            rank = card[0]
            pairs[rank] = pairs.get(rank, 0) + 1
        has_strong_three = any(v >= 3 and RANK_VALUES.get(k, -1) >= 9 for k, v in pairs.items())
        has_two_pairs = sum(1 for v in pairs.values() if v >= 2) >= 2
        if twos >= 2 or (twos >= 1 and has_strong_three) or (has_strong_three and has_two_pairs):
            valid_actions.append("declare_xam")
    if not hand:
        return valid_actions
    for card in hand:
        valid_actions.append([card])
    rank_counts = {}
    for card in hand:
        rank = card[0]
        if rank not in rank_counts:
            rank_counts[rank] = []
        rank_counts[rank].append(card)
    for rank, cards in rank_counts.items():
        if len(cards) >= 2:
            for combo in combinations(cards, 2):
                valid_actions.append(list(combo))
        if len(cards) >= 3:
            for combo in combinations(cards, 3):
                valid_actions.append(list(combo))
        if len(cards) == 4:
            valid_actions.append(cards)
    straights = find_straights(hand)
    valid_actions.extend(straights)
    filtered_actions = []
    for action in valid_actions:
        if action == "pass" or action == "declare_xam":
            filtered_actions.append(action)
        elif isinstance(action, list):
            try:
                if env._is_valid_play(action, hand):
                    if not would_end_with_two(hand, action):
                        filtered_actions.append(action)
            except:
                continue
    return filtered_actions

def predict_action(self, state, valid_actions, env):
    """Dự đoán hành động dựa trên state hiện tại."""
    if not valid_actions:
        return "pass"
    current_epsilon = self.epsilon * (0.95 ** (state["turn_count"] / 5))
    if random.random() < current_epsilon:
        return random.choice(valid_actions)
    return self._predict_dqn(self.encode_state(state), valid_actions)

def _predict_dqn(self, state_tensor, valid_actions):
    """Sử dụng DQN để dự đoán hành động tốt nhất."""
    with torch.no_grad():
        value, policy = self.model(state_tensor.unsqueeze(0))
        action_values = policy[0].numpy()
    action_indices = []
    for action in valid_actions:
        if action == "pass":
            action_indices.append(13)
        elif action == "declare_xam":
            action_indices.append(14)
        elif isinstance(action, list) and action:
            ranks = [RANK_VALUES.get(card[0], -1) for card in action]
            min_rank = min(ranks)
            if min_rank >= 0 and min_rank < 13:
                action_indices.append(min_rank)
    best_index = 0
    best_value = float('-inf')
    for i, idx in enumerate(action_indices):
        if idx < len(action_values) and action_values[idx] > best_value:
            best_value = action_values[idx]
            best_index = i
    return valid_actions[best_index]

def save(self, filename):
    """Lưu model và trạng thái học."""
    torch.save({
        'model_state_dict': self.model.state_dict(),
        'target_model_state_dict': self.target_model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'epsilon': self.epsilon,
        'games_played': self.games_played,
        'experience_log': self.experience_log
    }, filename)
    logger.info(f"Model saved to {filename}")
    return True

def load(self, filename, reset_stats=False, reset_epsilon=False):
    """Tải model và trạng thái học."""
    try:
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if not reset_epsilon:
            self.epsilon = checkpoint['epsilon']
        else:
            self.epsilon = 1.0
        if not reset_stats:
            self.games_played = checkpoint['games_played']
            self.experience_log = checkpoint['experience_log']
        else:
            self.games_played = 0
            self.experience_log = {
                "win_rate": {"games": 0, "wins": 0},
                "money_history": [],
                "turns_history": [],
                "pass_history": [],
                "xam_stats": {
                    "declared_ai": 0,
                    "declared_opp": 0,
                    "success_ai": 0,
                    "success_opp": 0
                },
                "timeout_games": 0,
                "logic_error_games": 0
            }
        logger.info(f"Model loaded from {filename}, reset_stats={reset_stats}, reset_epsilon={reset_epsilon}")
        return True
    except Exception as e:
        logger.error(f"Error loading model from {filename}: {e}")
        return False

def analyze_learning(self):
    """Phân tích dữ liệu học tập."""
    analysis = []
    games = self.experience_log["win_rate"]["games"]
    wins = self.experience_log["win_rate"]["wins"]
    win_rate = (wins / games * 100) if games > 0 else 0
    money_history = self.experience_log["money_history"]
    total_money = sum(money_history) if money_history else 0
    total_wins_money = sum([m for m in money_history if m > 0]) if money_history else 0
    total_losses_money = sum([m for m in money_history if m < 0]) if money_history else 0
    avg_money = total_money / len(money_history) if money_history else 0
    avg_win_money = total_wins_money / wins if wins > 0 else 0
    avg_loss_money = total_losses_money / (games - wins) if (games - wins) > 0 else 0
    xam_declared = self.experience_log["xam_stats"]["declared_ai"]
    xam_success = self.experience_log["xam_stats"]["success_ai"]
    xam_success_rate = (xam_success / xam_declared * 100) if xam_declared > 0 else 0
    turns_history = self.experience_log["turns_history"]
    avg_turns = sum(turns_history) / len(turns_history) if turns_history else 0
    analysis.append(f"Tổng số ván đã huấn luyện: {games}")
    analysis.append(f"Tỷ lệ thắng: {win_rate:.2f}% ({wins}/{games})")
    analysis.append(f"Tổng tiền thắng/thua: {total_money:.2f}")
    analysis.append(f"- Tổng tiền thắng: +{total_wins_money:.2f}")
    analysis.append(f"- Tổng tiền thua: {total_losses_money:.2f}")
    analysis.append(f"Trung bình tiền/ván: {avg_money:.2f}")
    analysis.append(f"- Trung bình tiền khi thắng: +{avg_win_money:.2f}")
    analysis.append(f"- Trung bình tiền khi thua: {avg_loss_money:.2f}")
    analysis.append(f"Epsilon hiện tại: {self.epsilon:.6f}")
    analysis.append(f"Tỷ lệ xâm thành công: {xam_success_rate:.2f}% ({xam_success}/{xam_declared})")
    analysis.append(f"Trung bình số lượt/ván: {avg_turns:.2f}")
    analysis.append(f"Số ván timeout: {self.experience_log['timeout_games']}")
    analysis.append(f"Số ván bị lỗi logic: {self.experience_log['logic_error_games']}")
    return analysis

def plot_money_history(self):
    """Vẽ biểu đồ thay đổi tiền theo thời gian."""
    try:
        import matplotlib.pyplot as plt
        money_history = self.experience_log["money_history"]
        if not money_history:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(money_history)
        plt.title("Tiền theo thời gian")
        plt.xlabel("Số ván")
        plt.ylabel("Tiền")
        plt.axhline(y=0, color='r', linestyle='-')
        plt.grid(True)
        plt.savefig("money_history.png")
        plt.close()
        cumulative_money = [sum(money_history[:i+1]) for i in range(len(money_history))]
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_money)
        plt.title("Tổng tiền tích lũy")
        plt.xlabel("Số ván")
        plt.ylabel("Tổng tiền")
        plt.axhline(y=0, color='r', linestyle='-')
        plt.grid(True)
        plt.savefig("cumulative_money.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting money history: {e}")