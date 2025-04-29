# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import logging
import numpy as np
import os
from ..model.neural_network import XamLocSoloModel
from ..game.game_rules import RANKS, SUITS

# Sử dụng logger chung
logger = logging.getLogger('xam_loc_solo')

class XamLocSoloAI:
    def __init__(self, state_size=47, action_size=15, gamma=0.95, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, learning_rate=0.001, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = XamLocSoloModel(state_size, action_size)
        self.target_model = XamLocSoloModel(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
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
        logger.info(f"AI initialized: state_size={state_size}, action_size={action_size}, epsilon={epsilon}")

    def encode_state(self, state):
        """Mã hóa trạng thái trò chơi thành vector số."""
        encoded_state = np.zeros(self.state_size, dtype=np.float32)
        ai_hand = state.get("ai_hand", [])
        for card in ai_hand:
            rank, suit = card
            rank_idx = RANKS.index(rank)
            suit_idx = SUITS.index(suit)
            card_idx = rank_idx + suit_idx * len(RANKS)
            if card_idx < 40:
                encoded_state[card_idx] = 1.0
        current_play = state.get("current_play", [])
        for card in current_play:
            rank, suit = card
            rank_idx = RANKS.index(rank)
            suit_idx = SUITS.index(suit)
            card_idx = rank_idx + suit_idx * len(RANKS)
            if card_idx < 40:
                encoded_state[card_idx] += 0.5
        idx = 40
        encoded_state[idx] = state.get("player_turn", 0)
        idx += 1
        encoded_state[idx] = state.get("consecutive_passes", 0) / 10.0
        idx += 1
        xam_declared = state.get("xam_declared", None)
        encoded_state[idx] = 1.0 if xam_declared == 1 else -1.0 if xam_declared == 0 else 0.0
        idx += 1
        last_player = state.get("last_player", None)
        encoded_state[idx] = 1.0 if last_player == 1 else -1.0 if last_player == 0 else 0.0
        idx += 1
        player_hand = state.get("player_hand", [])
        encoded_state[idx] = len(player_hand) / 10.0
        idx += 1
        if idx > self.state_size:
            logger.warning(f"Encoded state size exceeds state_size: {idx} > {self.state_size}")
            return encoded_state[:self.state_size]
        return encoded_state

    def remember(self, state, action_index, reward, next_state, done):
        """Lưu trữ trải nghiệm vào bộ nhớ sau khi mã hóa state và next_state."""
        if not isinstance(action_index, int):
            logger.error(f"action_index must be an integer, got {action_index}")
            raise ValueError(f"action_index must be an integer, got {action_index}")
        if action_index < 0 or action_index >= self.action_size:
            logger.error(f"action_index out of bounds: {action_index}, must be in [0, {self.action_size-1}]")
            raise ValueError(f"action_index out of bounds: {action_index}")
        encoded_state = self.encode_state(state)
        encoded_next_state = self.encode_state(next_state)
        self.memory.append((encoded_state, action_index, reward, encoded_next_state, done))

    def replay(self, batch_size=None):
        """Học từ những trải nghiệm đã có."""
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        end_game_samples = [mem for mem in self.memory if mem[4]]
        if end_game_samples and len(end_game_samples) > 5:
            minibatch.extend(random.sample(end_game_samples, min(5, len(end_game_samples))))
        for state, action_index, reward, next_state, done in minibatch:
            target = reward
            if not done:
                with torch.no_grad():
                    value, policy = self.model(torch.FloatTensor(next_state).unsqueeze(0))
                    action_vals = policy[0]
                    max_action_index = torch.argmax(action_vals).item()
                    next_value, next_policy = self.target_model(torch.FloatTensor(next_state).unsqueeze(0))
                    target = reward + self.gamma * next_policy[0][max_action_index].item()
            self.optimizer.zero_grad()
            value, policy = self.model(torch.FloatTensor(state).unsqueeze(0))
            action_values = policy[0]
            expected_values = action_values.clone()
            if not isinstance(action_index, int) or action_index < 0 or action_index >= self.action_size:
                logger.warning(f"Invalid action_index in replay: {action_index}, skipping")
                continue
            expected_values[action_index] = target
            loss = nn.MSELoss()(action_values, expected_values)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            if self.games_played < 1000:
                self.epsilon *= 0.998
            else:
                self.epsilon *= 0.9995

    def update_target_model(self):
        """Cập nhật target model từ model chính."""
        self.target_model.load_state_dict(self.model.state_dict())

    def get_valid_actions(self, state, env):
        """Lấy danh sách các hành động hợp lệ."""
        # Lấy tay bài của AI
        ai_hand = state.get("ai_hand", [])
        current_play = state.get("current_play", [])
        consecutive_passes = state.get("consecutive_passes", 0)
        xam_declared = state.get("xam_declared", None)
        last_player = state.get("last_player", None)

        valid_actions = []
        logger.debug(f"AI hand: {ai_hand}, Current play: {current_play}, Consecutive passes: {consecutive_passes}")

        # 1. Kiểm tra khả năng báo Xâm (chỉ được báo ở đầu ván, khi chưa ai đánh bài)
        if current_play == [] and consecutive_passes == 0 and xam_declared is None and last_player is None:
            # Đánh giá sức mạnh tay bài để quyết định báo Xâm
            # Ví dụ đơn giản: Báo Xâm nếu có ít nhất 2 lá 2 hoặc tứ quý
            two_count = sum(1 for card in ai_hand if card[0] == "2")
            has_quad = any(len([c for c in ai_hand if c[0] == rank]) >= 4 for rank in RANKS)
            if two_count >= 2 or has_quad:
                valid_actions.append("declare_xam")
                logger.debug("Added 'declare_xam' to valid actions")

        # 2. Kiểm tra khả năng bỏ lượt ("pass")
        # Có thể bỏ lượt nếu không phải lượt đầu tiên (current_play không rỗng) hoặc đã có người bỏ lượt
        if current_play or consecutive_passes > 0:
            valid_actions.append("pass")
            logger.debug("Added 'pass' to valid actions")

        # 3. Tạo các tổ hợp bài hợp lệ từ tay bài
        # 3.1. Bài lẻ
        for card in ai_hand:
            action = [card]
            if self._is_valid_play(action, current_play, state, env):
                valid_actions.append(action)
                logger.debug(f"Added single card action: {action}")

        # 3.2. Đôi
        for rank in RANKS:
            same_rank_cards = [card for card in ai_hand if card[0] == rank]
            if len(same_rank_cards) >= 2:
                action = same_rank_cards[:2]
                if self._is_valid_play(action, current_play, state, env):
                    valid_actions.append(action)
                    logger.debug(f"Added pair action: {action}")

        # 3.3. Sám (3 lá cùng số)
        for rank in RANKS:
            same_rank_cards = [card for card in ai_hand if card[0] == rank]
            if len(same_rank_cards) >= 3:
                action = same_rank_cards[:3]
                if self._is_valid_play(action, current_play, state, env):
                    valid_actions.append(action)
                    logger.debug(f"Added three-of-a-kind action: {action}")

        # 3.4. Tứ quý (4 lá cùng số)
        for rank in RANKS:
            same_rank_cards = [card for card in ai_hand if card[0] == rank]
            if len(same_rank_cards) >= 4:
                action = same_rank_cards[:4]
                if self._is_valid_play(action, current_play, state, env):
                    valid_actions.append(action)
                    logger.debug(f"Added four-of-a-kind action: {action}")

        # 3.5. Sảnh (tối thiểu 3 lá liên tiếp)
        sorted_hand = sorted(ai_hand, key=lambda x: RANKS.index(x[0]))
        current_straight = []
        last_rank_idx = -2
        for card in sorted_hand:
            rank_idx = RANKS.index(card[0])
            if rank_idx == last_rank_idx + 1:
                current_straight.append(card)
            else:
                if len(current_straight) >= 3:
                    if self._is_valid_play(current_straight, current_play, state, env):
                        valid_actions.append(current_straight[:])
                        logger.debug(f"Added straight action: {current_straight}")
                current_straight = [card]
            last_rank_idx = rank_idx
        if len(current_straight) >= 3:
            if self._is_valid_play(current_straight, current_play, state, env):
                valid_actions.append(current_straight)
                logger.debug(f"Added straight action: {current_straight}")

        # Đảm bảo luôn có ít nhất một hành động hợp lệ khi còn bài và bàn rỗng
        if not current_play and ai_hand and not any(isinstance(act, list) for act in valid_actions):
            # Thêm một lá bài lẻ ngẫu nhiên nếu không có tổ hợp nào hợp lệ
            valid_actions.append([ai_hand[0]])
            logger.debug(f"Added fallback single card action: {[ai_hand[0]]}")

        logger.debug(f"Final valid actions: {valid_actions}")
        return valid_actions

    def _is_valid_play(self, action, current_play, state, env):
        """Kiểm tra xem hành động có hợp lệ để chơi không."""
        if not action:
            logger.debug("Invalid play: Action is empty")
            return False

        # Nếu không có bài trên bàn, bất kỳ tổ hợp nào cũng hợp lệ
        if not current_play:
            logger.debug("Valid play: Table is empty, any combination is allowed")
            return True

        # Xác định loại tổ hợp của action
        if len(action) == 1:
            action_type = "single"
        elif len(action) == 2 and action[0][0] == action[1][0]:
            action_type = "pair"
        elif len(action) == 3 and all(card[0] == action[0][0] for card in action):
            action_type = "three_of_a_kind"
        elif len(action) == 4 and all(card[0] == action[0][0] for card in action):
            action_type = "four_of_a_kind"
        elif len(action) >= 3:
            # Kiểm tra sảnh
            ranks = sorted([RANKS.index(card[0]) for card in action])
            if all(ranks[i] + 1 == ranks[i + 1] for i in range(len(ranks) - 1)):
                action_type = "straight"
            else:
                logger.debug(f"Invalid play: {action} is not a straight")
                return False
        else:
            logger.debug(f"Invalid play: {action} has invalid length")
            return False

        # Xác định loại tổ hợp của current_play
        if len(current_play) == 1:
            current_type = "single"
        elif len(current_play) == 2 and current_play[0][0] == current_play[1][0]:
            current_type = "pair"
        elif len(current_play) == 3 and all(card[0] == current_play[0][0] for card in current_play):
            current_type = "three_of_a_kind"
        elif len(current_play) == 4 and all(card[0] == current_play[0][0] for card in current_play):
            current_type = "four_of_a_kind"
        elif len(current_play) >= 3:
            ranks = sorted([RANKS.index(card[0]) for card in current_play])
            if all(ranks[i] + 1 == ranks[i + 1] for i in range(len(ranks) - 1)):
                current_type = "straight"
            else:
                logger.debug(f"Invalid current play: {current_play} is not a straight")
                return False
        else:
            logger.debug(f"Invalid current play: {current_play} has invalid length")
            return False

        # So sánh tổ hợp
        if action_type != current_type:
            # Đặc biệt: Tứ quý có thể chặn lá 2
            if action_type == "four_of_a_kind" and current_type == "single" and RANKS.index(current_play[0][0]) == RANKS.index("2"):
                logger.debug("Valid play: Four of a kind beats a 2")
                return True
            logger.debug(f"Invalid play: Action type {action_type} does not match current play type {current_type}")
            return False

        # So sánh giá trị
        action_value = RANKS.index(action[-1][0])  # Lá cao nhất trong tổ hợp
        current_value = RANKS.index(current_play[-1][0])
        is_valid = action_value > current_value
        logger.debug(f"Comparing values: action_value={action_value}, current_value={current_value}, valid={is_valid}")
        return is_valid

    def action_to_index(self, action, valid_actions):
        """Chuyển đổi hành động thành chỉ số (dùng cho đầu ra của mô hình)."""
        if action == "pass":
            return self.action_size - 2  # 13 (vì action_size=15, chỉ số từ 0-14)
        elif action == "declare_xam":
            return self.action_size - 1  # 14
        else:
            if isinstance(action, list) and action:
                rank = action[0][0]
                rank_idx = RANKS.index(rank)  # 0-12 (3 đến 2)
                return rank_idx
            logger.warning(f"Invalid action format: {action}, defaulting to index 0")
            return 0  # Mặc định nếu không hợp lệ

    def _predict_dqn(self, state, valid_actions):
        """Dự đoán hành động dựa trên DQN (khai thác)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            value, policy = self.model(state_tensor)
            q_values = policy[0]
        valid_indices = [self.action_to_index(action, valid_actions) for action in valid_actions]
        valid_indices = [idx for idx in valid_indices if 0 <= idx < self.action_size]
        if not valid_indices:
            logger.warning("No valid indices found, selecting random action")
            return random.choice(valid_actions)
        q_values_valid = q_values[valid_indices]
        best_action_idx = valid_indices[torch.argmax(q_values_valid).item()]
        for i, action in enumerate(valid_actions):
            if self.action_to_index(action, valid_actions) == best_action_idx:
                return action
        return valid_actions[0]

    def predict_action(self, state, valid_actions, env):
        """Dự đoán hành động dựa trên epsilon-greedy."""
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        else:
            encoded_state = self.encode_state(state)
            return self._predict_dqn(encoded_state, valid_actions)

    def save(self, path):
        """Lưu mô hình và thống kê."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'games_played': self.games_played,
            'experience_log': self.experience_log
        }, path)

    def load(self, path, reset_stats=False, reset_epsilon=False):
        """Tải mô hình và thống kê."""
        if not os.path.exists(path):
            return False
        checkpoint = torch.load(path)
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
        return True