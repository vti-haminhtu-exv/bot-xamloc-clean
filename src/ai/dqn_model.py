# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import logging
import numpy as np
import os
import time
import copy
from ..model.neural_network import XamLocSoloModel
from ..game.game_rules import RANKS, SUITS, RANK_VALUES

# Thêm import cho các tính năng mới
from ..experience.experience_manager import ExperienceManager
from ..experience.card_probability import CardProbabilityTracker
from ..experience.pattern_memory import PatternMemory

# Sử dụng logger chung
logger = logging.getLogger('xam_loc_solo')

class XamLocSoloAI:
    def __init__(self, state_size=57, action_size=15, gamma=0.95, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, learning_rate=0.001, batch_size=32, experience_db=None):
        self.state_size = state_size  # Tăng kích thước state để thêm thông tin xác suất và pattern
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Hỗ trợ tính năng mới
        self.experience_manager = ExperienceManager(experience_db or "data/game_experience.db")
        self.card_probability_tracker = CardProbabilityTracker()
        self.pattern_memory = PatternMemory(experience_db or "data/game_experience.db")

        # Kiểm tra device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Khởi tạo model
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Khởi tạo các thuộc tính theo dõi huấn luyện
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
            "logic_error_games": 0,
            "pattern_stats": {},      # Thống kê về pattern
            "action_distribution": {} # Phân phối hành động
        }

        # Khởi tạo thống kê hành động
        self._init_action_distribution()

        logger.info(f"AI initialized with enhanced features: state_size={state_size}, action_size={action_size}, epsilon={epsilon}, device={self.device}")

    def _init_action_distribution(self):
        """Khởi tạo thống kê phân phối hành động"""
        self.experience_log["action_distribution"] = {
            "singles": {"count": 0, "success_rate": 0, "avg_reward": 0},
            "pairs": {"count": 0, "success_rate": 0, "avg_reward": 0},
            "triples": {"count": 0, "success_rate": 0, "avg_reward": 0},
            "straights": {"count": 0, "success_rate": 0, "avg_reward": 0},
            "four_kind": {"count": 0, "success_rate": 0, "avg_reward": 0},
            "passes": {"count": 0, "success_rate": 0, "avg_reward": 0},
            "xam_declared": {"count": 0, "success_rate": 0, "avg_reward": 0}
        }

    def _build_model(self):
        """Xây dựng mô hình mạng nơ-ron tốt hơn"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Thêm dropout để tránh overfitting
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def encode_state(self, state):
        """Mã hóa trạng thái trò chơi thành vector số với thông tin xác suất."""
        encoded_state = np.zeros(self.state_size, dtype=np.float32)

        # Mã hóa bài trên tay
        ai_hand = state.get("hand", [])
        for card in ai_hand:
            rank, suit = card
            rank_idx = RANKS.index(rank) if rank in RANKS else -1
            suit_idx = SUITS.index(suit) if suit in SUITS else -1
            if rank_idx >= 0 and suit_idx >= 0:
                card_idx = rank_idx + suit_idx * len(RANKS)
                if card_idx < 40:
                    encoded_state[card_idx] = 1.0

        # Mã hóa bài trên bàn
        current_play = state.get("current_play", [])
        for card in current_play:
            rank, suit = card
            rank_idx = RANKS.index(rank) if rank in RANKS else -1
            suit_idx = SUITS.index(suit) if suit in SUITS else -1
            if rank_idx >= 0 and suit_idx >= 0:
                card_idx = rank_idx + suit_idx * len(RANKS)
                if card_idx < 40:
                    encoded_state[card_idx] += 0.5

        # Mã hóa các đặc trưng trò chơi
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
        encoded_state[idx] = state.get("opponent_cards_count", 0) / 10.0
        idx += 1

        # Thêm đặc trưng: có lá 2 trong tay không
        has_two = any(c[0] == '2' for c in ai_hand)
        encoded_state[idx] = 1.0 if has_two else 0.0
        idx += 1

        # Thêm đặc trưng: chênh lệch số lá bài giữa hai người chơi
        hand_diff = len(ai_hand) - state.get("opponent_cards_count", 0)
        encoded_state[idx] = max(-1.0, min(1.0, hand_diff / 10.0))  # Chuẩn hóa vào khoảng [-1, 1]
        idx += 1

        # Bổ sung thông tin xác suất và pattern
        if hasattr(state, "game_id") and state.get("game_id"):
            # Nếu có game_id, sử dụng thông tin xác suất
            game_id = state.get("game_id")

            # Xác suất của rank
            rank_probs = self.card_probability_tracker.get_rank_probabilities(game_id)
            if len(rank_probs) == len(RANKS):
                for i, prob in enumerate(rank_probs):
                    if idx < self.state_size:
                        encoded_state[idx] = prob
                        idx += 1

        # Bổ sung thông tin pattern nếu còn chỗ
        pattern_features = self.pattern_memory.encode_active_patterns(state)
        for i, feat in enumerate(pattern_features):
            if idx + i < self.state_size:
                encoded_state[idx + i] = feat

        return torch.tensor(encoded_state, dtype=torch.float32, device=self.device)

    def remember(self, state, action, reward, next_state, done, is_timeout=False, is_logic_error=False):
        """Lưu trữ kinh nghiệm với thông tin bổ sung và điều chỉnh reward"""
        # Xử lý action nếu không phải là index
        if not isinstance(action, int):
            if isinstance(action, (list, str)):
                try:
                    valid_actions = self.get_valid_actions(state)
                    action = self.action_to_index(action, valid_actions)
                except Exception as e:
                    logger.error(f"Error converting action to index: {e}")
                    return
            else:
                logger.error(f"action must be an integer, list or string, got {type(action)}")
                return

        # Đảm bảo action_index nằm trong giới hạn
        if action < 0 or action >= self.action_size:
            logger.error(f"action_index out of bounds: {action}, must be in [0, {self.action_size-1}]")
            action = min(max(0, action), self.action_size - 1)

        # Mã hóa state và next_state
        encoded_state = self.encode_state(state)
        encoded_next_state = self.encode_state(next_state)

        # Điều chỉnh reward dựa trên pattern và heuristics
        adjusted_reward = reward

        # Xác định action thực tế từ index
        action_from_index = self.index_to_action(action, state.get("valid_actions", []))

        # Điều chỉnh reward dựa trên loại action
        if action_from_index == "pass":
            # Cập nhật thống kê
            self._update_action_stat("passes", reward, done and next_state.get("winner") == 1)

            # Phạt nhẹ cho việc bỏ lượt khi không cần thiết
            if not state.get("current_play") and state.get("hand"):
                adjusted_reward -= 1.0  # Phạt bỏ lượt khi bàn trống

        elif action_from_index == "declare_xam":
            # Cập nhật thống kê
            self._update_action_stat("xam_declared", reward, done and next_state.get("winner") == 1)

        elif isinstance(action_from_index, list) and action_from_index:
            # Bổ sung reward từ pattern
            pattern_bonus = self.pattern_memory.evaluate_pattern_match(state, action_from_index)
            adjusted_reward += pattern_bonus

            # Phân loại và cập nhật thống kê
            action_type = self._categorize_action(action_from_index)

            if len(action_from_index) == 1:
                self._update_action_stat("singles", reward, done and next_state.get("winner") == 1)

                # Phạt nhẹ việc đánh lẻ khi có cặp đôi
                ranks = [c[0] for c in state.get("hand", [])]
                if ranks.count(action_from_index[0][0]) >= 2 and not state.get("current_play"):
                    adjusted_reward -= 0.5  # Phạt đánh lẻ khi có đôi và bàn trống

            elif len(action_from_index) == 2 and action_from_index[0][0] == action_from_index[1][0]:
                self._update_action_stat("pairs", reward, done and next_state.get("winner") == 1)
                # Thưởng nhẹ cho việc đánh đôi
                adjusted_reward += 0.3

            elif len(action_from_index) == 3 and action_from_index[0][0] == action_from_index[1][0] == action_from_index[2][0]:
                self._update_action_stat("triples", reward, done and next_state.get("winner") == 1)
                # Thưởng cho việc đánh sám
                adjusted_reward += 0.5

            elif len(action_from_index) == 4 and action_from_index[0][0] == action_from_index[1][0] == action_from_index[2][0] == action_from_index[3][0]:
                self._update_action_stat("four_kind", reward, done and next_state.get("winner") == 1)
                # Thưởng đáng kể cho tứ quý
                adjusted_reward += 1.0

            elif self._is_straight(action_from_index):
                self._update_action_stat("straights", reward, done and next_state.get("winner") == 1)
                # Thưởng cho việc đánh sảnh
                adjusted_reward += 0.2 * len(action_from_index)

            # Bổ sung thưởng nếu đánh nhiều lá cùng lúc
            if len(action_from_index) >= 3:
                adjusted_reward += len(action_from_index) * 0.3

            # Thưởng/phạt dựa trên tình huống đặc biệt
            opponent_cards_count = state.get("opponent_cards_count", 0)

            # Thưởng nếu đánh bài cao khi đối thủ sắp hết bài
            if opponent_cards_count <= 3:
                high_cards = sum(1 for c in action_from_index if c[0] in ['J', 'Q', 'K', 'A', '2'])
                if high_cards > 0:
                    adjusted_reward += high_cards * 1.0

            # Phạt nếu đánh bài cao quá sớm khi đối thủ còn nhiều bài
            elif opponent_cards_count >= 7:
                highest_cards = sum(1 for c in action_from_index if c[0] in ['A', '2'])
                if highest_cards > 0 and len(state.get("hand", [])) > 5:
                    adjusted_reward -= highest_cards * 0.5

        # Lưu vào memory
        self.memory.append((encoded_state, action, adjusted_reward, encoded_next_state, done))

        # Lưu trữ kinh nghiệm qua ExperienceManager
        if hasattr(state, "game_id") and state.get("game_id"):
            self.experience_manager.record_move(
                state["game_id"],
                state["player_turn"],
                state,
                action_from_index,
                state.get("valid_actions", []),
                adjusted_reward
            )

        # Cập nhật xác suất bài của đối thủ
        if hasattr(state, "game_id") and state.get("game_id"):
            self.card_probability_tracker.update_probabilities(
                state["game_id"],
                state["player_turn"],
                action_from_index,
                state.get("current_play", [])
            )

    def _update_action_stat(self, action_type, reward, success):
        """Cập nhật thống kê cho loại hành động"""
        if action_type not in self.experience_log["action_distribution"]:
            self.experience_log["action_distribution"][action_type] = {
                "count": 0,
                "success_rate": 0,
                "avg_reward": 0
            }

        stat = self.experience_log["action_distribution"][action_type]
        old_count = stat["count"]
        old_reward_total = stat["avg_reward"] * old_count
        old_success_total = stat["success_rate"] * old_count

        stat["count"] += 1
        stat["avg_reward"] = (old_reward_total + reward) / stat["count"]
        stat["success_rate"] = (old_success_total + (1 if success else 0)) / stat["count"]

    def replay(self, batch_size=None):
        """Học từ những trải nghiệm đã có, tăng cường với pattern thành công."""
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            return

        # Lấy minibatch cơ bản
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        minibatch = [self.memory[i] for i in indices]

        # Bổ sung trải nghiệm từ pattern thành công
        successful_experiences = self.pattern_memory.sample_successful_experiences(5)
        if successful_experiences:
            minibatch.extend(successful_experiences)

        # Chuẩn bị dữ liệu huấn luyện
        states = torch.stack([s for s, _, _, _, _ in minibatch])
        next_states = torch.stack([ns for _, _, _, ns, _ in minibatch])

        # Dự đoán Q-values hiện tại và target
        with torch.no_grad():
            target_q_values = self.target_model(next_states)
        current_q_values = self.model(states)

        # Tính toán targets
        updated_q_values = current_q_values.clone()
        for idx, (_, action, reward, _, done) in enumerate(minibatch):
            if done:
                updated_q_values[idx, action] = reward
            else:
                updated_q_values[idx, action] = reward + self.gamma * torch.max(target_q_values[idx])

        # Tính loss và cập nhật model
        self.optimizer.zero_grad()
        loss = F.mse_loss(current_q_values, updated_q_values)
        loss.backward()

        # Clip gradients để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        """Cập nhật target model từ model chính."""
        self.target_model.load_state_dict(self.model.state_dict())

    def get_valid_actions(self, state, env=None):
        """Lấy danh sách các nước đi hợp lệ."""
        # Chú ý: Giữ nguyên phương thức này cho tương thích
        # Nhưng đảm bảo ghi nhận game_id vào state nếu có
        valid_actions = []

        # Lấy thông tin từ state
        hand = state.get("hand", [])
        current_play = state.get("current_play", [])
        consecutive_passes = state.get("consecutive_passes", 0)
        xam_declared = state.get("xam_declared", None)
        last_player = state.get("last_player", None)

        # Kiểm tra các nước đi hợp lệ:
        # 1. Pass
        can_pass = True
        if not current_play and hand:
            can_pass = False
        elif last_player == state.get("player_turn") and consecutive_passes >= 1:
            can_pass = False

        if can_pass:
            valid_actions.append("pass")

        # 2. Declare Xam
        if current_play == [] and consecutive_passes == 0 and xam_declared is None and last_player is None:
            # Báo Xâm chỉ ở đầu ván
            # Đánh giá bài có mạnh đủ để báo Xâm không
            two_count = sum(1 for card in hand if card[0] == "2")
            has_quad = any(len([c for c in hand if c[0] == rank]) >= 4 for rank in set(c[0] for c in hand))

            if two_count >= 2 or has_quad:
                valid_actions.append("declare_xam")

        # 3. Bài lẻ
        for card in hand:
            valid_actions.append([card])

        # 4. Đôi
        rank_groups = {}
        for card in hand:
            rank = card[0]
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(card)

        for rank, cards in rank_groups.items():
            # Đôi
            if len(cards) >= 2:
                for i in range(len(cards)):
                    for j in range(i+1, len(cards)):
                        valid_actions.append([cards[i], cards[j]])

            # Sám
            if len(cards) >= 3:
                for i in range(len(cards)):
                    for j in range(i+1, len(cards)):
                        for k in range(j+1, len(cards)):
                            valid_actions.append([cards[i], cards[j], cards[k]])

            # Tứ quý
            if len(cards) == 4:
                valid_actions.append(cards.copy())

        # 5. Sảnh (chuỗi >= 3 lá liên tiếp)
        straights = self._find_straights(hand)
        valid_actions.extend(straights)

        # Lọc các nước đi để đảm bảo hợp lệ dựa trên luật chơi
        filtered_actions = []

        for action in valid_actions:
            if action == "pass" or action == "declare_xam":
                filtered_actions.append(action)
            elif isinstance(action, list) and env:
                try:
                    if env._is_valid_play(action, hand):
                        # Không cho phép kết thúc với lá 2
                        if not self._would_end_with_two(hand, action):
                            filtered_actions.append(action)
                except Exception as e:
                    logger.error(f"Error checking valid play: {e}")
                    continue
            elif isinstance(action, list):
                # Nếu không có env, chấp nhận tất cả action là list
                filtered_actions.append(action)

        return filtered_actions

    def _find_straights(self, hand, min_length=3):
        """Tìm các sảnh có trong tay bài (độ dài >= min_length)"""
        if len(hand) < min_length:
            return []

        # Loại bỏ lá 2 khỏi sảnh
        non_two_cards = [card for card in hand if card[0] != '2']
        if len(non_two_cards) < min_length:
            return []

        # Sắp xếp theo giá trị
        sorted_cards = sorted(non_two_cards, key=lambda c: RANK_VALUES.get(c[0], -1))
        straights = []

        # Tìm các sảnh có độ dài khác nhau
        for length in range(min_length, len(sorted_cards) + 1):
            for i in range(len(sorted_cards) - length + 1):
                candidate = sorted_cards[i:i+length]
                if self._is_straight(candidate):
                    straights.append(candidate)

        return straights

    def _is_straight(self, cards):
        """Kiểm tra xem một dãy bài có tạo thành sảnh không"""
        if len(cards) < 3:
            return False

        # Sắp xếp theo giá trị rank
        sorted_cards = sorted(cards, key=lambda c: RANK_VALUES.get(c[0], -1))

        # Kiểm tra liên tục
        for i in range(len(sorted_cards) - 1):
            curr_rank = RANK_VALUES.get(sorted_cards[i][0], -1)
            next_rank = RANK_VALUES.get(sorted_cards[i+1][0], -1)

            if next_rank != curr_rank + 1:
                return False

        return True

    def _would_end_with_two(self, hand, play):
        """Kiểm tra xem lượt chơi này có làm hết bài và bài cuối là lá 2 không"""
        if set(play) == set(hand):
            return any(c[0] == '2' for c in play)
        return False

    def action_to_index(self, action, valid_actions):
        """Chuyển đổi hành động thành chỉ số (dùng cho đầu ra của mô hình)."""
        if action == "pass":
            return self.action_size - 2  # 13 (vì action_size=15, chỉ số từ 0-14)
        elif action == "declare_xam":
            return self.action_size - 1  # 14
        else:
            if isinstance(action, list) and action:
                rank = action[0][0]
                rank_idx = RANKS.index(rank) if rank in RANKS else 0  # 0-12 (3 đến 2)
                return rank_idx
            logger.warning(f"Invalid action format: {action}, defaulting to index 0")
            return 0  # Mặc định nếu không hợp lệ

    def index_to_action(self, index, valid_actions):
        """Chuyển đổi chỉ số hành động thành hành động thực tế"""
        if index == self.action_size - 2 and "pass" in valid_actions:  # 13
            return "pass"
        elif index == self.action_size - 1 and "declare_xam" in valid_actions:  # 14
            return "declare_xam"

        # Tìm hành động tương ứng với index
        target_rank = RANKS[index] if 0 <= index < len(RANKS) else None

        if target_rank:
            # Tìm kiếm các hành động có rank trùng khớp
            matching_actions = [a for a in valid_actions if isinstance(a, list)
                             and a and a[0][0] == target_rank]

            if matching_actions:
                # Ưu tiên theo thứ tự: tứ quý > sám > đôi > lẻ
                # Đối với cùng loại, ưu tiên bài thấp hơn
                for length in [4, 3, 2, 1]:
                    for action in matching_actions:
                        if len(action) == length:
                            return action

        # Nếu không tìm thấy, trả về hành động đầu tiên hợp lệ
        if valid_actions:
            return valid_actions[0]

        # Trường hợp không có hành động hợp lệ
        return "pass"

    def _predict_dqn(self, state_tensor, valid_actions):
        """Sử dụng DQN để dự đoán hành động tốt nhất."""
        with torch.no_grad():
            q_values = self.model(state_tensor.unsqueeze(0))[0].cpu().numpy()

        # Lấy các chỉ số hành động hợp lệ
        valid_indices = []
        for action in valid_actions:
            idx = self.action_to_index(action, valid_actions)
            if 0 <= idx < self.action_size:
                valid_indices.append(idx)

        # Nếu không có chỉ số hợp lệ
        if not valid_indices:
            return random.choice(valid_actions)

        # Chọn hành động có q-value cao nhất
        best_idx = valid_indices[np.argmax(q_values[valid_indices])]
        for action in valid_actions:
            if self.action_to_index(action, valid_actions) == best_idx:
                return action

        # Fallback nếu có lỗi
        return valid_actions[0]

    def predict_action(self, state, valid_actions, env=None):
        """Dự đoán hành động dựa trên trạng thái hiện tại."""
        if not valid_actions:
            logger.warning("No valid actions provided")
            return "pass"

        # Thêm game_id vào state nếu chưa có
        if not hasattr(state, "game_id") and state.get("turn_count", 0) == 0:
            game_id = self.experience_manager.start_new_game()
            state["game_id"] = game_id

            # Khởi tạo xác suất bài
            if state.get("player_turn") == 1:  # AI's turn
                self.card_probability_tracker.initialize_game(game_id, state.get("hand", []))

        # Kiểm tra explore hay exploit
        if np.random.rand() <= self.epsilon:
            # Exploration: 70% thời gian sử dụng heuristic thông minh, 30% hoàn toàn ngẫu nhiên
            if np.random.rand() < 0.7 and env is not None:
                return self._smart_heuristic_choice(state, valid_actions, env)
            else:
                return random
    def predict_action(self, state, valid_actions, env=None):
           """Dự đoán hành động dựa trên trạng thái hiện tại."""
           if not valid_actions:
               logger.warning("No valid actions provided")
               return "pass"

           # Thêm game_id vào state nếu chưa có
           if not hasattr(state, "game_id") and state.get("turn_count", 0) == 0:
               game_id = self.experience_manager.start_new_game()
               state["game_id"] = game_id

               # Khởi tạo xác suất bài
               if state.get("player_turn") == 1:  # AI's turn
                   self.card_probability_tracker.initialize_game(game_id, state.get("hand", []))

           # Kiểm tra explore hay exploit
           if np.random.rand() <= self.epsilon:
               # Exploration: 70% thời gian sử dụng heuristic thông minh, 30% hoàn toàn ngẫu nhiên
               if np.random.rand() < 0.7 and env is not None:
                   return self._smart_heuristic_choice(state, valid_actions, env)
               else:
                   return random.choice(valid_actions)
           else:
               # Exploitation
               # Thử MCTS nếu có đủ thời gian/tài nguyên
               use_mcts = np.random.rand() < 0.3 and env is not None  # 30% cơ hội dùng MCTS

               if use_mcts:
                   mcts_action = self._predict_mcts(state, valid_actions, env)
                   if mcts_action is not None:
                       return mcts_action

               # Mặc định sử dụng DQN
               encoded_state = self.encode_state(state)
               return self._predict_dqn(encoded_state, valid_actions)

       def _smart_heuristic_choice(self, state, valid_actions, env):
           """Lựa chọn thông minh dựa trên heuristics"""
           # Tạo một bản sao của valid_actions để không làm thay đổi nó
           actions = valid_actions.copy()

           # Nếu không có actions hợp lệ
           if not actions:
               return "pass"

           # Lấy thông tin từ state
           hand = state.get("hand", [])
           current_play = state.get("current_play", [])
           opponent_cards_count = state.get("opponent_cards_count", 0)

           # Chiến lược 1: Nếu có thể thắng trong một nước, đánh luôn
           for action in actions:
               if isinstance(action, list) and len(action) == len(hand):
                   return action

           # Chiến lược 2: Nếu đối thủ sắp hết bài (≤ 3 lá), ưu tiên đánh cao
           if opponent_cards_count <= 3:
               # Tìm các lá cao (A, 2, K) để chặn
               high_actions = []
               for action in actions:
                   if isinstance(action, list):
                       has_high = any(c[0] in ['K', 'A', '2'] for c in action)
                       if has_high:
                           high_actions.append((action, max(RANK_VALUES.get(c[0], 0) for c in action)))

               if high_actions:
                   # Sắp xếp theo giá trị giảm dần
                   high_actions.sort(key=lambda x: x[1], reverse=True)
                   return high_actions[0][0]

           # Chiến lược 3: Nếu bàn trống, ưu tiên đánh đôi thấp hoặc sảnh
           if not current_play:
               # Tìm tất cả các đôi
               pairs = []
               for action in actions:
                   if isinstance(action, list) and len(action) == 2 and action[0][0] == action[1][0]:
                       pairs.append((action, RANK_VALUES.get(action[0][0], 0)))

               # Nếu có đôi, đánh đôi thấp
               if pairs:
                   pairs.sort(key=lambda x: x[1])  # Sắp xếp tăng dần
                   return pairs[0][0]

               # Tìm tất cả sảnh
               straights = []
               for action in actions:
                   if isinstance(action, list) and len(action) >= 3 and self._is_straight(action):
                       straights.append((action, len(action)))

               # Nếu có sảnh, ưu tiên sảnh dài
               if straights:
                   straights.sort(key=lambda x: x[1], reverse=True)
                   return straights[0][0]

           # Chiến lược 4: Nếu bàn có bài, đánh bài nhỏ nhất có thể chặn được
           if current_play:
               beatable_actions = []
               for action in actions:
                   if action != "pass" and isinstance(action, list):
                       beatable_actions.append((action, max(RANK_VALUES.get(c[0], 0) for c in action)))

               if beatable_actions:
                   # Sắp xếp theo giá trị tăng dần
                   beatable_actions.sort(key=lambda x: x[1])
                   return beatable_actions[0][0]

           # Chiến lược 5: Nếu không có chiến lược nào khác, bỏ lượt (nếu có thể) hoặc đánh lẻ thấp nhất
           if "pass" in actions:
               return "pass"

           # Tìm lá đơn thấp nhất
           singles = []
           for action in actions:
               if isinstance(action, list) and len(action) == 1:
                   singles.append((action, RANK_VALUES.get(action[0][0], 0)))

           if singles:
               singles.sort(key=lambda x: x[1])
               return singles[0][0]

           # Nếu không có lựa chọn nào khác, chọn ngẫu nhiên
           return random.choice(actions)

       def _predict_mcts(self, state, valid_actions, env):
           """Sử dụng MCTS cải tiến để dự đoán hành động"""
           from ..ai.ai_utils import MCTS

           try:
               # Tạo bản sao của môi trường
               mcts = MCTS(
                   env=env,
                   ai=self,
                   simulations=500,  # Tăng số lần mô phỏng
                   c=1.5  # Tham số điều chỉnh giữa khai thác và khám phá
               )

               # Thực hiện tìm kiếm MCTS
               return mcts.search(state, state["player_turn"], valid_actions)
           except Exception as e:
               logger.error(f"Error in MCTS: {e}")
               return None

       def _categorize_action(self, action):
           """Phân loại action thành các loại như single, pair, etc."""
           if not action or not isinstance(action, list):
               return "unknown"

           if len(action) == 1:
               return "single"
           elif len(action) == 2 and action[0][0] == action[1][0]:
               return "pair"
           elif len(action) == 3 and action[0][0] == action[1][0] == action[2][0]:
               return "triple"
           elif len(action) == 4 and action[0][0] == action[1][0] == action[2][0] == action[3][0]:
               return "four_of_a_kind"
           elif self._is_straight(action):
               return f"straight_{len(action)}"
           return "unknown"

       def save(self, filename):
           """Lưu model và trạng thái học."""
           save_dir = os.path.dirname(filename)
           if save_dir and not os.path.exists(save_dir):
               os.makedirs(save_dir)

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
               if not os.path.exists(filename):
                   logger.error(f"Model file {filename} not found")
                   return False

               checkpoint = torch.load(filename, map_location=self.device)
               self.model.load_state_dict(checkpoint['model_state_dict'])
               self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
               self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

               if not reset_epsilon:
                   self.epsilon = checkpoint['epsilon']
               else:
                   self.epsilon = 1.0

               if not reset_stats:
                   self.games_played = checkpoint['games_played']
                   if 'experience_log' in checkpoint:
                       self.experience_log = checkpoint['experience_log']
               else:
                   self.games_played = 0
                   self._reset_experience_log()

               logger.info(f"Model loaded from {filename}, reset_stats={reset_stats}, reset_epsilon={reset_epsilon}")
               return True
           except Exception as e:
               logger.error(f"Error loading model from {filename}: {e}")
               return False

       def _reset_experience_log(self):
           """Reset log kinh nghiệm về trạng thái mặc định"""
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
               "logic_error_games": 0,
               "pattern_stats": {},
               "action_distribution": {}
           }
           self._init_action_distribution()

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

           # Phân tích cơ bản
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

           # Phân tích phân phối hành động
           if 'action_distribution' in self.experience_log:
               action_dist = self.experience_log['action_distribution']
               analysis.append("\nPhân phối hành động:")

               for action_type, stats in action_dist.items():
                   if stats['count'] > 0:
                       analysis.append(f"- {action_type.capitalize()}: {stats['count']} lần, "
                                      f"tỷ lệ thành công: {stats['success_rate']*100:.1f}%, "
                                      f"reward trung bình: {stats['avg_reward']:.2f}")

           # Phân tích pattern
           if 'pattern_stats' in self.experience_log:
               pattern_stats = self.experience_log['pattern_stats']
               if pattern_stats:
                   analysis.append("\nThống kê pattern:")
                   for pattern_type, stats in pattern_stats.items():
                       if isinstance(stats, dict) and 'count' in stats and stats['count'] > 0:
                           analysis.append(f"- {pattern_type}: {stats['count']} lần, "
                                         f"hiệu quả: {stats.get('effectiveness', 0):.2f}")

           # Phân tích theo thời gian (50 ván gần nhất)
           if money_history and len(money_history) >= 50:
               recent_money = money_history[-50:]
               recent_avg = sum(recent_money) / len(recent_money)
               recent_win_rate = sum(1 for m in recent_money if m > 0) / len(recent_money) * 100

               analysis.append(f"\nKết quả 50 ván gần nhất:")
               analysis.append(f"- Tỷ lệ thắng: {recent_win_rate:.2f}%")
               analysis.append(f"- Trung bình tiền/ván: {recent_avg:.2f}")

               # So sánh với 50 ván đầu tiên
               if len(money_history) >= 100:
                   first_money = money_history[:50]
                   first_avg = sum(first_money) / len(first_money)
                   first_win_rate = sum(1 for m in first_money if m > 0) / len(first_money) * 100

                   win_rate_change = recent_win_rate - first_win_rate
                   avg_change = recent_avg - first_avg

                   analysis.append(f"- Thay đổi tỷ lệ thắng: {'+' if win_rate_change >= 0 else ''}{win_rate_change:.2f}%")
                   analysis.append(f"- Thay đổi trung bình tiền: {'+' if avg_change >= 0 else ''}{avg_change:.2f}")

           return analysis

       def plot_money_history(self):
           """Vẽ biểu đồ thay đổi tiền theo thời gian."""
           try:
               import matplotlib.pyplot as plt

               money_history = self.experience_log["money_history"]
               if not money_history:
                   logger.warning("No money history to plot")
                   return

               plt.figure(figsize=(10, 6))
               plt.plot(money_history)
               plt.title("Tiền theo thời gian")
               plt.xlabel("Số ván")
               plt.ylabel("Tiền")
               plt.axhline(y=0, color='r', linestyle='-')
               plt.grid(True)
               plt.savefig("data/money_history.png")
               plt.close()

               # Biểu đồ tích lũy
               cumulative_money = [sum(money_history[:i+1]) for i in range(len(money_history))]
               plt.figure(figsize=(10, 6))
               plt.plot(cumulative_money)
               plt.title("Tổng tiền tích lũy")
               plt.xlabel("Số ván")
               plt.ylabel("Tổng tiền")
               plt.axhline(y=0, color='r', linestyle='-')
               plt.grid(True)
               plt.savefig("data/cumulative_money.png")
               plt.close()

               # Biểu đồ cửa sổ trượt
               if len(money_history) >= 20:
                   window_size = 20
                   moving_avg = [sum(money_history[i:i+window_size])/window_size
                              for i in range(len(money_history)-window_size+1)]
                   plt.figure(figsize=(10, 6))
                   plt.plot(range(window_size-1, len(money_history)), moving_avg)
                   plt.title(f"Trung bình trượt {window_size} ván")
                   plt.xlabel("Số ván")
                   plt.ylabel(f"Tiền trung bình ({window_size} ván)")
                   plt.axhline(y=0, color='r', linestyle='-')
                   plt.grid(True)
                   plt.savefig("data/moving_avg_money.png")
                   plt.close()

               logger.info("Money history plots saved to data/")
           except Exception as e:
               logger.error(f"Error plotting money history: {e}")