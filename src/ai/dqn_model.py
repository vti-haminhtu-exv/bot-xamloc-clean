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
from ..model.neural_network import XamLocSoloModel # Giả sử đường dẫn này đúng
from ..game.game_rules import RANKS, SUITS, RANK_VALUES # Giả sử đường dẫn này đúng

# Thêm import cho các tính năng mới
from ..experience.experience_manager import ExperienceManager # Giả sử đường dẫn này đúng
from ..experience.card_probability import CardProbabilityTracker # Giả sử đường dẫn này đúng
from ..experience.pattern_memory import PatternMemory # Giả sử đường dẫn này đúng

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
        db_path = experience_db or "data/game_experience.db"
        # Đảm bảo thư mục data tồn tại
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        self.experience_manager = ExperienceManager(db_path)
        self.card_probability_tracker = CardProbabilityTracker()
        self.pattern_memory = PatternMemory(db_path)

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
                if card_idx < 40: # Chỉ mã hóa 40 lá (không phải 52)
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
                    # Giá trị khác 1 để phân biệt với bài trên tay
                    encoded_state[card_idx] += 0.5

        # Mã hóa các đặc trưng trò chơi
        idx = 40 # Bắt đầu sau 40 vị trí cho lá bài
        if idx < self.state_size:
            encoded_state[idx] = state.get("player_turn", 0) # AI là 1, đối thủ là 0
            idx += 1
        if idx < self.state_size:
            encoded_state[idx] = state.get("consecutive_passes", 0) / 10.0 # Chuẩn hóa
            idx += 1
        if idx < self.state_size:
            xam_declared = state.get("xam_declared", None)
            # 1.0 nếu AI báo xâm, -1.0 nếu đối thủ báo xâm, 0.0 nếu chưa ai báo
            encoded_state[idx] = 1.0 if xam_declared == 1 else -1.0 if xam_declared == 0 else 0.0
            idx += 1
        if idx < self.state_size:
            last_player = state.get("last_player", None)
             # 1.0 nếu AI đánh cuối, -1.0 nếu đối thủ đánh cuối, 0.0 nếu chưa ai đánh
            encoded_state[idx] = 1.0 if last_player == 1 else -1.0 if last_player == 0 else 0.0
            idx += 1
        if idx < self.state_size:
            encoded_state[idx] = state.get("opponent_cards_count", 0) / 10.0 # Chuẩn hóa
            idx += 1

        # Thêm đặc trưng: có lá 2 trong tay không
        if idx < self.state_size:
            has_two = any(c[0] == '2' for c in ai_hand)
            encoded_state[idx] = 1.0 if has_two else 0.0
            idx += 1

        # Thêm đặc trưng: chênh lệch số lá bài giữa hai người chơi
        if idx < self.state_size:
            hand_diff = len(ai_hand) - state.get("opponent_cards_count", 0)
            encoded_state[idx] = max(-1.0, min(1.0, hand_diff / 10.0))  # Chuẩn hóa vào khoảng [-1, 1]
            idx += 1

        # Bổ sung thông tin xác suất và pattern
        # Kiểm tra xem state có thuộc tính 'game_id' không và giá trị có hợp lệ không
        game_id = state.get("game_id") if isinstance(state, dict) and "game_id" in state else None

        if game_id:
            # Xác suất của rank
            rank_probs = self.card_probability_tracker.get_rank_probabilities(game_id)
            if len(rank_probs) == len(RANKS):
                for i, prob in enumerate(rank_probs):
                    if idx < self.state_size:
                        encoded_state[idx] = prob
                        idx += 1
            else:
                 # Nếu không lấy được xác suất, điền giá trị mặc định (ví dụ 0)
                for _ in range(len(RANKS)):
                    if idx < self.state_size:
                        encoded_state[idx] = 0.0
                        idx += 1


            # Bổ sung thông tin pattern nếu còn chỗ
            pattern_features = self.pattern_memory.encode_active_patterns(state)
            remaining_slots = self.state_size - idx
            num_features_to_add = min(len(pattern_features), remaining_slots)
            for i in range(num_features_to_add):
                 encoded_state[idx + i] = pattern_features[i]
            idx += num_features_to_add

        # Điền 0 cho các vị trí còn lại nếu state_size lớn hơn số đặc trưng đã điền
        while idx < self.state_size:
            encoded_state[idx] = 0.0
            idx += 1

        # Đảm bảo vector cuối cùng có đúng self.state_size phần tử
        if len(encoded_state) != self.state_size:
             logger.warning(f"Encoded state length mismatch: expected {self.state_size}, got {len(encoded_state)}. Padding/truncating.")
             final_state = np.zeros(self.state_size, dtype=np.float32)
             copy_len = min(len(encoded_state), self.state_size)
             final_state[:copy_len] = encoded_state[:copy_len]
             encoded_state = final_state


        return torch.tensor(encoded_state, dtype=torch.float32, device=self.device)


    def remember(self, state, action, reward, next_state, done, is_timeout=False, is_logic_error=False):
        """Lưu trữ kinh nghiệm với thông tin bổ sung và điều chỉnh reward"""
        # Chuyển action thành index nếu cần
        action_index = -1
        action_details = None # Lưu action thực tế (list, "pass", "declare_xam")

        if isinstance(action, int):
             action_index = action
             # Cố gắng lấy action thực tế từ index (cần valid_actions)
             valid_actions = state.get("valid_actions", self.get_valid_actions(state)) # Lấy hoặc tái tạo valid_actions
             action_details = self.index_to_action(action_index, valid_actions)
        elif isinstance(action, (list, str)):
             action_details = action
             try:
                 valid_actions = state.get("valid_actions", self.get_valid_actions(state))
                 action_index = self.action_to_index(action, valid_actions)
             except Exception as e:
                 logger.error(f"Error converting action to index: {action}. Error: {e}")
                 # Không thể chuyển đổi, không lưu kinh nghiệm này
                 return
        else:
             logger.error(f"Invalid action type: {type(action)}. Action: {action}")
             return # Bỏ qua kinh nghiệm không hợp lệ

        # Đảm bảo action_index nằm trong giới hạn
        if not (0 <= action_index < self.action_size):
            logger.error(f"action_index out of bounds: {action_index}, must be in [0, {self.action_size-1}]")
            # Cố gắng sửa lỗi hoặc bỏ qua
            # Ví dụ: chọn index hợp lệ gần nhất hoặc bỏ qua
            # Ở đây ta sẽ bỏ qua để tránh dữ liệu lỗi
            logger.warning(f"Skipping experience due to invalid action index for action: {action_details}")
            return


        # Mã hóa state và next_state
        try:
            encoded_state = self.encode_state(state)
            encoded_next_state = self.encode_state(next_state)
        except Exception as e:
            logger.error(f"Error encoding state/next_state: {e}")
            return # Bỏ qua nếu không mã hóa được


        # Điều chỉnh reward dựa trên pattern và heuristics
        adjusted_reward = float(reward) # Đảm bảo reward là float

        # Điều chỉnh reward dựa trên loại action
        if action_details == "pass":
            self._update_action_stat("passes", adjusted_reward, done and next_state.get("winner") == 1)
            if not state.get("current_play") and state.get("hand"):
                adjusted_reward -= 1.0  # Phạt bỏ lượt khi bàn trống

        elif action_details == "declare_xam":
            self._update_action_stat("xam_declared", adjusted_reward, done and next_state.get("winner") == 1)

        elif isinstance(action_details, list) and action_details:
            # Bổ sung reward từ pattern
            pattern_bonus = self.pattern_memory.evaluate_pattern_match(state, action_details)
            adjusted_reward += pattern_bonus

            # Phân loại và cập nhật thống kê
            action_type = self._categorize_action(action_details)
            is_win_move = done and next_state.get("winner") == 1

            if action_type == "single":
                self._update_action_stat("singles", adjusted_reward, is_win_move)
                # Phạt nhẹ việc đánh lẻ khi có cặp đôi và bàn trống
                ranks = [c[0] for c in state.get("hand", [])]
                if ranks.count(action_details[0][0]) >= 2 and not state.get("current_play"):
                    adjusted_reward -= 0.5

            elif action_type == "pair":
                self._update_action_stat("pairs", adjusted_reward, is_win_move)
                adjusted_reward += 0.3 # Thưởng nhẹ

            elif action_type == "triple":
                self._update_action_stat("triples", adjusted_reward, is_win_move)
                adjusted_reward += 0.5 # Thưởng

            elif action_type == "four_of_a_kind":
                 self._update_action_stat("four_kind", adjusted_reward, is_win_move)
                 adjusted_reward += 1.0 # Thưởng lớn

            elif action_type.startswith("straight"):
                 self._update_action_stat("straights", adjusted_reward, is_win_move)
                 adjusted_reward += 0.2 * len(action_details) # Thưởng theo độ dài

            # Thưởng/phạt dựa trên tình huống đặc biệt
            opponent_cards_count = state.get("opponent_cards_count", 0)
            ai_hand_size = len(state.get("hand", []))

            # Thưởng nếu đánh nhiều lá cùng lúc
            if len(action_details) >= 3:
                 adjusted_reward += len(action_details) * 0.1 # Thưởng nhẹ thêm

            # Thưởng nếu đánh bài cao khi đối thủ sắp hết bài
            if opponent_cards_count <= 3:
                high_cards = sum(1 for c in action_details if c[0] in ['J', 'Q', 'K', 'A', '2'])
                if high_cards > 0:
                    adjusted_reward += high_cards * 0.5 # Điều chỉnh thưởng

            # Phạt nếu đánh bài cao (A, 2) quá sớm khi đối thủ còn nhiều bài (>6) và mình còn nhiều bài (>5)
            elif opponent_cards_count >= 7 and ai_hand_size > 5:
                highest_cards = sum(1 for c in action_details if c[0] in ['A', '2'])
                if highest_cards > 0:
                    adjusted_reward -= highest_cards * 0.3 # Điều chỉnh phạt

        # Lưu vào memory
        self.memory.append((encoded_state, action_index, adjusted_reward, encoded_next_state, done))

        # Lưu trữ kinh nghiệm qua ExperienceManager
        game_id = state.get("game_id") if isinstance(state, dict) and "game_id" in state else None
        if game_id:
            self.experience_manager.record_move(
                game_id,
                state.get("player_turn", -1), # Cần player_turn trong state
                state,
                action_details, # Lưu action thực tế
                state.get("valid_actions", []), # Cần valid_actions trong state
                adjusted_reward
            )

            # Cập nhật xác suất bài của đối thủ
            self.card_probability_tracker.update_probabilities(
                game_id,
                state.get("player_turn", -1), # Cần player_turn
                action_details, # Action thực tế
                state.get("current_play", []) # Cần current_play
            )

    def _update_action_stat(self, action_type, reward, success):
        """Cập nhật thống kê cho loại hành động"""
        if action_type not in self.experience_log["action_distribution"]:
            # Khởi tạo nếu chưa có
            self._init_action_distribution()
            if action_type not in self.experience_log["action_distribution"]:
                 logger.warning(f"Action type '{action_type}' not found in distribution after init.")
                 return # Bỏ qua nếu vẫn không tìm thấy


        stat = self.experience_log["action_distribution"][action_type]
        old_count = stat["count"]
        # Xử lý trường hợp chia cho 0
        old_reward_total = stat["avg_reward"] * old_count if old_count > 0 else 0
        old_success_total = stat["success_rate"] * old_count if old_count > 0 else 0


        stat["count"] += 1
        # Tính lại giá trị trung bình một cách an toàn
        stat["avg_reward"] = (old_reward_total + float(reward)) / stat["count"]
        stat["success_rate"] = (old_success_total + (1 if success else 0)) / stat["count"]

    def replay(self, batch_size=None):
        """Học từ những trải nghiệm đã có, tăng cường với pattern thành công."""
        effective_batch_size = batch_size if batch_size is not None else self.batch_size

        if len(self.memory) < effective_batch_size:
            return # Chưa đủ kinh nghiệm

        # Lấy minibatch cơ bản từ memory
        indices = np.random.choice(len(self.memory), effective_batch_size, replace=False)
        minibatch = [self.memory[i] for i in indices]

        # Bổ sung trải nghiệm từ pattern thành công (ví dụ: 10% batch size)
        num_pattern_samples = max(1, effective_batch_size // 10)
        successful_experiences = self.pattern_memory.sample_successful_experiences(num_pattern_samples)

        # Thay thế một phần minibatch bằng successful_experiences nếu có
        if successful_experiences:
             replace_indices = np.random.choice(effective_batch_size, len(successful_experiences), replace=False)
             for i, exp in enumerate(successful_experiences):
                  minibatch[replace_indices[i]] = exp

        # Chuẩn bị dữ liệu huấn luyện
        # Đảm bảo tất cả state tensors có cùng kích thước
        states = torch.stack([s for s, _, _, _, _ in minibatch if s.shape[0] == self.state_size])
        next_states = torch.stack([ns for _, _, _, ns, _ in minibatch if ns.shape[0] == self.state_size])
        actions = torch.tensor([a for s, a, _, _, _ in minibatch if s.shape[0] == self.state_size], dtype=torch.long, device=self.device)
        rewards = torch.tensor([r for s, _, r, _, _ in minibatch if s.shape[0] == self.state_size], dtype=torch.float32, device=self.device)
        dones = torch.tensor([d for s, _, _, _, d in minibatch if s.shape[0] == self.state_size], dtype=torch.bool, device=self.device)

        # Nếu không có dữ liệu hợp lệ sau khi lọc size
        if states.shape[0] == 0:
             logger.warning("No valid states found in minibatch after size check.")
             return


        # Dự đoán Q-values hiện tại và target
        # Lấy Q-values cho các action đã thực hiện
        current_q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Tính Q-values cho next_state từ target_model
        with torch.no_grad():
             # Sử dụng Double DQN: Chọn action tốt nhất từ model chính, đánh giá bằng target model
             next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
             max_next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(-1)
             # Nếu next_state là terminal (done=True), Q-value là 0
             max_next_q_values[dones] = 0.0


        # Tính toán target Q-values
        target_q_values = rewards + self.gamma * max_next_q_values

        # Tính loss (ví dụ: Smooth L1 Loss hoặc MSE Loss)
        # loss = F.smooth_l1_loss(current_q_values, target_q_values)
        loss = F.mse_loss(current_q_values, target_q_values)


        # Tối ưu hóa model
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def update_target_model(self):
        """Cập nhật target model từ model chính."""
        self.target_model.load_state_dict(self.model.state_dict())
        logger.debug("Target model updated.")


    def get_valid_actions(self, state, env=None):
        """Lấy danh sách các nước đi hợp lệ."""
        # Hàm này cần logic thực tế của game Xam Loc Solo
        # Đây là ví dụ đơn giản hóa, cần thay thế bằng logic đúng
        # Nếu có env (môi trường game), nên dùng phương thức của env
        if env and hasattr(env, 'get_valid_actions'):
             # Giả sử env.get_valid_actions trả về list các actions hợp lệ cho người chơi hiện tại
             # Player turn thường là 0 hoặc 1, cần đảm bảo state['player_turn'] đúng
             current_player_index = state.get('player_turn', 0) # Mặc định là người chơi 0 nếu không có
             return env.get_valid_actions(current_player_index)


        # Nếu không có env, tự tính toán (cần logic chi tiết hơn)
        logger.warning("Calculating valid actions without environment. Logic might be incomplete.")
        valid_actions = []
        hand = state.get("hand", [])
        current_play = state.get("current_play", []) # Bài trên bàn
        last_player = state.get("last_player") # Người chơi đánh bài cuối
        player_turn = state.get("player_turn")

        # 1. Pass (Bỏ lượt) - Chỉ hợp lệ nếu bàn có bài HOẶC không phải lượt đầu của mình
        can_pass = bool(current_play) or (last_player is not None and last_player != player_turn)
        if can_pass:
            valid_actions.append("pass")

        # 2. Declare Xam (Báo Xâm) - Chỉ hợp lệ ở đầu ván, khi chưa ai đánh
        if not current_play and last_player is None and state.get("xam_declared") is None:
             # Có thể thêm điều kiện về độ mạnh của bài ở đây
             valid_actions.append("declare_xam")

        # 3. Các action đánh bài (lẻ, đôi, sám, sảnh, tứ quý)
        # *** Cần logic chi tiết để tạo ra các bộ bài hợp lệ từ hand ***
        # Ví dụ rất đơn giản: đánh bài lẻ
        for card in hand:
             play = [card]
             # Kiểm tra xem play có hợp lệ để đánh đè current_play không
             # Hoặc nếu bàn trống (current_play rỗng) thì mọi play từ hand đều hợp lệ về mặt lý thuyết
             # Cần hàm is_valid_play(play, current_play)
             is_play_valid_theoretically = True # Tạm thời chấp nhận mọi play
             if is_play_valid_theoretically:
                 valid_actions.append(play)

        # *** Thêm logic tìm đôi, sám, sảnh, tứ quý và kiểm tra tính hợp lệ ***
        # Ví dụ tìm đôi:
        ranks = {}
        for card in hand:
            rank = card[0]
            ranks.setdefault(rank, []).append(card)

        for rank, cards_with_rank in ranks.items():
            if len(cards_with_rank) >= 2:
                # Tạo tất cả các cặp có thể từ các lá bài cùng rank
                from itertools import combinations
                for pair_tuple in combinations(cards_with_rank, 2):
                    pair = list(pair_tuple)
                    # Kiểm tra tính hợp lệ của pair
                    is_play_valid_theoretically = True # Tạm thời
                    if is_play_valid_theoretically:
                        valid_actions.append(pair)
            # Tương tự cho sám, tứ quý...

            # Tìm sảnh cần logic phức tạp hơn
            # straights = self._find_straights(hand) # Cần hàm này
            # valid_actions.extend(straights)


        # Lọc lại lần cuối để đảm bảo hợp lệ với current_play (nếu có)
        # final_valid_actions = []
        # if not current_play: # Bàn trống
        #     final_valid_actions = [a for a in valid_actions if a != "pass"] # Không pass khi bàn trống và mình đánh đầu
        # else:
        #     for action in valid_actions:
        #         if action == "pass":
        #             final_valid_actions.append(action)
        #         elif isinstance(action, list):
        #             # Cần hàm can_beat(action, current_play)
        #             if self._can_beat(action, current_play): # Giả sử có hàm này
        #                 final_valid_actions.append(action)

        # return final_valid_actions

        # *** Tạm thời trả về list đã tạo, cần hoàn thiện logic ***
        return valid_actions


    def _find_straights(self, hand, min_length=3):
        """Tìm các sảnh có trong tay bài (độ dài >= min_length) - Cần logic chính xác"""
        # Logic tìm sảnh cần được implement đúng theo luật Xâm
        straights = []
        if len(hand) < min_length:
            return straights

        # Loại bỏ lá 2, sắp xếp theo rank value
        non_two_cards = sorted([c for c in hand if c[0] != '2'], key=lambda card: RANK_VALUES.get(card[0], -1))

        if len(non_two_cards) < min_length:
            return straights

        # Sử dụng thuật toán để tìm tất cả các dãy con liên tiếp
        # Ví dụ đơn giản:
        n = len(non_two_cards)
        for length in range(min_length, n + 1):
            for i in range(n - length + 1):
                potential_straight = non_two_cards[i : i + length]
                # Kiểm tra xem có phải là sảnh không (rank liên tiếp)
                is_straight = True
                for j in range(length - 1):
                    rank1_val = RANK_VALUES.get(potential_straight[j][0], -1)
                    rank2_val = RANK_VALUES.get(potential_straight[j+1][0], -1)
                    if rank1_val == -1 or rank2_val == -1 or rank2_val != rank1_val + 1:
                        is_straight = False
                        break
                if is_straight:
                    # Kiểm tra xem có lá bài trùng rank không (không được phép trong sảnh)
                    ranks_in_straight = [c[0] for c in potential_straight]
                    if len(ranks_in_straight) == len(set(ranks_in_straight)):
                         straights.append(potential_straight)

        # Cần xử lý các trường hợp phức tạp hơn nếu luật cho phép (ví dụ sảnh từ nhiều lá bài giống nhau)

        # Loại bỏ sảnh trùng lặp nếu cần (ví dụ: có sảnh 3,4,5 và 3,4,5,6 thì chỉ giữ 3,4,5,6?)
        # Hoặc trả về tất cả các sảnh tìm được
        return straights


    def _is_straight(self, cards):
        """Kiểm tra xem một list bài có phải là sảnh hợp lệ không"""
        if not cards or len(cards) < 3:
            return False

        # Không chứa lá 2
        if any(c[0] == '2' for c in cards):
            return False

        # Rank không trùng lặp
        ranks_in_cards = [c[0] for c in cards]
        if len(ranks_in_cards) != len(set(ranks_in_cards)):
            return False

        # Sắp xếp theo giá trị rank
        sorted_cards = sorted(cards, key=lambda c: RANK_VALUES.get(c[0], -1))

        # Kiểm tra liên tục
        for i in range(len(sorted_cards) - 1):
            curr_rank_val = RANK_VALUES.get(sorted_cards[i][0], -1)
            next_rank_val = RANK_VALUES.get(sorted_cards[i+1][0], -1)
            if curr_rank_val == -1 or next_rank_val == -1 or next_rank_val != curr_rank_val + 1:
                return False

        return True


    def _would_end_with_two(self, hand, play):
        """Kiểm tra xem lượt chơi này có làm hết bài và bài cuối là lá 2 không"""
        # Chú ý: Luật Xâm thường không cấm kết thúc bằng 2 nếu đó là bộ (đôi 2, tứ quý 2)
        # Luật này áp dụng cho việc đánh lẻ lá 2 cuối cùng.
        if len(hand) == len(play): # Nếu đánh hết bài
             # Nếu play chỉ có 1 lá và đó là lá 2
             if len(play) == 1 and play[0][0] == '2':
                 return True
        return False


    def action_to_index(self, action, valid_actions):
        """Chuyển đổi hành động thành chỉ số (0-14)."""
        # Index 0-12: tương ứng rank 3-A (RANKS[0] là '3', RANKS[12] là 'A')
        # Index 13: 'pass' (action_size - 2)
        # Index 14: 'declare_xam' (action_size - 1)

        if action == "pass":
            return self.action_size - 2
        elif action == "declare_xam":
            return self.action_size - 1
        elif isinstance(action, list) and action:
             # Lấy rank của lá bài đầu tiên (hoặc lá bài cao nhất nếu cần?)
             # Hiện tại đơn giản lấy rank lá đầu tiên
             rank = action[0][0]
             if rank in RANKS:
                 # Tìm index của rank trong RANKS
                 # Chú ý: '2' không có index riêng trong action space này,
                 # hành động đánh 2 (lẻ, đôi) sẽ được map vào index của rank thấp hơn?
                 # Hoặc cần sửa đổi action space.
                 # Hiện tại, map theo rank từ 3 đến A (0-12)
                 try:
                     rank_idx = RANKS.index(rank)
                     # Giới hạn index trong khoảng 0 đến action_size - 3
                     return min(rank_idx, self.action_size - 3)
                 except ValueError:
                     logger.warning(f"Rank '{rank}' from action {action} not found in RANKS. Defaulting to index 0.")
                     return 0 # Mặc định nếu rank lạ
             else:
                 logger.warning(f"Invalid rank '{rank}' in action {action}. Defaulting to index 0.")
                 return 0 # Mặc định nếu rank không hợp lệ
        else:
            logger.warning(f"Invalid action format: {action}, defaulting to index 0")
            return 0 # Mặc định cho trường hợp không rõ

    def index_to_action(self, index, valid_actions):
        """Chuyển đổi chỉ số hành động (0-14) thành hành động thực tế từ valid_actions."""
        if not valid_actions:
             logger.warning("index_to_action called with no valid actions. Returning 'pass'.")
             return "pass" # Hoặc trả về None/lỗi


        # Xử lý các index đặc biệt trước
        if index == self.action_size - 2: # Index cho "pass"
            return "pass" if "pass" in valid_actions else valid_actions[0] # Ưu tiên pass nếu hợp lệ
        elif index == self.action_size - 1: # Index cho "declare_xam"
            return "declare_xam" if "declare_xam" in valid_actions else valid_actions[0] # Ưu tiên xam nếu hợp lệ


        # Xử lý các index 0-12 (tương ứng rank 3-A)
        if 0 <= index < len(RANKS):
             target_rank = RANKS[index]
             matching_actions = []
             # Tìm tất cả các action (list) hợp lệ bắt đầu bằng target_rank
             for action in valid_actions:
                 if isinstance(action, list) and action and action[0][0] == target_rank:
                     matching_actions.append(action)

             if matching_actions:
                 # Chiến lược chọn action tốt nhất từ matching_actions:
                 # Ưu tiên bộ lớn hơn (tứ quý > sảnh > sám > đôi > lẻ)
                 # Nếu cùng loại, ưu tiên bộ có giá trị thấp nhất? (để giữ bài cao)
                 # Hoặc đơn giản là chọn action đầu tiên tìm thấy?
                 # --> Hiện tại chọn action đầu tiên tìm thấy cho đơn giản
                 return matching_actions[0]

        # Fallback: Nếu index không map được với action nào hợp lệ
        logger.debug(f"Index {index} did not directly map to a valid action. Choosing first valid action.")
        # Loại bỏ 'pass' và 'declare_xam' khỏi lựa chọn fallback nếu có thể
        fallback_options = [a for a in valid_actions if a not in ["pass", "declare_xam"]]
        if fallback_options:
             return fallback_options[0]
        else:
             return valid_actions[0] # Nếu chỉ còn 'pass' hoặc 'declare_xam'


    def _predict_dqn(self, state_tensor, valid_actions):
        """Sử dụng DQN để dự đoán hành động tốt nhất từ các hành động hợp lệ."""
        if not valid_actions:
            return "pass" # Không có gì để chọn

        self.model.eval() # Chuyển sang chế độ đánh giá
        with torch.no_grad():
            # state_tensor đã được encode và đưa lên device
            q_values = self.model(state_tensor.unsqueeze(0))[0] # Thêm batch dim, lấy kết quả đầu tiên
        self.model.train() # Chuyển lại chế độ huấn luyện

        # Lọc Q-values chỉ cho các hành động hợp lệ
        valid_q_values = {}
        valid_indices_found = set()

        for action in valid_actions:
             try:
                 idx = self.action_to_index(action, valid_actions)
                 if 0 <= idx < self.action_size:
                     # Chỉ lấy Q-value nếu index chưa được xử lý (tránh trùng lặp do mapping)
                     if idx not in valid_indices_found:
                          valid_q_values[idx] = q_values[idx].item() # Lấy giá trị float
                          valid_indices_found.add(idx)
                 else:
                      logger.warning(f"Action {action} mapped to invalid index {idx}. Skipping.")
             except Exception as e:
                 logger.error(f"Error getting index for action {action}: {e}")


        if not valid_q_values:
             # Nếu không có q-value hợp lệ nào (lỗi mapping?), chọn ngẫu nhiên
             logger.warning("No valid Q-values found for valid actions. Choosing randomly.")
             return random.choice(valid_actions)

        # Tìm index có Q-value cao nhất trong số các index hợp lệ
        best_idx = max(valid_q_values, key=valid_q_values.get)

        # Tìm lại action tương ứng với best_idx từ valid_actions
        # Cần cẩn thận vì nhiều action có thể map về cùng 1 index
        # Ưu tiên action đã được dùng để lấy Q-value (nếu có thể truy ngược)
        # Hoặc dùng index_to_action để tìm action phù hợp nhất
        best_action = self.index_to_action(best_idx, valid_actions)

        return best_action


    # ---- ĐÂY LÀ HÀM predict_action ĐÚNG (sau khi xóa hàm trùng lặp) ----
    def predict_action(self, state, valid_actions, env=None):
        """Dự đoán hành động dựa trên trạng thái hiện tại."""
        if not valid_actions:
            logger.warning("No valid actions provided to predict_action")
            return "pass" # Hoặc nên raise lỗi?

        # Thêm game_id vào state nếu chưa có và là lượt đầu
        game_id = state.get("game_id") if isinstance(state, dict) else None
        turn_count = state.get("turn_count", 0) if isinstance(state, dict) else 0
        player_turn = state.get("player_turn", -1) if isinstance(state, dict) else -1

        if isinstance(state, dict) and not game_id and turn_count == 0:
            game_id = self.experience_manager.start_new_game()
            state["game_id"] = game_id # Gán lại vào state dict
            logger.info(f"Started new game with ID: {game_id}")
            # Khởi tạo xác suất bài nếu là lượt của AI
            if player_turn == 1: # Giả sử AI là player 1
                self.card_probability_tracker.initialize_game(game_id, state.get("hand", []))

        # Explore vs Exploit
        if np.random.rand() <= self.epsilon:
            # Exploration: Ưu tiên heuristic thông minh nếu có env
            if np.random.rand() < 0.7 and env is not None: # 70% dùng heuristic
                 logger.debug(f"Exploring with smart heuristic (epsilon={self.epsilon:.4f})")
                 # Đảm bảo _smart_heuristic_choice trả về một action trong valid_actions
                 chosen_action = self._smart_heuristic_choice(state, valid_actions, env)
                 if chosen_action in valid_actions:
                      return chosen_action
                 else:
                      logger.warning("Smart heuristic returned invalid action, falling back to random.")
                      return random.choice(valid_actions) # Fallback nếu heuristic lỗi
            else: # 30% hoàn toàn ngẫu nhiên
                logger.debug(f"Exploring randomly (epsilon={self.epsilon:.4f})")
                return random.choice(valid_actions)
        else:
            # Exploitation
            logger.debug(f"Exploiting (epsilon={self.epsilon:.4f})")
            # Thử MCTS nếu có env và xác suất cho phép (ví dụ 30%)
            use_mcts = np.random.rand() < 0.3 and env is not None

            if use_mcts:
                logger.debug("Trying MCTS for exploitation.")
                try:
                    mcts_action = self._predict_mcts(state, valid_actions, env)
                    if mcts_action is not None and mcts_action in valid_actions:
                        logger.debug(f"MCTS recommended action: {mcts_action}")
                        return mcts_action
                    elif mcts_action is not None:
                         logger.warning(f"MCTS recommended invalid action {mcts_action}. Falling back to DQN.")
                    else:
                         logger.debug("MCTS did not return an action. Falling back to DQN.")

                except ImportError:
                     logger.warning("MCTS dependency not found. Falling back to DQN.")
                except Exception as e:
                     logger.error(f"Error during MCTS prediction: {e}. Falling back to DQN.")


            # Mặc định sử dụng DQN
            logger.debug("Using DQN for exploitation.")
            try:
                encoded_state = self.encode_state(state)
                # Kiểm tra kích thước state sau khi encode
                if encoded_state.shape[0] != self.state_size:
                     logger.error(f"State encoding size mismatch in predict_action: expected {self.state_size}, got {encoded_state.shape[0]}. State: {state}")
                     # Fallback an toàn: chọn ngẫu nhiên
                     return random.choice(valid_actions)

                dqn_action = self._predict_dqn(encoded_state, valid_actions)
                # Đảm bảo action trả về là hợp lệ
                if dqn_action in valid_actions:
                     logger.debug(f"DQN recommended action: {dqn_action}")
                     return dqn_action
                else:
                     logger.warning(f"DQN returned invalid action {dqn_action}. Choosing first valid action.")
                     return valid_actions[0]
            except Exception as e:
                logger.error(f"Error during DQN prediction: {e}. Choosing randomly.")
                return random.choice(valid_actions) # Fallback nếu DQN lỗi


    # ---- HÀM NÀY BÂY GIỜ ĐƯỢC THỤT LỀ ĐÚNG ----
    def _smart_heuristic_choice(self, state, valid_actions, env):
        """Lựa chọn thông minh dựa trên heuristics (cần được tinh chỉnh)."""
        actions_to_consider = [a for a in valid_actions if a != "pass" and a != "declare_xam"]
        can_pass = "pass" in valid_actions

        if not actions_to_consider and not can_pass:
             # Trường hợp lạ: chỉ có declare_xam? Hoặc không có action nào?
             return valid_actions[0] if valid_actions else "pass" # Fallback

        hand = state.get("hand", [])
        current_play = state.get("current_play", [])
        opponent_cards_count = state.get("opponent_cards_count", 10) # Mặc định nếu thiếu

        # 1. Nếu có thể thắng ngay lập tức (đánh hết bài)
        for action in actions_to_consider:
            if isinstance(action, list) and len(action) == len(hand):
                logger.debug("Heuristic: Found winning move.")
                return action

        # 2. Nếu đối thủ còn 1 lá bài, ưu tiên đánh bộ lớn nhất hoặc lá cao nhất
        if opponent_cards_count == 1:
             # Tìm bộ tứ quý, sảnh dài, sám, đôi... hoặc lá bài cao nhất
             best_option = None
             max_val = -1
             # Sắp xếp actions_to_consider: tứ > sảnh dài > sám > đôi > lẻ cao
             # (Cần logic sắp xếp phức tạp hơn)
             # Tạm thời: tìm lá lẻ cao nhất hoặc bộ bất kỳ
             high_singles = sorted([a for a in actions_to_consider if isinstance(a, list) and len(a) == 1],
                                   key=lambda x: RANK_VALUES.get(x[0][0], 0), reverse=True)
             sets_or_straights = [a for a in actions_to_consider if isinstance(a, list) and len(a) > 1]

             if sets_or_straights:
                 best_option = sets_or_straights[0] # Ưu tiên bộ bất kỳ
                 logger.debug("Heuristic: Opponent has 1 card, playing a set/straight.")
             elif high_singles:
                 best_option = high_singles[0] # Đánh lẻ cao nhất
                 logger.debug("Heuristic: Opponent has 1 card, playing highest single.")

             if best_option:
                 return best_option


        # 3. Nếu bàn trống, ưu tiên đánh bài rác hoặc bộ nhỏ trước
        if not current_play:
            # Tìm sảnh ngắn nhất, đôi thấp nhất, hoặc lẻ thấp nhất
            options = []
            for action in actions_to_consider:
                 if isinstance(action, list):
                      val = min(RANK_VALUES.get(c[0], 0) for c in action) # Giá trị thấp nhất trong bộ
                      length = len(action)
                      options.append({'action': action, 'val': val, 'len': length})

            if options:
                 # Ưu tiên: lẻ thấp < đôi thấp < sám thấp < sảnh ngắn thấp
                 options.sort(key=lambda x: (x['len'], x['val']))
                 logger.debug("Heuristic: Empty table, playing lowest/shortest combo.")
                 return options[0]['action']

        # 4. Nếu bàn có bài, đánh bộ nhỏ nhất có thể chặn được
        if current_play:
             beatable_actions = []
             for action in actions_to_consider:
                  if isinstance(action, list): # Bỏ qua 'pass', 'declare_xam'
                      # Cần hàm kiểm tra chặt được: env._can_beat(action, current_play)
                      # Giả sử env có hàm này
                      try:
                           if env._can_beat(action, current_play): # Sử dụng hàm của env nếu có
                                val = min(RANK_VALUES.get(c[0], 0) for c in action)
                                length = len(action)
                                beatable_actions.append({'action': action, 'val': val, 'len': length})
                      except AttributeError:
                           logger.warning("Environment lacks _can_beat method for heuristic.")
                           # Fallback: Tạm coi mọi action list là có thể đánh nếu cùng loại/lớn hơn? (Logic không chính xác)
                           # Bỏ qua logic này nếu không có env._can_beat
                           pass
                      except Exception as e:
                            logger.error(f"Error calling env._can_beat: {e}")


             if beatable_actions:
                  # Ưu tiên bộ vừa đủ chặn, giá trị thấp nhất
                  beatable_actions.sort(key=lambda x: (x['len'], x['val']))
                  logger.debug("Heuristic: Beating current play with smallest possible combo.")
                  return beatable_actions[0]['action']


        # 5. Nếu không thể đánh bài (không có bộ chặn được) và có thể pass
        if can_pass:
             logger.debug("Heuristic: Cannot beat current play, passing.")
             return "pass"

        # 6. Trường hợp cuối: không thể pass, không thể chặn -> phải đánh bài (lỗi logic game?)
        # Hoặc trường hợp không rơi vào các heuristic trên -> đánh lẻ thấp nhất hoặc bộ bất kỳ
        if actions_to_consider:
             options = []
             for action in actions_to_consider:
                 if isinstance(action, list):
                      val = min(RANK_VALUES.get(c[0], 0) for c in action)
                      length = len(action)
                      options.append({'action': action, 'val': val, 'len': length})
             if options:
                  options.sort(key=lambda x: (x['len'], x['val']))
                  logger.debug("Heuristic: Fallback, playing lowest/shortest available combo.")
                  return options[0]['action']


        # Fallback cuối cùng nếu mọi thứ thất bại
        logger.warning("Heuristic could not decide, choosing first valid action.")
        return valid_actions[0]


    def _predict_mcts(self, state, valid_actions, env):
        """Sử dụng MCTS cải tiến để dự đoán hành động."""
        try:
            # Đảm bảo import cục bộ để tránh lỗi nếu module không tồn tại
            from ..ai.ai_utils import MCTS # Giả sử đường dẫn này đúng
        except ImportError:
            logger.error("MCTS module not found at src.ai.ai_utils")
            return None # Không thể chạy MCTS

        try:
            # Kiểm tra xem env có phải là đối tượng hợp lệ không
            if not hasattr(env, 'get_state') or not hasattr(env, 'step') or not hasattr(env, 'get_valid_actions'):
                 logger.error("MCTS requires a valid environment object with get_state, step, get_valid_actions methods.")
                 return None

            # Tạo bản sao của môi trường để MCTS mô phỏng mà không ảnh hưởng game chính
            # Cần một phương thức copy hoặc deepcopy đáng tin cậy cho env
            sim_env = copy.deepcopy(env) # Có thể cần tối ưu hóa nếu deepcopy chậm

            mcts = MCTS(
                env=sim_env, # Sử dụng bản sao
                ai=self,     # Truyền AI hiện tại để có thể dùng model nếu cần trong MCTS node evaluation
                simulations=100,  # Số lần mô phỏng (điều chỉnh)
                c=1.4              # Tham số UCB1 (điều chỉnh)
            )

            # Lấy player_turn từ state gốc
            current_player_index = state.get('player_turn')
            if current_player_index is None:
                 logger.error("State object missing 'player_turn' for MCTS.")
                 return None


            # Thực hiện tìm kiếm MCTS
            # Hàm search cần trả về action tốt nhất tìm được
            best_action = mcts.search(state, current_player_index, valid_actions)

            # Đảm bảo action trả về nằm trong valid_actions gốc
            if best_action in valid_actions:
                return best_action
            else:
                logger.warning(f"MCTS returned action {best_action} which is not in the original valid_actions list.")
                return None # Trả về None nếu action không hợp lệ

        except Exception as e:
            logger.exception(f"Error occurred during MCTS prediction: {e}") # Log cả traceback
            return None


    def _categorize_action(self, action):
        """Phân loại action thành các loại như single, pair, etc."""
        if not isinstance(action, list) or not action:
            return "unknown" # Hoặc có thể là "pass" / "declare_xam" nếu chúng được truyền vào

        length = len(action)
        ranks = [c[0] for c in action]
        rank_set = set(ranks)

        if length == 1:
            return "single"
        elif length == 2 and len(rank_set) == 1:
            return "pair"
        elif length == 3 and len(rank_set) == 1:
            return "triple"
        elif length == 4 and len(rank_set) == 1:
            return "four_of_a_kind"
        # Kiểm tra sảnh sau cùng vì nó phức tạp hơn
        elif self._is_straight(action): # Sử dụng hàm kiểm tra sảnh đã có
            return f"straight_{length}"
        else:
            # Các trường hợp khác (ví dụ: bộ không hợp lệ?)
            return "unknown_combo"


    def save(self, filename):
        """Lưu model và trạng thái học."""
        save_dir = os.path.dirname(filename)
        if save_dir and not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except OSError as e:
                 logger.error(f"Could not create directory {save_dir}: {e}")
                 return False


        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'target_model_state_dict': self.target_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'games_played': self.games_played,
                'experience_log': self.experience_log
                # Cân nhắc không lưu memory deque vì nó có thể rất lớn
            }, filename)
            logger.info(f"Model and state saved successfully to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {filename}: {e}")
            return False


    def load(self, filename, reset_stats=False, reset_epsilon=False):
        """Tải model và trạng thái học."""
        if not os.path.exists(filename):
            logger.error(f"Model file not found: {filename}")
            return False

        try:
            checkpoint = torch.load(filename, map_location=self.device) # Tải lên device hiện tại

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint.get('target_model_state_dict', checkpoint['model_state_dict'])) # Load target, fallback to model if missing
            # Chỉ load optimizer state nếu cấu trúc model không đổi đáng kể
            try:
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                 logger.warning(f"Could not load optimizer state_dict, likely due to model architecture change: {e}. Optimizer state reset.")
                 # Reset optimizer nếu không load được
                 self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


            if not reset_epsilon:
                self.epsilon = checkpoint.get('epsilon', self.epsilon) # Lấy epsilon, giữ nguyên nếu thiếu
            else:
                self.epsilon = 1.0 # Reset về giá trị ban đầu

            if not reset_stats:
                self.games_played = checkpoint.get('games_played', self.games_played)
                # Load log, đảm bảo tương thích nếu cấu trúc log thay đổi
                loaded_log = checkpoint.get('experience_log', {})
                # Cập nhật log hiện tại với dữ liệu đã load, giữ cấu trúc mới nếu có
                # self.experience_log.update(loaded_log) # Cách đơn giản, có thể ghi đè cấu trúc mới
                # Hoặc cập nhật từng phần một cách cẩn thận
                for key, value in loaded_log.items():
                     if key in self.experience_log:
                          if isinstance(self.experience_log[key], dict) and isinstance(value, dict):
                               self.experience_log[key].update(value)
                          else:
                               self.experience_log[key] = value
                     else:
                          self.experience_log[key] = value
                # Đảm bảo các key mặc định tồn tại nếu không có trong file save
                self._ensure_default_log_keys()

            else:
                self.games_played = 0
                self._reset_experience_log()

            self.model.to(self.device) # Đảm bảo model trên đúng device
            self.target_model.to(self.device) # Đảm bảo target model trên đúng device

            logger.info(f"Model loaded successfully from {filename}. reset_stats={reset_stats}, reset_epsilon={reset_epsilon}")
            return True
        except FileNotFoundError:
             logger.error(f"Model file not found: {filename}")
             return False
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
        self._init_action_distribution() # Khởi tạo lại action distribution
        logger.info("Experience log reset.")

    def _ensure_default_log_keys(self):
        """Đảm bảo các key cơ bản tồn tại trong experience_log sau khi load."""
        defaults = {
            "win_rate": {"games": 0, "wins": 0},
            "money_history": [],
            "turns_history": [],
            "pass_history": [],
            "xam_stats": {
                "declared_ai": 0, "declared_opp": 0,
                "success_ai": 0, "success_opp": 0
            },
            "timeout_games": 0,
            "logic_error_games": 0,
            "pattern_stats": {},
            "action_distribution": {}
        }
        for key, default_value in defaults.items():
            if key not in self.experience_log:
                self.experience_log[key] = default_value
            elif isinstance(default_value, dict):
                 # Đảm bảo các sub-key cũng tồn tại
                 if not isinstance(self.experience_log[key], dict):
                      self.experience_log[key] = default_value # Ghi đè nếu type sai
                 else:
                      for sub_key, sub_default in default_value.items():
                           if sub_key not in self.experience_log[key]:
                                self.experience_log[key][sub_key] = sub_default

        # Đảm bảo action_distribution có đủ các loại cơ bản
        default_action_keys = ["singles", "pairs", "triples", "straights", "four_kind", "passes", "xam_declared"]
        if "action_distribution" not in self.experience_log or not isinstance(self.experience_log["action_distribution"], dict):
             self._init_action_distribution()
        else:
             current_dist = self.experience_log["action_distribution"]
             for key in default_action_keys:
                  if key not in current_dist:
                       current_dist[key] = {"count": 0, "success_rate": 0, "avg_reward": 0}


    def analyze_learning(self):
        """Phân tích dữ liệu học tập và trả về list các string."""
        analysis = []
        log = self.experience_log
        self._ensure_default_log_keys() # Đảm bảo log có cấu trúc đúng

        # Truy cập an toàn vào các key
        games = log.get("win_rate", {}).get("games", 0)
        wins = log.get("win_rate", {}).get("wins", 0)
        money_history = log.get("money_history", [])
        xam_stats = log.get("xam_stats", {})
        turns_history = log.get("turns_history", [])
        action_dist = log.get("action_distribution", {})
        pattern_stats = log.get("pattern_stats", {})
        timeout_games = log.get("timeout_games", 0)
        logic_error_games = log.get("logic_error_games", 0)

        win_rate = (wins / games * 100) if games > 0 else 0
        total_money = sum(money_history)
        total_wins_money = sum(m for m in money_history if m > 0)
        total_losses_money = sum(m for m in money_history if m < 0)
        avg_money = total_money / len(money_history) if money_history else 0
        avg_win_money = total_wins_money / wins if wins > 0 else 0
        losses = games - wins
        avg_loss_money = total_losses_money / losses if losses > 0 else 0
        xam_declared_ai = xam_stats.get("declared_ai", 0)
        xam_success_ai = xam_stats.get("success_ai", 0)
        xam_success_rate = (xam_success_ai / xam_declared_ai * 100) if xam_declared_ai > 0 else 0
        avg_turns = sum(turns_history) / len(turns_history) if turns_history else 0

        # Phân tích cơ bản
        analysis.append(f"Tổng số ván đã huấn luyện: {games}")
        analysis.append(f"Tỷ lệ thắng: {win_rate:.2f}% ({wins}/{games})")
        analysis.append(f"Tổng tiền thắng/thua: {total_money:+.2f}")
        if total_wins_money > 0:
             analysis.append(f"- Tổng tiền thắng: +{total_wins_money:.2f}")
        if total_losses_money < 0:
             analysis.append(f"- Tổng tiền thua: {total_losses_money:.2f}")
        analysis.append(f"Trung bình tiền/ván: {avg_money:+.2f}")
        if wins > 0:
             analysis.append(f"- Trung bình tiền khi thắng: +{avg_win_money:.2f}")
        if losses > 0:
             analysis.append(f"- Trung bình tiền khi thua: {avg_loss_money:.2f}")
        analysis.append(f"Epsilon hiện tại: {self.epsilon:.6f}")
        analysis.append(f"Số lần AI báo xâm: {xam_declared_ai}")
        analysis.append(f"Tỷ lệ xâm thành công (AI): {xam_success_rate:.2f}% ({xam_success_ai}/{xam_declared_ai})")
        analysis.append(f"Trung bình số lượt/ván: {avg_turns:.2f}")
        analysis.append(f"Số ván timeout: {timeout_games}")
        analysis.append(f"Số ván bị lỗi logic: {logic_error_games}")

        # Phân tích phân phối hành động
        if action_dist:
            analysis.append("\nPhân phối hành động:")
            sorted_actions = sorted(action_dist.items(), key=lambda item: item[1].get('count', 0), reverse=True)
            for action_type, stats in sorted_actions:
                 count = stats.get('count', 0)
                 if count > 0:
                     success_rate = stats.get('success_rate', 0) * 100
                     avg_reward = stats.get('avg_reward', 0)
                     analysis.append(f"- {action_type.capitalize()}: {count} lần, "
                                    f"tỷ lệ thành công: {success_rate:.1f}%, "
                                    f"reward trung bình: {avg_reward:.2f}")

        # Phân tích pattern
        if pattern_stats:
            analysis.append("\nThống kê pattern:")
            sorted_patterns = sorted(pattern_stats.items(), key=lambda item: item[1].get('count', 0) if isinstance(item[1], dict) else 0, reverse=True)
            for pattern_type, stats in sorted_patterns:
                if isinstance(stats, dict) and stats.get('count', 0) > 0:
                    count = stats['count']
                    effectiveness = stats.get('effectiveness', 0)
                    analysis.append(f"- {pattern_type}: {count} lần, hiệu quả: {effectiveness:.2f}")

        # Phân tích theo thời gian (ví dụ: 100 ván gần nhất)
        window = 100
        if len(money_history) >= window:
            recent_money = money_history[-window:]
            recent_wins = sum(1 for m in recent_money if m > 0)
            recent_avg = sum(recent_money) / window
            recent_win_rate = recent_wins / window * 100

            analysis.append(f"\nKết quả {window} ván gần nhất:")
            analysis.append(f"- Tỷ lệ thắng: {recent_win_rate:.2f}% ({recent_wins}/{window})")
            analysis.append(f"- Trung bình tiền/ván: {recent_avg:+.2f}")

            # So sánh với giai đoạn trước đó nếu đủ dữ liệu
            if len(money_history) >= window * 2:
                previous_money = money_history[-window*2:-window]
                previous_wins = sum(1 for m in previous_money if m > 0)
                previous_avg = sum(previous_money) / window
                previous_win_rate = previous_wins / window * 100

                win_rate_change = recent_win_rate - previous_win_rate
                avg_change = recent_avg - previous_avg
                analysis.append(f"- Thay đổi tỷ lệ thắng so với {window} ván trước: {'+' if win_rate_change >= 0 else ''}{win_rate_change:.2f}%")
                analysis.append(f"- Thay đổi trung bình tiền so với {window} ván trước: {'+' if avg_change >= 0 else ''}{avg_change:.2f}")

        return analysis


    def plot_money_history(self, save_dir="data"):
        """Vẽ biểu đồ thay đổi tiền theo thời gian và lưu vào thư mục."""
        if not self.experience_log["money_history"]:
            logger.warning("No money history to plot.")
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg') # Sử dụng backend không cần GUI
        except ImportError:
            logger.error("Matplotlib not installed. Cannot plot money history. Install with: pip install matplotlib")
            return

        # Đảm bảo thư mục lưu tồn tại
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except OSError as e:
                 logger.error(f"Could not create directory {save_dir} for plots: {e}")
                 return


        money_history = self.experience_log["money_history"]
        games_axis = range(1, len(money_history) + 1)

        try:
            # 1. Biểu đồ tiền từng ván
            plt.figure(figsize=(12, 6))
            plt.plot(games_axis, money_history, label='Tiền mỗi ván')
            plt.title("Lịch sử tiền thưởng qua từng ván")
            plt.xlabel("Số ván")
            plt.ylabel("Tiền thưởng")
            plt.axhline(y=0, color='r', linestyle='--', linewidth=0.8, label='Hòa vốn')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plot_path1 = os.path.join(save_dir, "money_history_per_game.png")
            plt.savefig(plot_path1)
            plt.close()
            logger.debug(f"Plot saved to {plot_path1}")


            # 2. Biểu đồ tổng tiền tích lũy
            cumulative_money = np.cumsum(money_history)
            plt.figure(figsize=(12, 6))
            plt.plot(games_axis, cumulative_money, label='Tổng tiền tích lũy')
            plt.title("Tổng tiền tích lũy qua các ván")
            plt.xlabel("Số ván")
            plt.ylabel("Tổng tiền")
            plt.axhline(y=0, color='r', linestyle='--', linewidth=0.8, label='Hòa vốn ban đầu')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plot_path2 = os.path.join(save_dir, "cumulative_money.png")
            plt.savefig(plot_path2)
            plt.close()
            logger.debug(f"Plot saved to {plot_path2}")


            # 3. Biểu đồ trung bình trượt (ví dụ: cửa sổ 50 ván)
            window_size = 50
            if len(money_history) >= window_size:
                # Sử dụng pandas nếu có để tính toán dễ dàng hơn
                try:
                     import pandas as pd
                     moving_avg = pd.Series(money_history).rolling(window=window_size, min_periods=1).mean()
                except ImportError:
                     # Tính thủ công nếu không có pandas
                     moving_avg = [np.mean(money_history[max(0, i - window_size + 1):i + 1]) for i in range(len(money_history))]

                plt.figure(figsize=(12, 6))
                plt.plot(games_axis, moving_avg, label=f'Trung bình trượt {window_size} ván')
                plt.title(f"Trung bình tiền thưởng (cửa sổ trượt {window_size} ván)")
                plt.xlabel("Số ván")
                plt.ylabel(f"Tiền thưởng trung bình ({window_size} ván)")
                plt.axhline(y=0, color='r', linestyle='--', linewidth=0.8, label='Hòa vốn')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                plt.tight_layout()
                plot_path3 = os.path.join(save_dir, f"moving_avg_{window_size}_money.png")
                plt.savefig(plot_path3)
                plt.close()
                logger.debug(f"Plot saved to {plot_path3}")


            logger.info(f"Money history plots saved successfully to directory: {save_dir}")

        except Exception as e:
            logger.error(f"Error generating or saving plots: {e}", exc_info=True)