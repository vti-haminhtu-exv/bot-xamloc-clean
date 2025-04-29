# -*- coding: utf-8 -*-
import numpy as np
import json
from ..game.game_rules import SUITS, RANKS, RANK_VALUES

class CardProbabilityTracker:
    """Theo dõi xác suất bài của đối thủ"""
    def __init__(self):
        """Khởi tạo bộ theo dõi xác suất bài"""
        self.game_states = {}  # Lưu trữ trạng thái của mỗi trò chơi

    def initialize_game(self, game_id, player_hand, deck_size=52):
        """Khởi tạo một game mới để theo dõi"""
        initial_probabilities = np.ones(deck_size) / deck_size

        # Cập nhật xác suất dựa trên bài đã biết
        known_cards_indices = self._get_card_indices(player_hand)
        for idx in known_cards_indices:
            initial_probabilities[idx] = 0

        # Chuẩn hóa xác suất
        initial_probabilities = self._normalize_probabilities(initial_probabilities)

        self.game_states[game_id] = {
            "probabilities": initial_probabilities,
            "played_cards": [],
            "player_hand": player_hand.copy()
        }

    def update_probabilities(self, game_id, player_id, action, current_play=None):
        """Cập nhật xác suất khi có thêm thông tin mới"""
        if game_id not in self.game_states:
            return

        state = self.game_states[game_id]

        # Cập nhật bài đã đánh
        if isinstance(action, list) and action:
            state["played_cards"].extend(action)

            # Cập nhật xác suất
            played_indices = self._get_card_indices(action)
            for idx in played_indices:
                state["probabilities"][idx] = 0

            # Chuẩn hóa lại
            state["probabilities"] = self._normalize_probabilities(state["probabilities"])

        # Cập nhật dựa trên thông tin bỏ lượt
        elif action == "pass" and current_play:
            # Nếu không phải là lá 2 hoặc tứ quý, giảm xác suất
            # đối thủ có bài lớn hơn current_play
            self._update_pass_probabilities(game_id, player_id, current_play)

    def _update_pass_probabilities(self, game_id, player_id, current_play):
        """Cập nhật xác suất khi một người chơi bỏ lượt"""
        state = self.game_states[game_id]

        # Xác định loại bài trên bàn
        play_type = self._get_play_type(current_play)

        # Bỏ lượt nghĩa là không có bài cao hơn, giảm xác suất các lá bài cao hơn
        if play_type == "single":
            current_rank = RANK_VALUES.get(current_play[0][0], 0)
            for rank, value in RANK_VALUES.items():
                if value > current_rank:
                    # Giảm xác suất đối thủ có các lá bài cao hơn
                    for suit in SUITS:
                        idx = self._get_card_index((rank, suit))
                        if 0 <= idx < len(state["probabilities"]):
                            state["probabilities"][idx] *= 0.5  # Giảm 50%

        elif play_type == "pair":
            current_rank = RANK_VALUES.get(current_play[0][0], 0)
            # Giảm xác suất đối thủ có đôi cao hơn
            for rank, value in RANK_VALUES.items():
                if value > current_rank:
                    # Cần giảm xác suất đối thủ có cặp của rank này
                    suit_combinations = [(s1, s2) for s1 in SUITS for s2 in SUITS if s1 != s2]
                    for s1, s2 in suit_combinations:
                        idx1 = self._get_card_index((rank, s1))
                        idx2 = self._get_card_index((rank, s2))

                        # Chỉ giảm nếu đối thủ có khả năng có cả 2 lá
                        if (0 <= idx1 < len(state["probabilities"]) and
                            0 <= idx2 < len(state["probabilities"]) and
                            state["probabilities"][idx1] > 0 and
                            state["probabilities"][idx2] > 0):
                            # Giảm xác suất cả 2 lá
                            state["probabilities"][idx1] *= 0.6
                            state["probabilities"][idx2] *= 0.6

        # Tương tự cho các loại bài khác, nhưng đơn giản hóa ở đây

        # Chuẩn hóa lại
        state["probabilities"] = self._normalize_probabilities(state["probabilities"])

    def _get_play_type(self, play):
        """Xác định loại bài được đánh"""
        if len(play) == 1:
            return "single"
        elif len(play) == 2 and play[0][0] == play[1][0]:
            return "pair"
        elif len(play) == 3 and play[0][0] == play[1][0] == play[2][0]:
            return "triple"
        elif len(play) == 4 and play[0][0] == play[1][0] == play[2][0] == play[3][0]:
            return "four_of_a_kind"
        elif self._is_straight(play):
            return "straight"
        return "unknown"

    def _is_straight(self, cards):
        """Kiểm tra xem các lá bài có tạo thành sảnh không"""
        if len(cards) < 3:
            return False

        ranks = sorted([RANK_VALUES.get(c[0], -1) for c in cards])
        return all(ranks[i+1] == ranks[i] + 1 for i in range(len(ranks) - 1))

    def get_probabilities(self, game_id, player_id):
        """Lấy xác suất bài đối thủ hiện tại"""
        if game_id not in self.game_states:
            return np.zeros(52)

        return self.game_states[game_id]["probabilities"]

    def get_probability_matrix(self, game_id):
        """Lấy ma trận xác suất (rank x suit)"""
        if game_id not in self.game_states:
            return np.zeros((len(RANKS), len(SUITS)))

        probs = self.game_states[game_id]["probabilities"]
        matrix = np.zeros((len(RANKS), len(SUITS)))

        for rank_idx, rank in enumerate(RANKS):
            for suit_idx, suit in enumerate(SUITS):
                idx = rank_idx * len(SUITS) + suit_idx
                if idx < len(probs):
                    matrix[rank_idx, suit_idx] = probs[idx]

        return matrix

    def get_rank_probabilities(self, game_id):
        """Lấy xác suất cho từng rank"""
        matrix = self.get_probability_matrix(game_id)
        return np.sum(matrix, axis=1)  # Tổng theo hàng (rank)

    def _get_card_indices(self, cards):
        """Chuyển đổi các lá bài thành các chỉ số trong mảng xác suất"""
        indices = []
        for card in cards:
            idx = self._get_card_index(card)
            if idx >= 0:
                indices.append(idx)
        return indices

    def _get_card_index(self, card):
        """Lấy chỉ số của một lá bài"""
        if not card or len(card) != 2:
            return -1

        rank, suit = card
        if rank not in RANKS or suit not in SUITS:
            return -1

        rank_idx = RANKS.index(rank)
        suit_idx = SUITS.index(suit)
        idx = rank_idx * len(SUITS) + suit_idx

        return idx

    def _normalize_probabilities(self, probabilities):
        """Chuẩn hóa mảng xác suất để tổng bằng 1"""
        sum_prob = np.sum(probabilities)
        if sum_prob > 0:
            return probabilities / sum_prob
        return probabilities

    def simulate_opponent_hand(self, game_id, num_cards):
        """Mô phỏng bài của đối thủ dựa trên xác suất"""
        if game_id not in self.game_states or num_cards <= 0:
            return []

        probs = self.game_states[game_id]["probabilities"]

        # Lấy tất cả vị trí có xác suất > 0
        valid_indices = np.where(probs > 0)[0]

        if len(valid_indices) == 0:
            return []

        # Lấy xác suất tương ứng và chuẩn hóa
        valid_probs = probs[valid_indices]
        valid_probs = valid_probs / np.sum(valid_probs)

        # Chọn ngẫu nhiên theo xác suất
        num_to_select = min(num_cards, len(valid_indices))
        selected_indices = np.random.choice(valid_indices, size=num_to_select, replace=False, p=valid_probs)

        # Chuyển lại thành các lá bài
        simulated_hand = []
        for idx in selected_indices:
            rank_idx = idx // len(SUITS)
            suit_idx = idx % len(SUITS)

            if rank_idx < len(RANKS) and suit_idx < len(SUITS):
                card = (RANKS[rank_idx], SUITS[suit_idx])
                simulated_hand.append(card)

        return simulated_hand