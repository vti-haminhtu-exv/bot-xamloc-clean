# -*- coding: utf-8 -*-
import numpy as np
import json
import random
import time
import sqlite3
import hashlib
from ..game.game_rules import RANK_VALUES

class PatternMemory:
    """Lớp quản lý bộ nhớ pattern"""
    def __init__(self, database_path="data/game_experience.db", pattern_threshold=0.6):
        """Khởi tạo bộ nhớ pattern"""
        self.database_path = database_path
        self.patterns = {}  # Lưu trữ các pattern và trọng số
        self.pattern_threshold = pattern_threshold  # Ngưỡng nhận diện pattern
        self.load_patterns_from_db()

    def load_patterns_from_db(self):
        """Tải patterns từ cơ sở dữ liệu"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute('SELECT pattern_id, pattern_data, success_rate, reward_impact, usage_count FROM patterns')
            patterns = cursor.fetchall()

            conn.close()

            for p in patterns:
                self.patterns[p[0]] = {
                    "pattern": json.loads(p[1]),
                    "success_rate": p[2],
                    "reward_impact": p[3],
                    "usage_count": p[4]
                }

            return len(self.patterns)
        except (sqlite3.Error, json.JSONDecodeError) as e:
            print(f"Error loading patterns: {e}")
            return 0

    def add_pattern(self, pattern, success_rate, reward_impact):
        """Thêm một pattern mới vào bộ nhớ"""
        pattern_id = self._generate_pattern_id(pattern)
        self.patterns[pattern_id] = {
            "pattern": pattern,
            "success_rate": success_rate,
            "reward_impact": reward_impact,
            "usage_count": 0
        }

    def evaluate_pattern_match(self, state, action):
        """Kiểm tra xem action hiện tại có khớp với pattern nào không"""
        matching_patterns = self._find_matching_patterns(state, action)
        total_bonus = 0

        for pattern_id, match_score in matching_patterns:
            if match_score >= self.pattern_threshold:
                pattern = self.patterns[pattern_id]
                # Tính thưởng dựa trên độ khớp và tác động của pattern
                bonus = match_score * pattern["success_rate"] * pattern["reward_impact"]
                total_bonus += bonus

                # Cập nhật số lần sử dụng pattern
                pattern["usage_count"] += 1

        return total_bonus

    def _generate_pattern_id(self, pattern):
        """Tạo ID duy nhất cho pattern"""
        # Sắp xếp để đảm bảo tính nhất quán
        serialized = json.dumps(pattern, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def _find_matching_patterns(self, state, action):
        """Tìm các pattern khớp với trạng thái và hành động hiện tại"""
        if not action or not isinstance(action, list) or not state:
            return []

        matching_patterns = []
        current_action_type = self._categorize_action(action)

        for pattern_id, pattern_info in self.patterns.items():
            pattern = pattern_info["pattern"]

            # Lược bỏ các pattern không phù hợp
            if "pattern" in pattern and isinstance(pattern["pattern"], list):
                # Kiểm tra pattern từ dữ liệu lịch sử ván đấu
                match_score = self._match_history_pattern(pattern, state, action, current_action_type)
            else:
                # Pattern đơn lẻ
                match_score = self._match_single_pattern(pattern, state, action, current_action_type)

            if match_score > 0:
                matching_patterns.append((pattern_id, match_score))

        # Sắp xếp theo điểm khớp giảm dần
        matching_patterns.sort(key=lambda x: x[1], reverse=True)
        return matching_patterns

    def _match_history_pattern(self, pattern, state, action, action_type):
        """Kiểm tra độ khớp với pattern lịch sử"""
        if "pattern" not in pattern or not pattern["pattern"]:
            return 0.0

        # Kiểm tra pattern là chuỗi các hành động
        pattern_sequence = pattern["pattern"]

        # Pattern là một chuỗi các hành động, chúng ta cần kiểm tra
        # xem action hiện tại có khớp với một thành phần cuối cùng không
        if len(pattern_sequence) > 0:
            last_action_pattern = pattern_sequence[-1]

            # Nếu action type giống với pattern cuối
            if last_action_pattern == action_type:
                return 0.8  # 80% match

            # Nếu pattern bao gồm các action liên quan
            if (action_type.startswith("single_") and last_action_pattern.startswith("single_")) or \
               (action_type.startswith("pair_") and last_action_pattern.startswith("pair_")) or \
               (action_type.startswith("straight_") and last_action_pattern.startswith("straight_")):
                return 0.5  # 50% match

        return 0.0

    def _match_single_pattern(self, pattern, state, action, action_type):
        """Kiểm tra độ khớp với pattern đơn lẻ"""
        # Kiểm tra các yếu tố trạng thái
        match_score = 0.0

        # Kiểm tra trạng thái bàn
        has_current_play = bool(state.get("current_play", []))

        # Đánh giá dựa trên loại hành động
        if not has_current_play and action_type.startswith("pair_"):
            # Ưu tiên đánh đôi khi bàn trống
            match_score += 0.4

        elif not has_current_play and action_type.startswith("straight_"):
            # Ưu tiên đánh sảnh khi bàn trống
            straight_length = int(action_type.split("_")[1]) if "_" in action_type else 0
            if straight_length >= 4:
                match_score += 0.6  # Sảnh dài tốt hơn
            else:
                match_score += 0.3

        elif has_current_play and action_type.startswith("single_high"):
            # Đánh lá cao để chặn khi bàn có bài
            match_score += 0.3

        # Kiểm tra số lá đối thủ
        opponent_cards = state.get("opponent_cards_count", 0)

        if opponent_cards <= 3:
            # Bài đối thủ sắp hết, ưu tiên đánh cao
            if "high" in action_type:
                match_score += 0.4
        elif opponent_cards >= 7:
            # Bài đối thủ còn nhiều, ưu tiên đánh thấp hoặc nhiều lá
            if "low" in action_type or action_type.startswith("straight_"):
                match_score += 0.3

        return match_score

    def _categorize_action(self, action):
        """Phân loại một hành động thành category"""
        if not action:
            return "unknown"

        if len(action) == 1:
            return "single_" + self._rank_category(action[0][0])
        elif len(action) == 2 and action[0][0] == action[1][0]:
            return "pair_" + self._rank_category(action[0][0])
        elif len(action) == 3 and action[0][0] == action[1][0] == action[2][0]:
            return "triple_" + self._rank_category(action[0][0])
        elif len(action) == 4 and action[0][0] == action[1][0] == action[2][0] == action[3][0]:
            return "four_" + self._rank_category(action[0][0])
        elif self._is_straight(action):
            return f"straight_{len(action)}_{self._rank_category(action[0][0])}"
        return "unknown"

    def _rank_category(self, rank):
        """Phân loại rank thành low/medium/high"""
        rank_value = RANK_VALUES.get(rank, 0)
        if rank_value <= 5:  # 3-7
            return "low"
        elif rank_value <= 9:  # 8-J
            return "medium"
        else:  # Q-K-A-2
            return "high"

    def _is_straight(self, cards):
        """Kiểm tra xem các lá bài có tạo thành sảnh không"""
        if len(cards) < 3:
            return False

        ranks = sorted([RANK_VALUES.get(c[0], -1) for c in cards])
        return all(ranks[i+1] == ranks[i] + 1 for i in range(len(ranks) - 1))

    def evaluate_action(self, state, action):
        """Đánh giá giá trị của một hành động dựa trên pattern"""
        matching_patterns = self._find_matching_patterns(state, action)

        if not matching_patterns:
            return 0.0

        # Tính giá trị trung bình của các pattern khớp
        total_value = 0.0
        total_weight = 0.0

        for pattern_id, match_score in matching_patterns:
            pattern = self.patterns[pattern_id]
            weight = match_score * pattern["success_rate"]
            value = pattern["reward_impact"]

            total_value += weight * value
            total_weight += weight

        return total_value / max(total_weight, 1e-10)

    def encode_active_patterns(self, state):
        """Mã hóa các pattern đang hoạt động thành vector đặc trưng"""
        # Vector 10 chiều
        features = np.zeros(10)

        # Không thể mã hóa nếu không có trạng thái
        if not state:
            return features

        # Đặc trưng 1: Có bài trên bàn không
        features[0] = 1.0 if state.get("current_play") else 0.0

        # Đặc trưng 2: Đối thủ sắp hết bài không
        opponent_cards = state.get("opponent_cards_count", 0)
        features[1] = 1.0 if opponent_cards <= 3 else 0.0

        # Đặc trưng 3: Đối thủ còn nhiều bài không
        features[2] = 1.0 if opponent_cards >= 7 else 0.0

        # Đặc trưng 4: Lượt bỏ liên tiếp
        features[3] = min(state.get("consecutive_passes", 0) / 3.0, 1.0)

        # Đặc trưng 5: Đã báo Xâm chưa
        xam_declared = state.get("xam_declared")
        features[4] = 0.0 if xam_declared is None else (1.0 if xam_declared == 1 else -1.0)

        # Đặc trưng 6-10: Phân tích tay bài của người chơi
        hand = state.get("hand", [])
        if hand:
            # Đếm các loại lá bài trong tay
            ranks = [card[0] for card in hand]

            # Đặc trưng 6: Số lá 2
            features[5] = min(ranks.count('2') / 2.0, 1.0)

            # Đặc trưng 7: Tỷ lệ lá cao (Q-K-A)
            high_cards = sum(1 for r in ranks if r in ['Q', 'K', 'A'])
            features[6] = min(high_cards / 4.0, 1.0)

            # Đặc trưng 8: Có đôi không
            rank_counts = {r: ranks.count(r) for r in set(ranks)}
            has_pairs = any(count >= 2 for count in rank_counts.values())
            features[7] = 1.0 if has_pairs else 0.0

            # Đặc trưng 9: Có sám không
            has_triples = any(count >= 3 for count in rank_counts.values())
            features[8] = 1.0 if has_triples else 0.0

            # Đặc trưng 10: Có tứ quý không
            has_four = any(count == 4 for count in rank_counts.values())
            features[9] = 1.0 if has_four else 0.0

        return features

    def sample_successful_experiences(self, n_samples=5):
        """Lấy mẫu trải nghiệm thành công từ các pattern"""
        if not self.patterns:
            return []

        # Lọc patterns thành công
        successful_patterns = [(pid, p) for pid, p in self.patterns.items()
                               if p["success_rate"] >= 0.6 and p["usage_count"] >= 5]

        if not successful_patterns:
            return []

        # Sắp xếp theo success_rate * reward_impact giảm dần
        successful_patterns.sort(key=lambda x: x[1]["success_rate"] * x[1]["reward_impact"], reverse=True)

        # Lấy mẫu
        n_to_sample = min(n_samples, len(successful_patterns))
        sampled_patterns = successful_patterns[:n_to_sample]

        # Chuyển patterns thành trải nghiệm cho replay
        experiences = []
        for _, pattern in sampled_patterns:
            # Tạo trạng thái ảo và hành động dựa trên pattern
            # Trạng thái và hành động thực tế sẽ được xử lý bởi
            # mô-đun khác, ở đây chỉ đơn giản hóa
            experience = (
                np.random.rand(47),  # state vector - mocked
                random.randint(0, 14),  # action index - mocked
                pattern["reward_impact"] * 2,  # reward được tăng cường
                np.random.rand(47),  # next state - mocked
                random.choice([True, False])  # done - mocked
            )
            experiences.append(experience)

        return experiences