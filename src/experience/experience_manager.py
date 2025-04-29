# -*- coding: utf-8 -*-
import time
import random
import sqlite3
import json
import os
import numpy as np
from datetime import datetime

class GameExperience:
    """Lớp đại diện cho trải nghiệm của một trò chơi"""
    def __init__(self, game_id):
        self.game_id = game_id
        self.timestamp = time.time()
        self.initial_hands = {}  # Bài ban đầu của cả hai người chơi
        self.moves = []          # Danh sách các nước đi
        self.result = None       # Kết quả (thắng/thua)
        self.money_earned = 0    # Tiền thắng/thua
        self.game_features = {}  # Các đặc tính của trò chơi
        self.strategies_used = [] # Các chiến thuật được sử dụng
        self.pattern_stats = {}  # Thống kê các pattern đánh bài

    def add_move(self, player, state, action, valid_actions, reward=None):
        """Thêm một nước đi vào lịch sử"""
        move_data = {
            "player": player,
            "state": self._extract_state_features(state),
            "action": action,
            "valid_actions": valid_actions,
            "reward": reward,
            "turn": len(self.moves) + 1
        }
        self.moves.append(move_data)

    def _extract_state_features(self, state):
        """Trích xuất các đặc trưng quan trọng từ trạng thái"""
        features = {
            "hand": state.get("hand", []).copy(),
            "current_play": state.get("current_play", []).copy(),
            "opponent_cards_count": state.get("opponent_cards_count", 0),
            "consecutive_passes": state.get("consecutive_passes", 0),
            "xam_declared": state.get("xam_declared"),
            "last_player": state.get("last_player"),
            "turn_count": state.get("turn_count", 0)
        }
        return features

    def finalize(self, winner, money_earned, penalty_details=None):
        """Hoàn thiện thông tin trò chơi sau khi kết thúc"""
        self.result = winner
        self.money_earned = money_earned
        self.penalty_details = penalty_details or {}
        self.analyze_patterns()

    def analyze_patterns(self):
        """Phân tích các pattern đánh bài trong trò chơi"""
        patterns = self._extract_move_patterns()
        self.pattern_stats = {
            "offensive_patterns": patterns["offensive"],
            "defensive_patterns": patterns["defensive"],
            "successful_patterns": patterns["successful"],
            "failed_patterns": patterns["failed"]
        }

    def _extract_move_patterns(self):
        """Trích xuất các pattern từ chuỗi nước đi"""
        # Triển khai thuật toán phát hiện pattern
        patterns = {
            "offensive": [],  # Các pattern tấn công
            "defensive": [],  # Các pattern phòng thủ
            "successful": [], # Các pattern dẫn đến thắng
            "failed": []      # Các pattern dẫn đến thua
        }

        # Phát hiện các pattern tấn công (đánh bài chủ động)
        offensive_sequence = []
        for move in self.moves:
            if move["player"] == 1 and isinstance(move["action"], list) and not move["state"]["current_play"]:
                offensive_sequence.append(self._categorize_action(move["action"]))

        if offensive_sequence:
            patterns["offensive"] = self._find_frequent_subsequences(offensive_sequence)

        # Phát hiện pattern phòng thủ (đánh bài đáp trả)
        defensive_sequence = []
        for i, move in enumerate(self.moves):
            if move["player"] == 1 and isinstance(move["action"], list) and move["state"]["current_play"]:
                response_type = self._categorize_response(move["action"], move["state"]["current_play"])
                defensive_sequence.append(response_type)

        if defensive_sequence:
            patterns["defensive"] = self._find_frequent_subsequences(defensive_sequence)

        # Phân loại pattern thành công và thất bại dựa trên kết quả
        if self.result == 1:  # AI thắng
            patterns["successful"] = patterns["offensive"] + patterns["defensive"]
        else:
            patterns["failed"] = patterns["offensive"] + patterns["defensive"]

        return patterns

    def _categorize_action(self, action):
        """Phân loại một hành động thành category"""
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
        from ..game.game_rules import RANK_VALUES
        rank_value = RANK_VALUES.get(rank, 0)
        if rank_value <= 5:  # 3-7
            return "low"
        elif rank_value <= 9:  # 8-J
            return "medium"
        else:  # Q-K-A-2
            return "high"

    def _categorize_response(self, action, current_play):
        """Phân loại phản ứng đối với bài trên bàn"""
        from ..game.game_rules import RANK_VALUES

        action_type = self._categorize_action(action)
        current_type = self._categorize_action(current_play)

        # Tính chênh lệch rank
        action_rank = max(RANK_VALUES.get(c[0], 0) for c in action)
        current_rank = max(RANK_VALUES.get(c[0], 0) for c in current_play)
        rank_diff = action_rank - current_rank

        if rank_diff <= 2:
            return f"minimal_beat_{current_type}"
        elif rank_diff <= 5:
            return f"medium_beat_{current_type}"
        else:
            return f"large_beat_{current_type}"

    def _is_straight(self, cards):
        """Kiểm tra xem các lá bài có tạo thành sảnh không"""
        from ..game.game_rules import RANK_VALUES

        if len(cards) < 3:
            return False

        ranks = sorted([RANK_VALUES.get(c[0], -1) for c in cards])
        return all(ranks[i+1] == ranks[i] + 1 for i in range(len(ranks) - 1))

    def _find_frequent_subsequences(self, sequence, min_length=2, min_frequency=2):
        """Tìm các chuỗi con xuất hiện nhiều lần"""
        if len(sequence) < min_length:
            return []

        frequent_patterns = []
        for length in range(min_length, min(len(sequence), 5) + 1):
            for i in range(len(sequence) - length + 1):
                subseq = tuple(sequence[i:i+length])
                count = 0

                # Đếm số lần xuất hiện
                for j in range(len(sequence) - length + 1):
                    if tuple(sequence[j:j+length]) == subseq:
                        count += 1

                if count >= min_frequency:
                    frequent_patterns.append({
                        "pattern": list(subseq),
                        "frequency": count,
                        "confidence": count / (len(sequence) - length + 1)
                    })

        return frequent_patterns


class ExperienceManager:
    """Lớp quản lý kinh nghiệm trò chơi"""
    def __init__(self, database_path="data/game_experience.db"):
        self.database_path = database_path
        self.current_games = {}
        self.initialize_database()

    def initialize_database(self):
        """Khởi tạo cơ sở dữ liệu SQLite để lưu trữ kinh nghiệm"""
        # Tạo thư mục data nếu chưa tồn tại
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)

        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Tạo bảng games
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            timestamp REAL,
            winner INTEGER,
            money_earned REAL,
            initial_state TEXT,
            final_state TEXT,
            pattern_stats TEXT
        )
        ''')

        # Tạo bảng moves
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS moves (
            move_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            player INTEGER,
            turn INTEGER,
            state TEXT,
            action TEXT,
            valid_actions TEXT,
            reward REAL,
            FOREIGN KEY (game_id) REFERENCES games (game_id)
        )
        ''')

        # Tạo bảng patterns
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS patterns (
            pattern_id TEXT PRIMARY KEY,
            pattern_data TEXT,
            success_rate REAL,
            reward_impact REAL,
            usage_count INTEGER,
            last_updated REAL
        )
        ''')

        conn.commit()
        conn.close()

    def start_new_game(self, game_id=None):
        """Bắt đầu một game mới để theo dõi"""
        if game_id is None:
            game_id = f"game_{int(time.time())}_{random.randint(1000, 9999)}"

        self.current_games[game_id] = GameExperience(game_id)
        return game_id

    def record_move(self, game_id, player, state, action, valid_actions, reward=None):
        """Ghi lại một nước đi"""
        if game_id not in self.current_games:
            game_id = self.start_new_game(game_id)

        self.current_games[game_id].add_move(player, state, action, valid_actions, reward)

    def end_game(self, game_id, winner, money_earned, penalty_details=None):
        """Kết thúc và lưu trữ trò chơi"""
        if game_id not in self.current_games:
            raise ValueError(f"Không tìm thấy game_id {game_id}")

        game = self.current_games[game_id]
        game.finalize(winner, money_earned, penalty_details)

        # Lưu vào cơ sở dữ liệu
        self._save_game_to_database(game)

        # Ghi nhớ vào bộ nhớ tập thể
        self._update_collective_memory(game)

        # Xóa khỏi games hiện tại để giải phóng bộ nhớ
        del self.current_games[game_id]

    def _save_game_to_database(self, game):
        """Lưu thông tin trò chơi vào cơ sở dữ liệu"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Lưu thông tin game
        cursor.execute('''
        INSERT OR REPLACE INTO games
        (game_id, timestamp, winner, money_earned, initial_state, final_state, pattern_stats)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            game.game_id,
            game.timestamp,
            game.result,
            game.money_earned,
            json.dumps(game.initial_hands),
            json.dumps(game.moves[-1]["state"] if game.moves else {}),
            json.dumps(game.pattern_stats)
        ))

        # Lưu các nước đi
        for move in game.moves:
            cursor.execute('''
            INSERT INTO moves
            (game_id, player, turn, state, action, valid_actions, reward)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                game.game_id,
                move["player"],
                move["turn"],
                json.dumps(move["state"]),
                json.dumps(move["action"]),
                json.dumps(move["valid_actions"]),
                move["reward"]
            ))

        conn.commit()
        conn.close()

    def _update_collective_memory(self, game):
        """Cập nhật bộ nhớ tập thể với các pattern từ trò chơi này"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Lấy tất cả patterns từ game này
        all_patterns = []
        if "offensive_patterns" in game.pattern_stats:
            all_patterns.extend(game.pattern_stats["offensive_patterns"])
        if "defensive_patterns" in game.pattern_stats:
            all_patterns.extend(game.pattern_stats["defensive_patterns"])

        # Lưu hoặc cập nhật patterns
        for pattern_data in all_patterns:
            pattern_id = self._generate_pattern_id(pattern_data["pattern"])

            # Kiểm tra xem pattern đã tồn tại chưa
            cursor.execute('SELECT * FROM patterns WHERE pattern_id = ?', (pattern_id,))
            existing = cursor.fetchone()

            success_rate = 1.0 if game.result == 1 else 0.0
            reward_impact = abs(game.money_earned) / 10  # Chuẩn hóa

            if existing:
                # Cập nhật pattern đã tồn tại
                pattern_data = json.loads(existing[1])
                old_success_rate = existing[2]
                old_usage = existing[4]

                # Cập nhật tỷ lệ thành công và tác động
                new_success_rate = (old_success_rate * old_usage + success_rate) / (old_usage + 1)
                new_impact = (pattern_data.get("reward_impact", 0) * old_usage + reward_impact) / (old_usage + 1)

                cursor.execute('''
                UPDATE patterns
                SET success_rate = ?, reward_impact = ?, usage_count = ?, last_updated = ?
                WHERE pattern_id = ?
                ''', (
                    new_success_rate,
                    new_impact,
                    old_usage + 1,
                    time.time(),
                    pattern_id
                ))
            else:
                # Thêm pattern mới
                cursor.execute('''
                INSERT INTO patterns
                (pattern_id, pattern_data, success_rate, reward_impact, usage_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    pattern_id,
                    json.dumps(pattern_data),
                    success_rate,
                    reward_impact,
                    1,
                    time.time()
                ))

        conn.commit()
        conn.close()

    def _generate_pattern_id(self, pattern):
        """Tạo ID duy nhất cho pattern"""
        import hashlib

        # Sắp xếp để đảm bảo tính nhất quán
        serialized = json.dumps(pattern, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def get_game_history(self, game_id):
        """Lấy lịch sử trò chơi từ cơ sở dữ liệu"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM games WHERE game_id = ?', (game_id,))
        game_data = cursor.fetchone()

        if not game_data:
            conn.close()
            return None

        cursor.execute('SELECT * FROM moves WHERE game_id = ? ORDER BY turn', (game_id,))
        moves_data = cursor.fetchall()

        conn.close()

        # Chuyển đổi dữ liệu thành đối tượng GameExperience
        game = GameExperience(game_id)
        game.timestamp = game_data[1]
        game.result = game_data[2]
        game.money_earned = game_data[3]
        game.initial_hands = json.loads(game_data[4])
        game.pattern_stats = json.loads(game_data[6])

        # Thêm các nước đi
        for move in moves_data:
            move_obj = {
                "player": move[2],
                "turn": move[3],
                "state": json.loads(move[4]),
                "action": json.loads(move[5]),
                "valid_actions": json.loads(move[6]),
                "reward": move[7]
            }
            game.moves.append(move_obj)

        return game

    def get_successful_patterns(self, min_success_rate=0.6, min_usage=5, limit=20):
        """Lấy các pattern thành công từ cơ sở dữ liệu"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT pattern_id, pattern_data, success_rate, reward_impact, usage_count
        FROM patterns
        WHERE success_rate >= ? AND usage_count >= ?
        ORDER BY success_rate * reward_impact DESC
        LIMIT ?
        ''', (min_success_rate, min_usage, limit))

        patterns = cursor.fetchall()
        conn.close()

        return [{
            "pattern_id": p[0],
            "pattern": json.loads(p[1]),
            "success_rate": p[2],
            "reward_impact": p[3],
            "usage_count": p[4]
        } for p in patterns]

    def get_statistics(self):
        """Lấy thống kê tổng quan từ toàn bộ trải nghiệm"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        stats = {}

        # Tổng số trò chơi
        cursor.execute('SELECT COUNT(*) FROM games')
        stats["total_games"] = cursor.fetchone()[0]

        # Tỷ lệ thắng thua
        cursor.execute('SELECT winner, COUNT(*) FROM games GROUP BY winner')
        win_stats = cursor.fetchall()
        stats["win_stats"] = {row[0]: row[1] for row in win_stats}

        # Tiền thắng thua trung bình
        cursor.execute('SELECT AVG(money_earned) FROM games')
        stats["avg_money"] = cursor.fetchone()[0]

        # Số nước đi trung bình
        cursor.execute('SELECT AVG(turn) FROM moves GROUP BY game_id')
        turn_stats = cursor.fetchall()
        stats["avg_turns"] = sum(row[0] for row in turn_stats) / len(turn_stats) if turn_stats else 0

        # Thống kê pattern thành công
        cursor.execute('''
        SELECT COUNT(*), AVG(success_rate), AVG(reward_impact)
        FROM patterns
        WHERE success_rate > 0.5 AND usage_count > 3
        ''')
        pattern_stats = cursor.fetchone()

        stats["successful_patterns"] = {
            "count": pattern_stats[0] if pattern_stats else 0,
            "avg_success_rate": pattern_stats[1] if pattern_stats else 0,
            "avg_reward_impact": pattern_stats[2] if pattern_stats else 0
        }

        conn.close()
        return stats