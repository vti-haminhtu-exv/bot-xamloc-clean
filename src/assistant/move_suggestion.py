# -*- coding: utf-8 -*-
import logging
from itertools import combinations
from ..game.game_environment import XamLocSoloEnvironment
from ..game.game_rules import SUITS, RANKS, RANK_VALUES
from ..ai.ai_interface import get_valid_actions, predict_action

# Sử dụng logger chung
logger = logging.getLogger('xam_loc_solo')

def suggest_move(self, player_hand, current_play, opponent_cards, xam_declared=None, last_player=None, consecutive_passes=0):
    """Gợi ý nước đi cho người chơi dựa trên trạng thái trò chơi."""
    env = XamLocSoloEnvironment(self.betting_unit)
    env.player_hand = player_hand
    env.ai_hand = []  # AI hand rỗng vì chỉ quan tâm đến người chơi
    env.current_play = current_play
    env.current_player = 0  # Đặt người chơi là người hành động
    env.consecutive_passes = consecutive_passes
    env.xam_declared = xam_declared
    env.last_player = last_player

    state = {
        "hand": player_hand,
        "current_play": current_play,
        "opponent_cards_count": opponent_cards,
        "player_turn": 0,
        "consecutive_passes": consecutive_passes,
        "xam_declared": xam_declared,
        "last_player": last_player,
        "money_earned": 0,
        "winner": None,
        "turn_count": 0,
        "pass_count": 0
    }

    # Lấy và lọc các hành động hợp lệ
    valid_actions = get_valid_actions(self.ai, state, env)
    filtered_valid_actions = []
    for action in valid_actions:
        if action == "pass" or action == "declare_xam":
            filtered_valid_actions.append(action)
        elif isinstance(action, list) and all(card in player_hand for card in action):
            filtered_valid_actions.append(action)

    if not filtered_valid_actions:
        filtered_valid_actions = ["pass"]
        logger.warning("No valid actions found, defaulting to 'pass'")

    # Dự đoán hành động
    action = predict_action(self.ai, state, filtered_valid_actions, env)

    # Tạo giải thích
    explanation = _generate_explanation(self, action, filtered_valid_actions)

    return {"action": action, "explanation": explanation}

def _format_action(self, action):
    """Định dạng hành động thành chuỗi để hiển thị."""
    if isinstance(action, list):
        return ", ".join([f"{c[0]}{c[1]}" for c in action])
    return action

def _generate_explanation(self, action, valid_actions):
    """Tạo giải thích cho hành động được chọn."""
    if action == "pass":
        if len(valid_actions) == 1 and valid_actions[0] == "pass":
            return "Bỏ lượt vì không có bài chặn."
        return "Bỏ lượt vì chiến thuật dựa trên kinh nghiệm."
    elif action == "declare_xam":
        return "Tuyên bố Xâm vì bài mạnh và có khả năng thắng nhanh."
    elif isinstance(action, list):
        if len(action) == 1:
            return f"Đánh {_format_action(self, action)} (Đơn)."
        elif len(action) == 2:
            return f"Đánh {_format_action(self, action)} (Đôi)."
        elif len(action) == 3:
            return f"Đánh {_format_action(self, action)} (Sám)."
        elif len(action) == 4:
            return f"Đánh {_format_action(self, action)} (Tứ quý)."
        else:
            return f"Đánh {_format_action(self, action)} (Sảnh)."
    return "Hành động không xác định."

def get_player_valid_actions(self, env):
    """Tính các hành động hợp lệ cho người chơi (P0) dựa trên tay bài hiện tại."""
    hand = env.player_hand
    cp = env.current_play
    valid = []
    can_pass = True

    # Kiểm tra xem có thể bỏ lượt không
    if not cp and hand:
        can_pass = False
    elif env.last_player == 0 and env.consecutive_passes >= 1:
        can_pass = False

    if can_pass:
        valid.append(("pass", 0))

    if hand:
        pp = []
        # Thêm các lá bài đơn
        for card in hand:
            pp.append(([card], 1))

        # Tạo các tổ hợp đôi, sám, tứ quý từ các lá bài thực sự có trong tay
        rank_counts = {r: [c for c in hand if c[0] == r] for r, _ in hand}
        for r, cards in rank_counts.items():
            if len(cards) >= 2:
                for cb in combinations(cards, 2):
                    pp.append((list(cb), 10))
            if len(cards) >= 3:
                for cb in combinations(cards, 3):
                    pp.append((list(cb), 15))
            if len(cards) == 4:
                pp.append((list(cards), 20))

        # Tạo sảnh (bao gồm cả lá 2 nếu nằm trong chuỗi liên tục)
        if len(hand) >= 3:
            try:
                sorted_by_rank = sorted(hand, key=lambda c: RANK_VALUES.get(c[0], -1))
                for length in range(3, len(hand) + 1):
                    for i in range(len(hand) - length + 1):
                        candidate = sorted_by_rank[i:i+length]
                        ranks_nums = [RANK_VALUES.get(c[0], -1) for c in candidate]
                        ranks_nums.sort()
                        is_sequential = True
                        for j in range(len(ranks_nums) - 1):
                            if ranks_nums[j + 1] != ranks_nums[j] + 1:
                                is_sequential = False
                                break
                        has_K = 11 in ranks_nums  # K = 11
                        has_A = 12 in ranks_nums  # A = 12
                        has_2 = 13 in ranks_nums  # 2 = 13
                        if has_K and has_A and has_2:
                            is_sequential = False
                        if is_sequential:
                            pp.append((candidate, 20))
            except Exception as e:
                logger.error(f"Error creating straights: {e}")

        valid_actions = []
        for play, base_priority in pp:
            ps = sorted(play, key=lambda c: RANK_VALUES.get(c[0], -1))
            if all(card in hand for card in ps):
                try:
                    if env._is_valid_play(ps, hand):
                        would_end_with_two = False
                        if set(ps) == set(hand):
                            would_end_with_two = any(c[0] == '2' for c in ps)
                        if not would_end_with_two:
                            priority = base_priority
                            valid_actions.append((ps, priority))
                        else:
                            logger.debug(f"Skipping play {ps} as it would end with two")
                except Exception as e:
                    logger.error(f"Validate play error: {e}, play: {ps}")
                    continue

        valid_actions.sort(key=lambda x: x[1], reverse=True)
        valid.extend(valid_actions)

    # Loại bỏ các hành động trùng lặp
    deduped = []
    seen = set()
    for a in valid:
        ar = tuple(sorted(a[0], key=lambda c: RANK_VALUES.get(c[0], -1)) if isinstance(a[0], list) else a[0])
        if ar not in seen:
            deduped.append(a[0])
            seen.add(ar)

    logger.info(f"Player valid actions: {deduped}")
    return deduped if deduped else ["pass"]

def parse_cards(cards_str):
    """Chuyển đổi chuỗi bài thành danh sách các lá bài."""
    if not cards_str:
        return []
    cards = []
    for card in cards_str.split(','):
        card = card.strip()
        if len(card) < 2:
            continue
        rank = card[:-1]
        suit = card[-1]
        if rank in RANKS and suit in SUITS:
            cards.append((rank, suit))
    return cards