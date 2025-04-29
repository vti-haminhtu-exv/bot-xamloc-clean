# -*- coding: utf-8 -*-
import logging

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hằng số trò chơi
SUITS = ['♣', '♦', '♥', '♠']
RANKS = [str(i) for i in range(3, 11)] + ['J', 'Q', 'K', 'A', '2']
RANK_VALUES = {str(i): i-2 for i in range(3, 11)}
RANK_VALUES.update({'J': 9, 'Q': 10, 'K': 11, 'A': 12, '2': 13})
CARDS_PER_PLAYER = 10
PENALTY_PER_CARD = 1
PENALTY_2_OR_FOUR_OF_KIND = 11
REWARD_XAM_SUCCESS = 20
PENALTY_XAM_FAILURE = 10
MAX_TURNS_PENALTY_THRESHOLD = 100

def get_combination_type(cards):
    """Xác định loại tổ hợp bài."""
    logger.debug(f"DEBUG - get_combination_type: cards={cards}")
    if len(cards) == 1:
        logger.debug("DEBUG - get_combination_type: Type is single")
        return "single"
    ranks = [c[0] for c in cards]
    rank_counts = {r: ranks.count(r) for r in ranks}
    logger.debug(f"DEBUG - get_combination_type: rank_counts={rank_counts}")
    if len(cards) == 2 and len(rank_counts) == 1:
        logger.debug("DEBUG - get_combination_type: Type is pair")
        return "pair"
    if len(cards) == 3 and len(rank_counts) == 1:
        logger.debug("DEBUG - get_combination_type: Type is three_of_a_kind")
        return "three_of_a_kind"
    if len(cards) == 4 and len(rank_counts) == 1:
        logger.debug("DEBUG - get_combination_type: Type is four_of_a_kind")
        return "four_of_a_kind"
    if len(cards) >= 3:
        ranks_nums = sorted([RANK_VALUES.get(c[0], -1) for c in cards])
        logger.debug(f"DEBUG - get_combination_type: ranks_nums={ranks_nums}")
        has_two = any(c[0] == '2' for c in cards)
        if has_two and 13 in ranks_nums and 12 in ranks_nums:
            logger.debug("DEBUG - get_combination_type: Invalid straight with K-A-2 sequence")
            return None
        if -1 in ranks_nums:
            logger.error(f"Invalid ranks found: {ranks_nums}, cards: {cards}")
            return None
        for i in range(len(ranks_nums) - 1):
            if ranks_nums[i + 1] != ranks_nums[i] + 1:
                logger.error(f"Non-sequential straight: {ranks_nums}, cards: {cards}")
                return None
        logger.debug("DEBUG - get_combination_type: Type is straight")
        return "straight"
    return None

def is_stronger_combination(new_play, old_play):
    """Kiểm tra xem tổ hợp mới có mạnh hơn tổ hợp cũ không."""
    logger.debug(f"DEBUG - is_stronger_combination: new_play={new_play}, old_play={old_play}")
    new_type = get_combination_type(new_play)
    old_type = get_combination_type(old_play)
    logger.debug(f"DEBUG - is_stronger_combination: new_type={new_type}, old_type={old_type}")
    if new_type == "four_of_a_kind" and old_type == "single" and old_play[0][0] == '2':
        logger.debug("DEBUG - is_stronger_combination: Four of a kind can chop a single 2")
        return True
    if new_type != old_type:
        logger.debug(f"Combination type mismatch: new={new_type}, old={old_type}")
        return False
    if new_type == "straight" and old_type == "straight":
        if len(new_play) < len(old_play):
            logger.debug(f"Straight length too short: new={len(new_play)}, old={len(old_play)}")
            return False
        new_min = min(RANK_VALUES.get(c[0], -1) for c in new_play)
        old_min = min(RANK_VALUES.get(c[0], -1) for c in old_play)
        if new_min <= old_min:
            logger.debug(f"New straight not stronger: new_min={new_min}, old_min={old_min}")
            return False
        return True
    if len(new_play) != len(old_play):
        logger.debug(f"Combination length mismatch: new={len(new_play)}, old={len(old_play)}")
        return False
    new_ranks = sorted([RANK_VALUES.get(c[0], -1) for c in new_play], reverse=True)
    old_ranks = sorted([RANK_VALUES.get(c[0], -1) for c in old_play], reverse=True)
    logger.debug(f"DEBUG - is_stronger_combination: new_ranks={new_ranks}, old_ranks={old_ranks}")
    if new_ranks[-1] <= old_ranks[-1]:
        logger.debug(f"New play not stronger: new_ranks={new_ranks}, old_ranks={old_ranks}")
        return False
    return True