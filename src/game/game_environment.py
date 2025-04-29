# -*- coding: utf-8 -*-
import random
import logging
import traceback
from copy import deepcopy
from .game_rules import SUITS, RANKS, RANK_VALUES, CARDS_PER_PLAYER, PENALTY_PER_CARD, \
                       PENALTY_2_OR_FOUR_OF_KIND, REWARD_XAM_SUCCESS, PENALTY_XAM_FAILURE, \
                       MAX_TURNS_PENALTY_THRESHOLD, get_combination_type, is_stronger_combination

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XamLocSoloEnvironment:
    def __init__(self, betting_unit=1):
        self.betting_unit = betting_unit
        self.deck = [(rank, suit) for suit in SUITS for rank in RANKS]
        self.player_hand = []
        self.ai_hand = []
        self.current_play = []
        self.current_player = 0
        self.consecutive_passes = 0
        self.xam_declared = None
        self.last_player = None
        self.money_earned = 0
        self.winner = None
        self.turn_count = 0
        self.pass_count = 0
        self.MAX_TURNS_PENALTY_THRESHOLD = MAX_TURNS_PENALTY_THRESHOLD
        logger.debug("DEBUG - __init__: Environment initialized without reset")

    def reset(self):
        logger.debug("DEBUG - reset: Starting")
        self.deck = [(rank, suit) for suit in SUITS for rank in RANKS]
        random.shuffle(self.deck)
        logger.debug(f"DEBUG - reset: Deck shuffled - {self.deck}")
        self.player_hand = []
        self.ai_hand = []
        self.player_hand = sorted(self.deck[:CARDS_PER_PLAYER], key=lambda c: (RANK_VALUES[c[0]], SUITS.index(c[1])))
        self.ai_hand = sorted(self.deck[CARDS_PER_PLAYER:2*CARDS_PER_PLAYER], key=lambda c: (RANK_VALUES[c[0]], SUITS.index(c[1])))
        logger.debug(f"DEBUG - reset: player_hand={self.player_hand}, ai_hand={self.ai_hand}")
        self.current_play = []
        self.current_player = random.randint(0, 1)
        self.consecutive_passes = 0
        self.xam_declared = None
        self.last_player = None
        self.money_earned = 0
        self.winner = None
        self.turn_count = 0
        self.pass_count = 0
        logger.debug(f"DEBUG - reset: State reset - current_player={self.current_player}, current_play={self.current_play}")
        logger.info(f"Game reset: current_player={self.current_player}")
        state = self._get_state()
        logger.debug(f"DEBUG - reset: Returning state - {state}")
        return state

    def step(self, action):
        info = {}
        done = False
        reward = 0
        logger.debug(f"DEBUG - step: Starting with action={action}")
        try:
            action_value = action
            if isinstance(action, tuple) and action[0] in ("pass", "declare_xam"):
                action_value = action[0]
            logger.debug(f"DEBUG - step: action_value={action_value}")
            if self.current_player == 0 and not self.player_hand:
                logger.error(f"Invalid state: Player 0 has no cards but is playing - player_hand={self.player_hand}")
                info["error"] = "Severe Logic Error: Player 0 has no cards"
                self.winner = 1
                self.money_earned = -PENALTY_XAM_FAILURE * self.betting_unit
                done = True
                return self._get_state(), -200, done, info
            if self.current_player == 1 and not self.ai_hand:
                logger.error(f"Invalid state: Player 1 has no cards but is playing - ai_hand={self.ai_hand}")
                info["error"] = "Severe Logic Error: Player 1 has no cards"
                self.winner = 0
                self.money_earned = -PENALTY_XAM_FAILURE * self.betting_unit
                done = True
                return self._get_state(), -200, done, info
            if action_value != "pass" and action_value != "declare_xam":
                hand = self.player_hand if self.current_player == 0 else self.ai_hand
                logger.debug(f"DEBUG - step: Checking action validity - hand={hand}")
                if not isinstance(action_value, list):
                    logger.error(f"Invalid action type: action={action_value} is not a list, pass, or declare_xam")
                    info["error"] = "Severe Logic Error: Invalid action type"
                    self.winner = 1 if self.current_player == 0 else 0
                    self.money_earned = -PENALTY_XAM_FAILURE * self.betting_unit
                    done = True
                    return self._get_state(), -200, done, info
                missing_cards = [card for card in action_value if card not in hand]
                if missing_cards:
                    logger.error(f"Invalid action: Missing cards {missing_cards} not in hand={hand}")
                    info["error"] = f"Severe Logic Error: Action has cards not in hand: {missing_cards}"
                    self.winner = 1 if self.current_player == 0 else 0
                    self.money_earned = -PENALTY_XAM_FAILURE * self.betting_unit
                    done = True
                    return self._get_state(), -200, done, info
            if action_value == "declare_xam":
                logger.debug("DEBUG - step: Handling 'declare_xam'")
                if self.current_play or self.xam_declared is not None or self.turn_count > 0:
                    logger.error(f"Invalid Xâm declaration: play={self.current_play}, xam_declared={self.xam_declared}, turn_count={self.turn_count}")
                    info["error"] = "Severe Logic Error: Cannot declare Xâm after play starts"
                    self.winner = 1 if self.current_player == 0 else 0
                    self.money_earned = -PENALTY_XAM_FAILURE * self.betting_unit
                    done = True
                    return self._get_state(), -200, done, info
                self.xam_declared = self.current_player
                self.current_player = 1 - self.current_player
                reward += 1
                logger.debug(f"DEBUG - step: Xâm declared - xam_declared={self.xam_declared}, current_player={self.current_player}")
            elif action_value == "pass":
                logger.debug("DEBUG - step: Handling 'pass'")
                if not self.current_play and (self.current_player == 0 and self.player_hand or self.current_player == 1 and self.ai_hand):
                    logger.error(f"Invalid pass: empty table, player_hand={len(self.player_hand)}, ai_hand={len(self.ai_hand)}")
                    info["error"] = "Severe Logic Error: Cannot pass with empty table and cards in hand"
                    self.winner = 1 if self.current_player == 0 else 0
                    self.money_earned = -PENALTY_XAM_FAILURE * self.betting_unit
                    done = True
                    if self.current_player == 0:
                        info["loser_hand"] = self.player_hand
                    else:
                        info["loser_hand"] = self.ai_hand
                    return self._get_state(), -200, done, info
                if self.last_player == self.current_player and self.consecutive_passes >= 1:
                    logger.error(f"Invalid pass: player won last round, consecutive_passes={self.consecutive_passes}")
                    info["error"] = "Severe Logic Error: Cannot pass after winning a round"
                    self.winner = 1 if self.current_player == 0 else 0
                    self.money_earned = -PENALTY_XAM_FAILURE * self.betting_unit
                    done = True
                    if self.current_player == 0:
                        info["loser_hand"] = self.player_hand
                    else:
                        info["loser_hand"] = self.ai_hand
                    return self._get_state(), -200, done, info
                self.consecutive_passes += 1
                self.pass_count += 1
                if self.last_player is not None and self.last_player != self.current_player and self.consecutive_passes > 0:
                    self.current_play = []
                    self.consecutive_passes = 0
                    self.current_player = self.last_player
                    info["message"] = f"P{1-self.current_player} passed. New round for P{self.current_player}."
                    logger.debug(f"DEBUG - step: Pass - new round, current_player={self.current_player}, current_play={self.current_play}")
                else:
                    self.current_player = 1 - self.current_player
                    logger.debug(f"DEBUG - step: Pass - switch player, current_player={self.current_player}")
            else:
                hand = self.player_hand if self.current_player == 0 else self.ai_hand
                logger.info(f"Before play: action={action_value}, hand={hand}")
                if not all(card in hand for card in action_value):
                    missing = [card for card in action_value if card not in hand]
                    logger.error(f"Invalid action: Missing cards {missing} not in hand={hand}")
                    info["error"] = f"Severe Logic Error: Cards not in hand: {missing}"
                    self.winner = 1 if self.current_player == 0 else 0
                    self.money_earned = -PENALTY_XAM_FAILURE * self.betting_unit
                    done = True
                    return self._get_state(), -200, done, info
                try:
                    if not self._is_valid_play(action_value, hand):
                        logger.error(f"Invalid play: action={action_value}, hand={hand}")
                        info["error"] = "Severe Logic Error: Invalid play attempted"
                        self.winner = 1 if self.current_player == 0 else 0
                        self.money_earned = -PENALTY_XAM_FAILURE * self.betting_unit
                        done = True
                        if self.current_player == 0:
                            info["loser_hand"] = self.player_hand
                        else:
                            info["loser_hand"] = self.ai_hand
                        return self._get_state(), -200, done, info
                except Exception as e:
                    logger.error(f"Error in _is_valid_play: {e}")
                    info["error"] = f"Severe Logic Error: {str(e)}"
                    self.winner = 1 if self.current_player == 0 else 0
                    self.money_earned = -PENALTY_XAM_FAILURE * self.betting_unit
                    done = True
                    return self._get_state(), -200, done, info
                if self._would_end_with_two(hand, action_value):
                    logger.error(f"Cannot end with a 2: action={action_value}")
                    info["error"] = "Severe Logic Error: Cannot end with a 2"
                    self.winner = 1 if self.current_player == 0 else 0
                    self.money_earned = -PENALTY_XAM_FAILURE * self.betting_unit * 2
                    done = True
                    if self.current_player == 0:
                        info["loser_hand"] = self.player_hand
                    else:
                        info["loser_hand"] = self.ai_hand
                    return self._get_state(), -200, done, info
                hand_copy = hand.copy()
                for card in action_value:
                    hand_copy.remove(card)
                if self.current_player == 0:
                    self.player_hand = hand_copy
                else:
                    self.ai_hand = hand_copy
                self.current_play = action_value
                self.last_player = self.current_player
                self.consecutive_passes = 0
                logger.info(f"P{self.current_player} played {action_value}, new hand={hand_copy}")
                reward += min(len(action_value), 3)
                new_hand_size = len(hand_copy)
                if new_hand_size <= 3:
                    reward += (4 - new_hand_size) * 2
                if self.xam_declared is not None and self.xam_declared != self.current_player and len(action_value) > 0:
                    self.winner = self.current_player
                    if self.current_player == 1:
                        reward += REWARD_XAM_SUCCESS * 2
                    self.money_earned = REWARD_XAM_SUCCESS * self.betting_unit
                    done = True
                if self.current_player == 0 and not self.player_hand:
                    self.winner = 0
                    done = True
                    self._calculate_money(info)
                    reward -= 50
                elif self.current_player == 1 and not self.ai_hand:
                    self.winner = 1
                    done = True
                    self._calculate_money(info)
                    reward += 50
                    if self.turn_count < 20:
                        reward += (20 - self.turn_count)
                self.current_player = 1 - self.current_player
                logger.debug(f"DEBUG - step: After play - current_player={self.current_player}, current_play={self.current_play}")
            self.turn_count += 1
            if self.turn_count >= self.MAX_TURNS_PENALTY_THRESHOLD:
                logger.error(f"Timeout after {self.turn_count} turns")
                self.winner = 1 if self.current_player == 0 else 0
                done = True
                self.money_earned = -PENALTY_XAM_FAILURE * self.betting_unit
                info["error"] = f"Timeout after {self.turn_count} turns"
                if self.current_player == 0:
                    info["loser_hand"] = self.player_hand
                else:
                    info["loser_hand"] = self.ai_hand
                return self._get_state(), -50, done, info
            logger.info(f"Step end: player_hand={len(self.player_hand)} cards, ai_hand={len(self.ai_hand)} cards, done={done}")
            return self._get_state(), reward, done, info
        except Exception as e:
            logger.error(f"Unhandled exception in step: {e}")
            logger.error(traceback.format_exc())
            info["error"] = f"System error: {str(e)}"
            return self._get_state(), -200, True, info

    def _calculate_money(self, info):
        loser_hand = self.ai_hand if self.winner == 0 else self.player_hand
        penalty = len(loser_hand) * self.betting_unit * PENALTY_PER_CARD
        n2 = sum(1 for c in loser_hand if c[0] == '2')
        thoi_2_penalty = n2 * self.betting_unit * PENALTY_2_OR_FOUR_OF_KIND * 2
        penalty += thoi_2_penalty
        thoi_4 = sum(1 for ct in {r: sum(1 for c in loser_hand if c[0] == r) for r, _ in loser_hand}.values() if ct == 4)
        thoi_4_penalty = thoi_4 * self.betting_unit * PENALTY_2_OR_FOUR_OF_KIND
        penalty += thoi_4_penalty
        xam_result = 0
        if self.xam_declared is not None:
            if self.xam_declared == self.winner:
                xam_result = REWARD_XAM_SUCCESS * self.betting_unit
            else:
                xam_result = -PENALTY_XAM_FAILURE * self.betting_unit
        self.money_earned = penalty + xam_result if self.winner == 1 else -(penalty + xam_result)
        info["base_penalty_calc"] = f"{len(loser_hand)}c*{PENALTY_PER_CARD}*{self.betting_unit}={penalty-self.betting_unit*(thoi_2_penalty+thoi_4_penalty)}"
        info["thoi_2_penalty_calc"] = f"{n2}*{PENALTY_2_OR_FOUR_OF_KIND*2}*{self.betting_unit}={thoi_2_penalty}"
        info["thoi_4_penalty_calc"] = f"{thoi_4}*{PENALTY_2_OR_FOUR_OF_KIND}*{self.betting_unit}={thoi_4_penalty}"
        info["xam_result_note"] = f"Xâm {'thành công' if xam_result > 0 else 'thất bại'}: {xam_result}" if self.xam_declared is not None else "Không có Xâm"
        info["final_amount_calc"] = f"Total: {abs(self.money_earned)}"

    def _get_state(self):
        hand = self.ai_hand if self.current_player == 1 else self.player_hand
        opponent_cards_count = len(self.player_hand) if self.current_player == 1 else len(self.ai_hand)
        state = {
            "hand": hand,
            "current_play": self.current_play,
            "opponent_cards_count": opponent_cards_count,
            "player_turn": self.current_player,
            "consecutive_passes": self.consecutive_passes,
            "xam_declared": self.xam_declared,
            "last_player": self.last_player,
            "money_earned": self.money_earned,
            "winner": self.winner,
            "turn_count": self.turn_count,
            "pass_count": self.pass_count
        }
        logger.debug(f"DEBUG - _get_state: Returning state - {state}")
        return state

    def _is_valid_play(self, play, hand):
        logger.debug(f"DEBUG - _is_valid_play: Starting with play={play}, hand={hand}")
        if not play or not all(card in hand for card in play):
            logger.error(f"Invalid play: play={play} not in hand={hand}")
            return False
        play_type = get_combination_type(play)
        logger.debug(f"DEBUG - _is_valid_play: play_type={play_type}")
        if play_type is None:
            logger.error(f"Invalid combination: play={play}")
            return False
        if not self.current_play:
            logger.debug("DEBUG - _is_valid_play: Empty table, play is valid")
            return True
        try:
            result = is_stronger_combination(play, self.current_play)
            logger.debug(f"DEBUG - _is_valid_play: is_stronger_combination result={result}")
            return result
        except ValueError as e:
            logger.debug(f"DEBUG - _is_valid_play: ValueError in is_stronger_combination - {str(e)}")
            return False

    def _would_end_with_two(self, hand, play):
        if set(play) == set(hand):
            return any(c[0] == '2' for c in play)
        return False