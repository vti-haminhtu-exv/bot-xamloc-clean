# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import torch
import random
import logging
import json
import traceback
from collections import deque
from ..game.game_environment import XamLocSoloEnvironment
from ..game.game_rules import RANK_VALUES, is_stronger_combination, get_combination_type
from .dqn_model import XamLocSoloAI
from .memory import remember, encode_state, encode_action
from .ai_interface import get_valid_actions, predict_action, load, save

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('xam_loc_solo')

class XamLocSoloAssistant:
    def __init__(self, betting_unit=1, model_path="xam_loc_solo_model.pth"):
        """Khởi tạo XamLocSoloAssistant."""
        self.betting_unit = betting_unit
        self.model_path = model_path
        self.env = XamLocSoloEnvironment(betting_unit=betting_unit)
        self.ai = self._initialize_ai()

        # Khởi tạo experience log để lưu trữ thông tin huấn luyện
        if not hasattr(self.ai, 'experience_log'):
            self.ai.experience_log = {
                "win_rate": {"games": 0, "wins": 0},
                "money_history": [],
                "turns_history": [],
                "pass_history": [],
                "xam_stats": {"declared_ai": 0, "success_ai": 0, "declared_opp": 0},
                "timeout_games": 0,
                "logic_error_games": 0,
                "strategy_analysis": {}
            }
        logger.info(f"Initialized assistant: model_path={model_path}, betting_unit={betting_unit}")

    def _initialize_ai(self):
        """Khởi tạo model AI."""
        ai = XamLocSoloAI()
        ai.epsilon = 0.1
        ai.epsilon_decay = 0.9995
        ai.epsilon_min = 0.01
        ai.batch_size = 64
        ai.memory = deque(maxlen=10000)
        ai.gamma = 0.95
        ai.learning_rate = 0.001
        ai.games_played = 0

        # Nếu model đã tồn tại, load nó
        if os.path.exists(self.model_path):
            try:
                load(ai, self.model_path, reset_stats=False, reset_epsilon=False)
                logger.info(f"Loaded existing model: {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"Model not found: {self.model_path}. Starting fresh.")

        return ai

    def train(self, num_episodes=10000, reset_stats_before_train=True, reset_epsilon=True):
        """Huấn luyện AI với số lượng episode nhất định."""
        # Lưu lại cấu hình ban đầu để khôi phục sau khi huấn luyện
        original_batch_size = self.ai.batch_size
        original_epsilon_decay = self.ai.epsilon_decay

        # Reset thống kê nếu cần
        if reset_stats_before_train:
            self.ai.experience_log = {
                "win_rate": {"games": 0, "wins": 0},
                "money_history": [],
                "turns_history": [],
                "pass_history": [],
                "xam_stats": {"declared_ai": 0, "success_ai": 0, "declared_opp": 0},
                "timeout_games": 0,
                "logic_error_games": 0,
                "strategy_analysis": {}
            }

        # Reset epsilon nếu cần
        if reset_epsilon:
            self.ai.epsilon = 1.0  # Bắt đầu với 100% exploration

        # Tăng batch size và điều chỉnh epsilon decay cho quá trình huấn luyện
        self.ai.batch_size = min(128, self.ai.batch_size * 2)
        self.ai.epsilon_decay = max(0.995, self.ai.epsilon_decay)

        # Khởi tạo summary để theo dõi quá trình huấn luyện
        training_summary = {
            "wins": 0,
            "losses": 0,
            "money_earned": 0,
            "money_lost": 0,
            "games_with_xam": 0,
            "xam_success": 0,
            "episode_details": [],
            "strategy_stats": {
                "singles_played": 0,
                "pairs_played": 0,
                "triples_played": 0,
                "straights_played": 0,
                "four_kind_played": 0,
                "total_passes": 0,
                "unnecessary_passes": 0,
                "singles_when_could_pair": 0,
                "high_cards_played": 0,
            },
            "key_decisions": []
        }

        for ep in range(num_episodes):
            self.env = XamLocSoloEnvironment(betting_unit=self.betting_unit)
            state_dict = self.env.reset()
            logger.debug(f"Game {ep}: Initial state - current_player={self.env.current_player}, player_hand={self.env.player_hand}, ai_hand={self.env.ai_hand}")
            done = False
            episode_info = {
                "episode": ep,
                "winner": None,
                "money": 0,
                "xam_declared": None,
                "num_turns": 0,
                "actions": []
            }

            last_hand_size = len(self.env.ai_hand)
            has_broken_straight = False
            has_broken_pair = False

            while not done:
                state = sails._get_state()
                current_player = state["player_turn"]
                logger.debug(f"Game {ep}: Turn - current_player={current_player}, state['player_turn']={state['player_turn']}")
                if state["player_turn"] != self.env.current_player:
                    logger.error(f"Game {ep}: Player turn mismatch - state['player_turn']={state['player_turn']}, env.current_player={self.env.current_player}")
                    done = True
                    break
                hand = self.env.ai_hand if current_player == 1 else self.env.player_hand
                logger.debug(f"Game {ep}: Hand for P{current_player} - {hand}")
                if state["hand"] != hand:
                    logger.error(f"Game {ep}: Hand mismatch - state['hand']={state['hand']}, env.hand={hand}")
                    done = True
                    break
                valid_actions = get_valid_actions(self.ai, state, self.env)
                logger.debug(f"Game {ep}: Valid actions for P{current_player} - {valid_actions}")
                filtered_valid_actions = []
                for action in valid_actions:
                    if action == "pass" or action == "declare_xam":
                        filtered_valid_actions.append(action)
                    elif isinstance(action, list) and all(card in hand for card in action):
                        filtered_valid_actions.append(action)
                    else:
                        logger.error(f"Game {ep}: Train: Removed invalid action {action} not in hand {hand}")
                if not filtered_valid_actions and current_player == 1:
                    filtered_valid_actions = ["pass"]
                    logger.warning(f"Game {ep}: No valid actions found for AI, defaulting to 'pass'")
                elif not filtered_valid_actions and current_player == 0 and hand:
                    filtered_valid_actions = [[hand[0]]]
                    logger.warning(f"Game {ep}: No valid actions found for P0, forcing first card")

                action = predict_action(self.ai, state, filtered_valid_actions, self.env)
                logger.debug(f"Game {ep}: Selected action for P{current_player} - {action}")

                if isinstance(action, list) and not all(card in hand for card in action):
                    logger.error(f"Game {ep}: Train: AI selected invalid action {action} not in hand {hand}")
                    action = "pass"

                # Lưu action để phân tích sau này
                episode_info["actions"].append({
                    "turn": episode_info["num_turns"],
                    "player": current_player,
                    "action": action,
                    "valid_actions": filtered_valid_actions,
                    "hand": hand.copy() if isinstance(hand, list) else hand,
                    "current_play": state.get("current_play", []).copy()
                })

                # Phân tích chiến lược
                if current_player == 1:
                    if action == "pass":
                        training_summary["strategy_stats"]["total_passes"] += 1
                        if len(filtered_valid_actions) > 1:
                            training_summary["strategy_stats"]["unnecessary_passes"] += 1
                    if isinstance(action, list):
                        if len(action) == 1:
                            training_summary["strategy_stats"]["singles_played"] += 1
                            card_rank = action[0][0]
                            matching_cards = [c for c in hand if c[0] == card_rank and c != action[0]]
                            if matching_cards and not state.get("current_play", []):
                                training_summary["strategy_stats"]["singles_when_could_pair"] += 1
                                training_summary["key_decisions"].append({
                                    "episode": ep,
                                    "turn": episode_info["num_turns"],
                                    "type": "missed_pair",
                                    "action": action,
                                    "better_action": [action[0], matching_cards[0]]
                                })
                        elif len(action) == 2 and action[0][0] == action[1][0]:
                            training_summary["strategy_stats"]["pairs_played"] += 1
                        elif len(action) == 3 and action[0][0] == action[1][0] == action[2][0]:
                            training_summary["strategy_stats"]["triples_played"] += 1
                        elif len(action) == 4 and action[0][0] == action[1][0] == action[2][0] == action[3][0]:
                            training_summary["strategy_stats"]["four_kind_played"] += 1
                        elif len(action) >= 3 and all(RANK_VALUES.get(action[i][0], -1) + 1 == RANK_VALUES.get(action[i+1][0], -1) for i in range(len(action)-1)):
                            training_summary["strategy_stats"]["straights_played"] += 1
                        high_cards = sum(1 for c in action if c[0] in ['J', 'Q', 'K', 'A', '2'])
                        if high_cards > 0:
                            training_summary["strategy_stats"]["high_cards_played"] += high_cards

                # Cải thiện reward function
                additional_reward = 0
                if current_player == 1:
                    if isinstance(action, list) and action:
                        new_hand_size = len(hand) - len(action)
                        cards_reduced = last_hand_size - new_hand_size
                        if cards_reduced > 0:
                            additional_reward += 0.5 * cards_reduced
                        last_hand_size = new_hand_size
                        if not state["current_play"]:
                            if len(action) >= 3 and all(
                                    RANK_VALUES.get(action[i+1][0], -1) == RANK_VALUES.get(action[i][0], -1) + 1
                                    for i in range(len(action)-1)):
                                additional_reward += 0.3 * len(action)
                            elif len(action) == 2 and action[0][0] == action[1][0]:
                                additional_reward += 0.2
                            elif len(action) == 1 and RANK_VALUES.get(action[0][0], -1) < RANK_VALUES.get('9', -1):
                                additional_reward += 0.2
                    elif action == "pass" and not state["current_play"]:
                        additional_reward -= 1.0
                    if has_broken_straight:
                        additional_reward -= 0.5
                        has_broken_straight = False
                    if has_broken_pair:
                        additional_reward -= 0.3
                        has_broken_pair = False
                    if isinstance(action, list) and len(action) == 1:
                        has_broken_pair = self._check_if_breaks_pair(hand, action)
                        has_broken_straight = self._check_if_breaks_straight(hand, action)

                next_state, reward, done, info = self.env.step(action)
                logger.debug(f"Game {ep}: After step - done={done}, info={info}")

                reward += additional_reward
                if action == "declare_xam" and current_player == 1:
                    episode_info["xam_declared"] = 1
                elif action == "declare_xam" and current_player == 0:
                    episode_info["xam_declared"] = 0

                remember(self.ai, state, action, reward, next_state, done)
                if len(self.ai.memory) > self.ai.batch_size * 2:
                    self.ai.replay()

                episode_info["num_turns"] += 1
                if done:
                    break

            self.ai.games_played += 1
            self.ai.experience_log["money_history"].append(self.env.money_earned)
            self.ai.experience_log["turns_history"].append(self.env.turn_count)
            self.ai.experience_log["pass_history"].append(self.env.pass_count)
            self.ai.experience_log["win_rate"]["games"] += 1
            episode_info["winner"] = self.env.winner
            episode_info["money"] = self.env.money_earned
            training_summary["episode_details"].append(episode_info)

            if self.env.winner == 1:
                self.ai.experience_log["win_rate"]["wins"] += 1
                training_summary["wins"] += 1
                training_summary["money_earned"] += abs(self.env.money_earned)
            else:
                training_summary["losses"] += 1
                training_summary["money_lost"] += abs(self.env.money_earned)

            if self.env.xam_declared is not None:
                if self.env.xam_declared == 1:
                    self.ai.experience_log["xam_stats"]["declared_ai"] += 1
                    training_summary["games_with_xam"] += 1
                    if self.env.winner == 1:
                        self.ai.experience_log["xam_stats"]["success_ai"] += 1
                        training_summary["xam_success"] += 1
                else:
                    self.ai.experience_log["xam_stats"]["declared_opp"] += 1

            if self.env.turn_count >= self.env.MAX_TURNS_PENALTY_THRESHOLD:
                self.ai.experience_log["timeout_games"] += 1

            if "error" in info and "Severe Logic Error" in info["error"]:
                self.ai.experience_log["logic_error_games"] += 1

            if ep % 10 == 0:
                self.ai.update_target_model()

            if ep % 50 == 0:
                save(self.ai, self.model_path)
                logger.info(f"Checkpoint saved at game {ep}")
                win_rate = training_summary["wins"] / (training_summary["wins"] + training_summary["losses"]) * 100 if (training_summary["wins"] + training_summary["losses"]) > 0 else 0
                net_profit = training_summary["money_earned"] - training_summary["money_lost"]
                xam_success_rate = training_summary["xam_success"] / training_summary["games_with_xam"] * 100 if training_summary["games_with_xam"] > 0 else 0
                logger.info(f"Training progress - Game {ep}/{num_episodes}: Win rate: {win_rate:.2f}%, Net profit: {net_profit}, Xam success: {xam_success_rate:.2f}%")

        self.ai.batch_size = original_batch_size
        self.ai.epsilon_decay = original_epsilon_decay
        save(self.ai, self.model_path)

        total_games = training_summary["wins"] + training_summary["losses"]
        win_rate = training_summary["wins"] / total_games * 100 if total_games > 0 else 0
        net_profit = training_summary["money_earned"] - training_summary["money_lost"]
        profit_per_game = net_profit / total_games if total_games > 0 else 0
        xam_success_rate = training_summary["xam_success"] / training_summary["games_with_xam"] * 100 if training_summary["games_with_xam"] > 0 else 0

        strategy_stats = training_summary["strategy_stats"]
        total_actions = sum([
            strategy_stats["singles_played"],
            strategy_stats["pairs_played"],
            strategy_stats["triples_played"],
            strategy_stats["straights_played"],
            strategy_stats["four_kind_played"]
        ])

        singles_percentage = strategy_stats["singles_played"] / total_actions * 100 if total_actions > 0 else 0
        pairs_percentage = strategy_stats["pairs_played"] / total_actions * 100 if total_actions > 0 else 0
        straights_percentage = strategy_stats["straights_played"] / total_actions * 100 if total_actions > 0 else 0
        singles_when_could_pair_percentage = strategy_stats["singles_when_could_pair"] / strategy_stats["singles_played"] * 100 if strategy_stats["singles_played"] > 0 else 0
        unnecessary_passes_percentage = strategy_stats["unnecessary_passes"] / strategy_stats["total_passes"] * 100 if strategy_stats["total_passes"] > 0 else 0

        summary_text = [
            "===== TRAINING SUMMARY =====",
            f"Total games played: {total_games}",
            f"Wins: {training_summary['wins']} ({win_rate:.2f}%)",
            f"Losses: {training_summary['losses']} ({100-win_rate:.2f}%)",
            f"Total money earned: {training_summary['money_earned']}",
            f"Total money lost: {training_summary['money_lost']}",
            f"Net profit: {net_profit}",
            f"Profit per game: {profit_per_game:.2f}",
            f"Games with Xam declared by AI: {training_summary['games_with_xam']}",
            f"Successful Xam: {training_summary['xam_success']} ({xam_success_rate:.2f}%)",
            f"Current epsilon: {self.ai.epsilon:.6f}",
            "",
            "===== STRATEGY ANALYSIS =====",
            f"Total actions played: {total_actions}",
            f"Singles: {strategy_stats['singles_played']} ({singles_percentage:.2f}%)",
            f"Pairs: {strategy_stats['pairs_played']} ({pairs_percentage:.2f}%)",
            f"Triples: {strategy_stats['triples_played']} ({strategy_stats['triples_played']/total_actions*100:.2f}% if total_actions > 0 else 0)",
            f"Straights: {strategy_stats['straights_played']} ({straights_percentage:.2f}%)",
            f"Four of a kind: {strategy_stats['four_kind_played']} ({strategy_stats['four_kind_played']/total_actions*100:.2f}% if total_actions > 0 else 0)",
            f"Total passes: {strategy_stats['total_passes']}",
            f"Unnecessary passes: {strategy_stats['unnecessary_passes']} ({unnecessary_passes_percentage:.2f}%)",
            f"Singles when could have played pairs: {strategy_stats['singles_when_could_pair']} ({singles_when_could_pair_percentage:.2f}%)",
            f"High cards played (J,Q,K,A,2): {strategy_stats['high_cards_played']}",
            "",
            "===== OPTIMIZATION OPPORTUNITIES =====",
            f"1. {'IMPROVE' if singles_when_could_pair_percentage > 10 else 'GOOD'}: Reduce playing singles when pairs are available ({singles_when_could_pair_percentage:.2f}%)",
            f"2. {'IMPROVE' if unnecessary_passes_percentage > 20 else 'GOOD'}: Reduce unnecessary passes ({unnecessary_passes_percentage:.2f}%)",
            f"3. {'IMPROVE' if pairs_percentage < 20 else 'GOOD'}: Increase use of pairs ({pairs_percentage:.2f}%)",
            f"4. {'IMPROVE' if straights_percentage < 15 else 'GOOD'}: Increase use of straights ({straights_percentage:.2f}%)",
            "============================"
        ]
        for line in summary_text:
            logger.info(line)

        self.ai.experience_log["strategy_analysis"] = {
            "singles_percentage": singles_percentage,
            "pairs_percentage": pairs_percentage,
            "straights_percentage": straights_percentage,
            "singles_when_could_pair_percentage": singles_when_could_pair_percentage,
            "unnecessary_passes_percentage": unnecessary_passes_percentage
        }

        return self.ai, self.ai.experience_log, summary_text

    def _find_straights_in_hand(self, hand):
        """Tìm các sảnh có thể có trong tay bài."""
        if len(hand) < 3:
            return []

        sorted_cards = sorted(hand, key=lambda c: RANK_VALUES.get(c[0], -1))
        straights = []

        for length in range(len(sorted_cards), 2, -1):
            for i in range(len(sorted_cards) - length + 1):
                if sorted_cards[i][0] == '2':
                    continue
                potential_straight = sorted_cards[i:i+length]
                is_straight = True
                for j in range(1, len(potential_straight)):
                    prev_rank = RANK_VALUES.get(potential_straight[j-1][0], -1)
                    curr_rank = RANK_VALUES.get(potential_straight[j][0], -1)
                    if curr_rank != prev_rank + 1 or potential_straight[j][0] == '2':
                        is_straight = False
                        break
                if is_straight and len(potential_straight) >= 3:
                    straights.append(potential_straight)

        return straights

    def _check_if_breaks_straight(self, hand, action):
        """Kiểm tra xem action có làm vỡ sảnh nào không."""
        if not isinstance(action, list) or len(action) != 1:
            return False

        all_straights = self._find_straights_in_hand(hand)
        if not all_straights:
            return False

        new_hand = [card for card in hand if card != action[0]]
        remaining_straights = self._find_straights_in_hand(new_hand)

        for straight in all_straights:
            if action[0] in straight and not any(all(card in s for card in straight if card != action[0]) for s in remaining_straights):
                return True

        return False

    def _check_if_breaks_pair(self, hand, action):
        """Kiểm tra xem action có làm vỡ đôi nào không."""
        if not isinstance(action, list) or len(action) != 1:
            return False

        rank_counts = {}
        for card in hand:
            rank = card[0]
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        action_rank = action[0][0]
        if action_rank in rank_counts and rank_counts[action_rank] == 2:
            return True

        return False

    def analyze_learning(self, window_size=100):
        """Phân tích quá trình học tập của AI."""
        if not self.ai.experience_log["money_history"]:
            return {"error": "No training data available"}

        money_history = self.ai.experience_log["money_history"]
        win_rate_history = []
        avg_money_per_game = []
        recent_games = min(len(money_history), window_size)

        for i in range(len(money_history) - recent_games, len(money_history)):
            window = money_history[max(0, i - window_size + 1):i + 1]
            wins = sum(1 for m in window if m > 0)
            win_rate = wins / len(window) * 100
            avg_money = sum(window) / len(window)
            win_rate_history.append(win_rate)
            avg_money_per_game.append(avg_money)

        strategy_stats = self.ai.experience_log.get("strategy_analysis", {})
        analysis = {
            "recent_win_rate": win_rate_history[-1] if win_rate_history else 0,
            "avg_money_per_game": avg_money_per_game[-1] if avg_money_per_game else 0,
            "win_rate_trend": "increasing" if len(win_rate_history) > 1 and win_rate_history[-1] > win_rate_history[-2] else "stable" if len(win_rate_history) > 1 else "unknown",
            "money_trend": "increasing" if len(avg_money_per_game) > 1 and avg_money_per_game[-1] > avg_money_per_game[-2] else "stable" if len(avg_money_per_game) > 1 else "unknown",
            "strategy_stats": {
                "singles_percentage": strategy_stats.get("singles_percentage", 0),
                "pairs_percentage": strategy_stats.get("pairs_percentage", 0),
                "straights_percentage": strategy_stats.get("straights_percentage", 0),
                "singles_when_could_pair_percentage": strategy_stats.get("singles_when_could_pair_percentage", 0),
                "unnecessary_passes_percentage": strategy_stats.get("unnecessary_passes_percentage", 0)
            },
            "recommendations": []
        }

        if analysis["strategy_stats"]["singles_when_could_pair_percentage"] > 10:
            analysis["recommendations"].append("Reduce playing singles when pairs are available")
        if analysis["strategy_stats"]["unnecessary_passes_percentage"] > 20:
            analysis["recommendations"].append("Reduce unnecessary passes")
        if analysis["strategy_stats"]["pairs_percentage"] < 20:
            analysis["recommendations"].append("Increase use of pairs")
        if analysis["strategy_stats"]["straights_percentage"] < 15:
            analysis["recommendations"].append("Increase use of straights")

        return analysis

    def analyze_game_for_money(self, episode_info):
        """Phân tích một ván chơi cụ thể để tìm cơ hội tối ưu hóa tiền thắng."""
        actions = episode_info.get("actions", [])
        money_earned = episode_info.get("money", 0)
        analysis = {
            "episode": episode_info.get("episode", -1),
            "money_earned": money_earned,
            "missed_opportunities": [],
            "good_decisions": [],
            "critical_mistakes": []
        }

        for action_info in actions:
            if action_info["player"] != 1:
                continue

            action = action_info["action"]
            valid_actions = action_info["valid_actions"]
            hand = action_info["hand"]
            current_play = action_info["current_play"]

            if action == "pass" and len(valid_actions) > 1 and not current_play:
                analysis["missed_opportunities"].append({
                    "turn": action_info["turn"],
                    "description": "Passed on empty board when could play",
                    "alternative_actions": [a for a in valid_actions if a != "pass"]
                })

            if isinstance(action, list) and len(action) == 1:
                card_rank = action[0][0]
                matching_cards = [c for c in hand if c[0] == card_rank and c != action[0]]
                if matching_cards and not current_play:
                    analysis["missed_opportunities"].append({
                        "turn": action_info["turn"],
                        "description": "Played single when could play pair",
                        "alternative_action": [action[0], matching_cards[0]]
                    })

            if isinstance(action, list) and len(action) >= 3 and all(
                    RANK_VALUES.get(action[i+1][0], -1) == RANK_VALUES.get(action[i][0], -1) + 1
                    for i in range(len(action)-1)):
                analysis["good_decisions"].append({
                    "turn": action_info["turn"],
                    "description": f"Played a straight of {len(action)} cards"
                })

            if isinstance(action, list) and any(c[0] in ['J', 'Q', 'K', 'A', '2'] for c in action) and len(hand) > 5:
                analysis["critical_mistakes"].append({
                    "turn": action_info["turn"],
                    "description": "Played high card(s) too early",
                    "cards_played": action
                })

        return analysis

def train_xam_loc_AI(betting_unit=1, num_episodes=50000, model_path="xam_loc_solo_model.pth",
                     reset_stats_before_train=True, reset_epsilon=True):
    """Huấn luyện AI Xâm Lốc với số lượng ván chơi lớn hơn."""
    assistant = XamLocSoloAssistant(betting_unit=betting_unit, model_path=model_path)
    ai, log, summary = assistant.train(num_episodes, reset_stats_before_train, reset_epsilon)
    return ai, log, summary