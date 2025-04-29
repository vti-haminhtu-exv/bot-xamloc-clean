# -*- coding: utf-8 -*-
import os
import logging
import random
import traceback
import time
import numpy as np
from copy import deepcopy
# Assuming the relative imports work based on your project structure
# If running this file directly, you might need to adjust imports or run as a module
try:
    from ..ai.ai_training import XamLocSoloAssistant
    from ..game.game_environment import XamLocSoloEnvironment
    from ..game.game_rules import RANK_VALUES, get_combination_type, is_stronger_combination
except ImportError:
    # Fallback for potential direct execution or different structure
    # This might need adjustment based on your actual project setup
    print("Warning: Relative imports failed. Attempting alternative imports (adjust if needed).")
    # Example: Assuming src is the root for these modules
    from ai.ai_training import XamLocSoloAssistant
    from game.game_environment import XamLocSoloEnvironment
    from game.game_rules import RANK_VALUES, get_combination_type, is_stronger_combination


# Sử dụng logger chung
# Configure logging if not already configured elsewhere
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('xam_loc_solo')

class MCTSNode:
    """Node trong cây MCTS."""
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None  # Sẽ được khởi tạo sau

    def uct_value(self, c_param=1.2):  # Giảm từ 1.414 xuống 1.2 để giảm exploration
        """Tính giá trị UCT để lựa chọn node."""
        if self.visits == 0:
            return float('inf')
        # Ensure parent.visits is not zero before division/log
        if self.parent is None or self.parent.visits == 0:
             # Root node or parent not visited yet, treat as infinite potential
             return float('inf')
        # Tránh lỗi log(0) hoặc chia cho 0 nếu self.visits là 0 (mặc dù đã check ở trên)
        if self.visits <= 0:
            return float('inf') # Hoặc một giá trị rất lớn khác
        return self.value / self.visits + c_param * np.sqrt(2 * np.log(self.parent.visits) / self.visits)

    def best_child(self, c_param=1.2):
        """Chọn node con tốt nhất dựa trên UCT."""
        if not self.children:
            return None # No children to choose from
        # Lọc ra những child có visit > 0 để tránh lỗi trong uct_value
        valid_children = [c for c in self.children if c.visits >= 0] # Allow 0 visits for initial calculation
        if not valid_children:
            return None # Hoặc trả về child đầu tiên nếu muốn
        return max(valid_children, key=lambda c: c.uct_value(c_param))

    def expand(self, action, next_state):
        """Mở rộng node hiện tại với action mới."""
        child = MCTSNode(next_state, action, self)
        self.children.append(child)
        return child

    def is_terminal(self):
        """Kiểm tra node có phải là terminal state không."""
        # Check if the hand associated with the current player in the state is empty
        current_player_hand_key = "hand" # Assuming 'hand' always refers to the player whose turn it is conceptually in the node state
        opponent_hand_key = "opponent_cards" # Should maybe be opponent_hand_count

        player_hand_empty = len(self.state.get(current_player_hand_key, [])) == 0
        # Check opponent count if available
        opponent_hand_empty = self.state.get("opponent_hand_count", -1) == 0

        return (player_hand_empty or
                opponent_hand_empty or
                self.state.get("done", False) or
                self.state.get("winner") is not None)


class XamLocSoloDemo(XamLocSoloAssistant):
    def __init__(self, betting_unit=1, model_path="xam_loc_solo_model.pth"):
        """Khởi tạo demo environment với model đã được train"""
        # Initialize the base class (which might load the AI model)
        # Need to handle potential errors during base class init
        try:
            # Assuming base class handles AI loading
            super().__init__(betting_unit=betting_unit, model_path=model_path)
            # If base class doesn't load AI, load it here or ensure it's loaded
            # Example: self.ai = self.load_model(model_path) # Assuming a load_model method
        except Exception as e:
             logger.error(f"Error initializing XamLocSoloAssistant: {e}")
             logger.error(traceback.format_exc())
             # Decide how to handle this - maybe raise the error, or set a flag
             self.ai = None # Indicate AI failed to load

        self.betting_unit = betting_unit # Store betting_unit locally as well
        self.model_path = model_path # Store model_path locally as well

        # Đặt epsilon thấp cho demo if AI was loaded successfully
        if self.ai and hasattr(self.ai, 'epsilon'):
            self.ai.epsilon = 0.01  # Giảm xuống chỉ còn 1% ngẫu nhiên để tận dụng tối đa model đã train
            logger.info(f"Demo initialized: model_path={model_path}, betting_unit={betting_unit}, epsilon={self.ai.epsilon}")
        elif self.ai is None:
             logger.error("AI model could not be loaded. Demo will not function correctly.")
        else:
            logger.warning("AI object exists but does not have 'epsilon' attribute. Check AI model structure.")


    def play_demo_game(self, betting_unit=None, model_path=None, demo_log=None):
        """Chạy một ván chơi demo giữa AI và đối thủ, ghi lại log và phân tích ván đấu."""
        if betting_unit is None:
            betting_unit = self.betting_unit
        # Model path is usually set during init, reloading here might not be intended unless specified
        # if model_path is None:
        #     model_path = self.model_path

        if demo_log is None:
            demo_log = []

        # Khởi tạo lịch sử để phân tích sau ván đấu
        player_hands_history = []  # [(turn, player_id, hand)]
        actions_history = []       # [(action, player_id, hand, valid_actions, current_play)]

        # Ensure AI is available before starting
        if self.ai is None:
             # Check if initialization failed earlier based on self.ai status
             demo_log.append('<span style="color: #FF0000; font-weight: bold;">Lỗi: Không thể tải hoặc khởi tạo mô hình AI. Không thể bắt đầu demo.</span>')
             return demo_log

        # Tạo môi trường demo riêng biệt
        env = XamLocSoloEnvironment(betting_unit=betting_unit)
        env.is_demo = True  # Đánh dấu đây là môi trường demo

        # Reset môi trường để bắt đầu ván mới
        try:
            state_dict = env.reset()
        except Exception as e:
             logger.error(f"Error resetting environment: {e}")
             logger.error(traceback.format_exc())
             demo_log.append(f'<span style="color: #FF0000; font-weight: bold;">Lỗi khi reset môi trường: {e}</span>')
             return demo_log

        # Log thông tin ban đầu
        demo_log.append('<span style="color: #1E90FF; font-weight: bold;">>>> BÀI BAN ĐẦU <<<</span>')
        demo_log.append(f'<span style="color: #000000;">P0 (Người chơi): {self._format_hand(env.player_hand)} ({len(env.player_hand)} lá)</span>')
        demo_log.append(f'<span style="color: #000000;">P1 (AI): {self._format_hand(env.ai_hand)} ({len(env.ai_hand)} lá)</span>')
        demo_log.append(f'<span style="color: #000000;">Lượt đầu: P{env.current_player}</span>')
        demo_log.append("")

        # Lưu bài ban đầu vào lịch sử
        player_hands_history.append((0, 0, env.player_hand.copy()))
        player_hands_history.append((0, 1, env.ai_hand.copy()))

        # Theo dõi trạng thái để phát hiện loop
        p1_pass_count = 0
        p0_pass_count = 0
        last_p1_action = None

        done = False
        turn = 0
        info = {} # Initialize info dict

        # Tăng số lượng iterations MCTS cho demo
        mcts_iterations = 1000  # Tăng từ 500 lên 1000 để có kết quả tốt hơn

        # Tăng độ sâu mô phỏng
        sim_depth = 80  # Tăng từ 50 lên 80 để mô phỏng sâu hơn

        while not done and turn < 100:  # Giới hạn tối đa 100 lượt
            demo_log.append(f'<span style="color: #000080; font-weight: bold;">--- Lượt {turn + 1} ---</span>') # Use turn+1 for 1-based display
            current_player_id = env.current_player
            demo_log.append(f'<span style="color: #000000;">Lượt: P{current_player_id}</span>')
            demo_log.append(f'<span style="color: #000000;">Bàn: {self._format_hand(env.current_play) if env.current_play else "(Trống)"}</span>')

            action = None
            valid_actions = []

            try:
                # Xử lý lượt chơi của P0 (Người chơi - simulated using MCTS) or P1 (AI)
                if current_player_id == 0:
                    current_hand = env.player_hand
                    opponent_hand_len = len(env.ai_hand)
                    pass_count = p0_pass_count
                    log_prefix = "P0"
                else:
                    current_hand = env.ai_hand
                    opponent_hand_len = len(env.player_hand)
                    pass_count = p1_pass_count
                    log_prefix = "P1"

                demo_log.append(f'<span style="color: #000000;">{log_prefix} Bài: {self._format_hand(current_hand)} ({len(current_hand)} lá) | Đối thủ: {opponent_hand_len} lá</span>')

                # Lấy các nước đi hợp lệ
                valid_actions = self._get_valid_actions(env, current_hand)

                # Luôn phải có ít nhất một nước đi nếu còn bài
                if not valid_actions and current_hand:
                    if env.current_play: # Can only pass if there's something to pass on
                        valid_actions = ["pass"]
                        logger.warning(f"{log_prefix} has cards but no generated actions vs non-empty board. Allowing only pass.")
                    else: # Must play something if board is empty and has cards
                         # Heuristic: play lowest single card if forced
                         try:
                            sorted_hand = sorted(current_hand, key=lambda c: RANK_VALUES.get(c[0], -1))
                            if sorted_hand: # Ensure hand is not empty after sorting
                                valid_actions = [[sorted_hand[0]]]
                                logger.warning(f"{log_prefix} has no valid actions generated but has cards and board is empty. Forcing lowest card: {valid_actions[0]}")
                            else: # Should not happen if current_hand is true, but safety check
                                logger.error(f"{log_prefix} has no cards but logic proceeded. Setting action to 'pass'.")
                                valid_actions = ["pass"]
                         except Exception as sort_err:
                            logger.error(f"Error sorting hand for forced play: {sort_err}. Hand: {current_hand}. Forcing pass.")
                            valid_actions = ["pass"]

                # Apply strategic considerations BEFORE MCTS/random choice
                original_valid_actions = valid_actions.copy() # Keep a copy for logging/debugging

                # CẢI TIẾN: Giảm việc bỏ lượt không cần thiết
                # Ưu tiên đánh bài hơn là bỏ lượt khi bàn trống
                if not env.current_play and "pass" in valid_actions and len(valid_actions) > 1:
                    valid_actions_filtered = [a for a in valid_actions if a != "pass"]
                    if valid_actions_filtered: # Only remove pass if other options exist
                        valid_actions = valid_actions_filtered
                        logger.debug(f"{log_prefix} strategy: Removed 'pass' (board empty).")

                # Nếu đã bỏ lượt quá nhiều, bắt buộc đánh nếu có thể
                if pass_count >= 1 and "pass" in valid_actions and len(valid_actions) > 1:  # Giảm từ 2 xuống 1
                    valid_actions_filtered = [a for a in valid_actions if a != "pass"]
                    if valid_actions_filtered: # Only remove pass if other options exist
                        valid_actions = valid_actions_filtered
                        logger.debug(f"{log_prefix} strategy: Removed 'pass' (passed {pass_count} times).")

                # Chọn nước đi
                if not valid_actions: # Should only happen if hand is empty (already won) or error
                     if current_hand: # Error case
                         logger.error(f"{log_prefix} has cards but no valid actions generated (after strategy filtering)! Original: {original_valid_actions}. Forced pass.")
                         action = "pass"
                     else: # Hand is empty, game should have ended. Break loop?
                          logger.warning(f"{log_prefix}'s turn but hand is empty. Should be game over.")
                          done = True
                          continue # Skip to end of loop

                elif len(valid_actions) == 1:
                    action = valid_actions[0]
                else:
                    # CẢI TIẾN: Áp dụng các heuristic trước khi dùng MCTS
                    action = self._apply_heuristics(env, current_hand, valid_actions, opponent_hand_len, current_player_id)

                    if action is None:
                        # Nếu không có heuristic nào áp dụng được, sử dụng MCTS
                        mcts_state = self._create_mcts_state(env, current_player_id)
                        action = self._mcts_search(mcts_state, valid_actions, current_player_id, mcts_iterations, sim_depth)

                # Lưu thông tin hành động và trạng thái hiện tại vào lịch sử để phân tích sau ván đấu
                actions_history.append((action, current_player_id, current_hand.copy(), valid_actions.copy(), env.current_play.copy()))

                # --- Update pass counts and last action (outside the if/else block) ---
                if action == "pass":
                    if current_player_id == 0:
                        p0_pass_count += 1
                    else:
                        p1_pass_count += 1
                        # Check for P1 pass loop mitigation (already present in original code)
                        if last_p1_action == "pass" and len(valid_actions) > 1:
                             non_pass_actions = [a for a in valid_actions if a != "pass"]
                             if non_pass_actions:
                                 logger.info("P1 pass loop mitigation: Forcing non-pass action.")
                                 action = random.choice(non_pass_actions) # Override MCTS choice
                                 p1_pass_count = 0 # Reset count as P1 is forced to play
                else: # Played a card/combination
                    if current_player_id == 0:
                        p0_pass_count = 0
                    else:
                        p1_pass_count = 0

                # Update last P1 action regardless if it was pass or play
                if current_player_id == 1:
                     last_p1_action = action

                # Log action chosen
                log_color = "#32CD32" # Green for play
                formatted_action = self._format_action(action)
                if action == "pass":
                     log_color = "#FFA500" # Orange for pass
                     # Distinguish strategic pass from forced pass
                     if len(original_valid_actions) <= 1 and "pass" in original_valid_actions:
                          demo_log.append(f'<span style="color: {log_color};">{log_prefix} buộc phải bỏ lượt.</span>')
                     elif pass_count >=2 and len(valid_actions) <=1 and "pass" in valid_actions:
                          demo_log.append(f'<span style="color: {log_color};">{log_prefix} bỏ lượt (đã pass {pass_count} lần, bị giới hạn).</span>')
                     else:
                          demo_log.append(f'<span style="color: {log_color};">{log_prefix} bỏ lượt (chiến thuật/MCTS).</span>')
                else:
                     demo_log.append(f'<span style="color: {log_color};">{log_prefix} đánh: {formatted_action}</span>')

                # Thực hiện bước tiếp theo trong môi trường
                logger.info(f"Turn {turn}: P{current_player_id} action={action}. Before step: P0={len(env.player_hand)}, P1={len(env.ai_hand)}")
                # Ensure action is valid before passing to env.step
                # (Technically _get_valid_actions should ensure this, but add safety)
                state, reward, done, info = env.step(action) # Execute the chosen action
                logger.info(f"Turn {turn}: P{current_player_id} action={action}. After step: P0={len(env.player_hand)}, P1={len(env.ai_hand)}, done={done}, info={info}")

                # Lưu bài sau mỗi lượt vào lịch sử
                player_hands_history.append((turn+1, 0, env.player_hand.copy()))
                player_hands_history.append((turn+1, 1, env.ai_hand.copy()))

                # Log thông tin sau khi đánh
                demo_log.append(f'<span style="color: #800080;">P0 Bài còn: {self._format_hand(env.player_hand) if env.player_hand else "(Hết bài)"} ({len(env.player_hand)} lá)</span>')
                demo_log.append(f'<span style="color: #800080;">P1 Bài còn: {self._format_hand(env.ai_hand) if env.ai_hand else "(Hết bài)"} ({len(env.ai_hand)} lá)</span>')

                # Check game end conditions explicitly after step, as env.step might set done=True
                # env.step should ideally set the winner correctly when done becomes True
                if done:
                     if env.winner == 0:
                        demo_log.append('<span style="color: #808080;"> -> P0 hết bài, P0 thắng!</span>')
                     elif env.winner == 1:
                        demo_log.append('<span style="color: #808080;"> -> P1 hết bài, P1 thắng!</span>')
                     # Handle other potential reasons for 'done' if necessary (e.g., error state)

                # Log các thông tin bổ sung from info dict
                if info.get("message"):
                    demo_log.append(f'<span style="color: #808080;"> -> {info["message"]}</span>')
                if info.get("error"):
                    demo_log.append(f'<span style="color: #FF0000;"> -> Lỗi từ Env: {info["error"]}</span>')

            except Exception as e:
                demo_log.append(f'<span style="color: #FF0000; font-weight: bold;">Lỗi nghiêm trọng trong lượt chơi {turn + 1}: {str(e)}</span>')
                logger.error(f"Error during turn {turn + 1} (Player {env.current_player}): {str(e)}")
                logger.error(traceback.format_exc())
                # Include current state for debugging
                try:
                    env_state_log = f' Env State: P{env.current_player}, P0({len(env.player_hand)}), P1({len(env.ai_hand)}), Board: {self._format_hand(env.current_play)}, Action attempted: {self._format_action(action)}, Valid actions calculated: {self._format_action_list(valid_actions)}'
                except:
                    env_state_log = " Env State: Error formatting state for log."
                demo_log.append(f'<span style="color: #FF0000;">{env_state_log}</span>')
                done = True # Stop the game on unhandled exception
                env.winner = None # Indicate error state

            demo_log.append("")
            turn += 1

            # Kiểm tra điều kiện dừng để tránh vòng lặp vô hạn
            if turn >= 100:
                demo_log.append('<span style="color: #FF8C00; font-weight: bold;">Quá 100 lượt, dừng demo để tránh treo!</span>')
                done = True
                if env.winner is None: # If no winner decided yet, determine by card count? Or declare draw?
                    # env should handle this timeout case in its step/check logic ideally
                    if len(env.player_hand) < len(env.ai_hand):
                        env.winner = 0
                    elif len(env.ai_hand) < len(env.player_hand):
                         env.winner = 1
                    else:
                         env.winner = -1 # Indicate a draw/timeout state
                    # Recalculate money/info for timeout state if needed
                    # info = env._calculate_money(info, timeout=True) # Hypothetical


        # Tóm tắt kết quả ván chơi
        demo_log.append('<span style="color: #1E90FF; font-weight: bold;">>>> KẾT THÚC DEMO <<<</span>')

        # Winner determined by env, including timeout (-1) or error (None)
        winner_text = f"P{env.winner}" if env.winner in [0, 1] else ("Hòa/Timeout" if env.winner == -1 else "Không xác định (Lỗi)")
        demo_log.append(f'<span style="color: #000000; font-weight: bold;">Người thắng: {winner_text}</span>')

        # Thông tin về người thua (nếu có)
        loser_hand = []
        loser_id = -1
        if env.winner == 1: # P1 won, P0 lost
            loser_hand = env.player_hand
            loser_id = 0
        elif env.winner == 0: # P0 won, P1 lost
            loser_hand = env.ai_hand
            loser_id = 1

        if loser_id != -1:
            demo_log.append(f'<span style="color: #000000;">Người thua (P{loser_id}) Bài: {self._format_hand(loser_hand) if loser_hand else "(Hết bài)"} ({len(loser_hand)} lá)</span>')
        elif env.winner == -1: # Timeout/Draw
             demo_log.append(f'<span style="color: #000000;">P0 Bài còn: {self._format_hand(env.player_hand)} ({len(env.player_hand)} lá)</span>')
             demo_log.append(f'<span style="color: #000000;">P1 Bài còn: {self._format_hand(env.ai_hand)} ({len(env.ai_hand)} lá)</span>')
        else: # Error
            demo_log.append('<span style="color: #FF0000;">Không xác định được người thua do lỗi.</span>')
            # Log remaining hands in case of error
            demo_log.append(f'<span style="color: #000000;">P0 Bài còn: {self._format_hand(env.player_hand)} ({len(env.player_hand)} lá)</span>')
            demo_log.append(f'<span style="color: #000000;">P1 Bài còn: {self._format_hand(env.ai_hand)} ({len(env.ai_hand)} lá)</span>')

        demo_log.append('<span style="color: #000000;">--------------------</span>')

        # Thông tin về tiền thưởng/phạt - Use info from the last step or recalculate if needed
        # The final info dict from the step where done=True should contain the calculation.
        final_amount = info.get("final_amount_calc", 0)
        try:
            # Chuyển đổi final_amount thành số nếu có thể
            final_amount_value = 0
            if isinstance(final_amount, str) and "Total:" in final_amount:
                final_amount_value = float(final_amount.split("Total:")[1].strip())
            else:
                final_amount_value = float(final_amount) if isinstance(final_amount, (int, float, str)) and str(final_amount).replace('.', '', 1).replace('-', '', 1).isdigit() else 0

            if env.winner == 1: # P1 won, P0 pays P1
                demo_log.append(f'<span style="color: #000000;">Tiền: P0 trả {abs(final_amount_value):.2f} điểm cho P1</span>')
            elif env.winner == 0: # P0 won, P1 pays P0
                demo_log.append(f'<span style="color: #000000;">Tiền: P1 trả {abs(final_amount_value):.2f} điểm cho P0</span>')
            elif env.winner == -1: # Draw or Timeout
                 demo_log.append(f'<span style="color: #000000;">Tiền: Không tính (Hòa/Timeout)</span>')
            else: # Error
                demo_log.append(f'<span style="color: #000000;">Tiền: Không tính (Lỗi)</span>')
        except (ValueError, TypeError) as conv_err:
            # Nếu không thể chuyển đổi thành số, hiển thị giá trị nguyên bản
            logger.error(f"Could not convert final_amount_calc '{final_amount}' to float: {conv_err}")
            demo_log.append(f'<span style="color: #000000;">Tiền: Không thể tính chính xác (giá trị: {final_amount})</span>')

        # Chi tiết điểm from the final 'info' dict
        # Ensure keys exist or provide defaults
        demo_log.append('<span style="color: #000000;">Chi tiết điểm (nếu có):</span>')
        demo_log.append(f'<span style="color: #000000;">  Phạt cơ bản (lá): {info.get("base_penalty_calc", "N/A")}</span>')
        demo_log.append(f'<span style="color: #000000;">  Phạt thối 2: {info.get("thoi_2_penalty_calc", "N/A")}</span>')
        # Add other penalties if they exist in your rules/env
        demo_log.append(f'<span style="color: #000000;">  Phạt tứ quý chặt 2: {info.get("four_of_kind_penalty_calc", "N/A")}</span>') # Assuming this key exists
        demo_log.append(f'<span style="color: #000000;">  Kết quả Xâm: {info.get("xam_result_note", "Không có Xâm")}</span>')
        demo_log.append(f'<span style="color: #000000;">  Tổng điểm cuối cùng: {info.get("final_amount_calc", "N/A")}</span>')

        # Phân tích sau ván đấu
        analysis = self.analyze_game(env, player_hands_history, actions_history)
        demo_log.extend(analysis)

        return demo_log
    # ----------------------------------------------------------------------
    # Helper Methods - Correctly indented as part of the class
    # ----------------------------------------------------------------------

    def _apply_heuristics(self, env, hand, valid_actions, opponent_cards_count, player_id):
        """Áp dụng các heuristic thông minh hơn."""
        current_play = env.current_play

        # 1. Nếu có thể thắng trong 1 nước, đánh luôn
        for action in valid_actions:
            if isinstance(action, list) and len(action) == len(hand):
                return action

        # 2. Nếu bàn trống, ưu tiên đánh đôi thay vì lẻ
        if not current_play:
            # Tìm các đôi
            pairs = [a for a in valid_actions if isinstance(a, list) and len(a) == 2
                    and a[0][0] == a[1][0]]

            # ƯU TIÊN CAO cho đôi thấp
            if pairs:
                low_pairs = [p for p in pairs if RANK_VALUES.get(p[0][0], -1) < RANK_VALUES.get('J', -1)]
                if low_pairs:
                    lowest_pair = min(low_pairs, key=lambda p: RANK_VALUES.get(p[0][0], -1))
                    return lowest_pair

            # Nếu không có đôi thấp, tìm sảnh
            straights = [a for a in valid_actions if isinstance(a, list) and len(a) >= 3
                        and all(RANK_VALUES.get(a[i][0], -1) + 1 == RANK_VALUES.get(a[i+1][0], -1) for i in range(len(a)-1))]
            if straights:
                longest_straight = max(straights, key=len)
                if len(longest_straight) >= 3:
                    return longest_straight

        # 3. Nếu đối thủ sắp hết bài, ưu tiên đánh MẠNH
        if opponent_cards_count <= 3:
            non_pass_actions = [a for a in valid_actions if a != "pass"]
            if non_pass_actions:
                # Đánh bài cao nhất nếu đối thủ sắp hết
                high_cards = sorted([a for a in non_pass_actions if isinstance(a, list)],
                                  key=lambda a: max(RANK_VALUES.get(c[0], -1) for c in a),
                                  reverse=True)
                if high_cards:
                    return high_cards[0]

        return None  # Để MCTS quyết định

    def _format_hand(self, hand):
        """Helper to format a hand of cards consistently."""
        if not hand:
            return "(Không có)"
        # Sort hand for display consistency (optional, but nice)
        try:
            # Sort primarily by rank value, then suit (less important for display)
            sorted_hand = sorted(hand, key=lambda c: (RANK_VALUES.get(c[0], -1), c[1]))
            return ", ".join([f"{card[0]}{card[1]}" for card in sorted_hand])
        except Exception as e: # Fallback if sorting fails (e.g., unexpected card format)
             logger.warning(f"Could not sort hand for display: {hand}. Error: {e}")
             # Cố gắng join dù không sort được
             try:
                 return ", ".join([f"{c[0]}{c[1]}" if isinstance(c, (list, tuple)) and len(c)==2 else str(c) for c in hand])
             except:
                 return str(hand) # Last resort

    def _format_action(self, action):
        """Định dạng hành động để hiển thị trong log."""
        if action == "pass":
            return "Bỏ lượt"
        elif action == "declare_xam":
            return "Báo Xâm"
        elif isinstance(action, list):
            # Format the cards within the action list using _format_hand
            return self._format_hand(action)
        else:
            # Handle unexpected action types
            logger.warning(f"Attempting to format unknown action type: {action} (type: {type(action)})")
            return str(action)

    def _format_action_list(self, actions):
        """Formats a list of actions for logging."""
        if not actions: return "[]"
        return "[" + ", ".join(self._format_action(a) for a in actions) + "]"


    def _get_valid_actions(self, env, hand):
        """Lấy danh sách các nước đi hợp lệ."""
        # This method relies heavily on the game rules. Ensure it's accurate.
        valid_actions = []
        current_play = env.current_play
        hand_size = len(hand)
        if hand_size == 0: return [] # No actions if no cards

        # Sort hand once for efficiency in finding combinations
        try:
             sorted_hand = sorted(hand, key=lambda c: RANK_VALUES.get(c[0], -1))
        except Exception as e:
             logger.error(f"Error sorting hand in _get_valid_actions: {e}. Hand: {hand}")
             return ["pass"] if current_play else [] # Cannot determine plays if hand is invalid

        # Group cards by rank
        rank_groups = {}
        for card in sorted_hand:
            rank = card[0]
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(card)

        # Case 1: Board is empty (player leads)
        if not current_play:
            # Can declare Xam only at the very start of the game (turn 0, no passes)
            # Check env attributes directly
            if env.xam_declared is None and env.turn_count == 0 and env.consecutive_passes == 0:
                # Player must have 10 cards initially to declare Xam? Check rules.
                # Assume any hand size can declare if turn 0.
                valid_actions.append("declare_xam")

            # Add singles
            valid_actions.extend([[card] for card in sorted_hand])

            # Add pairs, triples, four_of_a_kind
            for rank, cards in rank_groups.items():
                if len(cards) >= 2:
                    valid_actions.append(cards[:2]) # Add pair
                if len(cards) >= 3:
                    valid_actions.append(cards[:3]) # Add triple
                if len(cards) >= 4:
                     # Standard Tien Len rules usually don't allow leading with 4-of-a-kind.
                     # Check Xam Loc rules. Assuming allowed for now.
                     valid_actions.append(cards[:4]) # Add four of a kind

            # Add straights (min length 3)
            straights = self._find_straights(sorted_hand) # Pass sorted hand
            valid_actions.extend(straights)

        # Case 2: Board has cards (player must beat current_play or pass)
        else:
            # Always possible to pass if not leading
            valid_actions.append("pass")

            try:
                # Use the more robust is_stronger_combination logic directly
                # Need to find all valid combinations in hand first

                # 1. Find potential plays in hand (singles, pairs, etc.)
                potential_plays = []
                # Singles
                potential_plays.extend([[card] for card in sorted_hand])
                # Pairs, Triples, Four-of-a-kind
                for rank, cards in rank_groups.items():
                    if len(cards) >= 2: potential_plays.append(cards[:2])
                    if len(cards) >= 3: potential_plays.append(cards[:3])
                    if len(cards) >= 4: potential_plays.append(cards[:4])
                # Straights
                potential_plays.extend(self._find_straights(sorted_hand))

                # 2. Filter potential plays: check if they are stronger than current_play
                for play in potential_plays:
                    if is_stronger_combination(play, current_play):
                        valid_actions.append(play)

            except Exception as e:
                logger.error(f"Error determining valid actions against {current_play}: {e}")
                logger.error(traceback.format_exc())
                # Fallback to only allowing pass on error, ensure 'pass' is included
                if "pass" not in valid_actions: valid_actions.append("pass")
                return [a for a in valid_actions if a == "pass"] # Return only pass

        # Remove duplicates using a canonical representation (sorted tuples)
        unique_actions = []
        seen_tuples = set()
        for action in valid_actions:
            if isinstance(action, list):
                # Sort cards within the action first by rank then suit for consistency
                try:
                    sorted_action_cards = sorted(action, key=lambda c: (RANK_VALUES.get(c[0], -1), c[1]))
                    action_tuple = tuple(tuple(card) for card in sorted_action_cards)
                except: # Fallback if card format is weird
                    action_tuple = tuple(str(card) for card in action) # Less reliable

                if action_tuple not in seen_tuples:
                    unique_actions.append(action)
                    seen_tuples.add(action_tuple)
            else: # Handle strings like "pass", "declare_xam"
                action_str = str(action) # Ensure it's a string
                if action_str not in seen_tuples:
                     unique_actions.append(action)
                     seen_tuples.add(action_str)


        return unique_actions


    def _find_straights(self, sorted_hand, min_length=3):
        """Tìm các sảnh trong tay bài (đã sắp xếp)."""
        straights = []
        hand_size = len(sorted_hand)
        if hand_size < min_length:
            return []

        # Use a more efficient way to find straights in a sorted list
        # Store the actual card for each unique rank encountered
        unique_ranks_cards = {}
        for card in sorted_hand:
             # Cannot use '2' in straights in standard Xam/Tien Len
             if card[0] == '2':
                 continue
             rank_val = RANK_VALUES.get(card[0], -1)
             # Keep the lowest suit card for a given rank if duplicates exist? Or just one?
             # Let's just keep the first encountered card for that rank.
             if rank_val not in unique_ranks_cards:
                 unique_ranks_cards[rank_val] = card # Store one card for each rank

        # Get sorted unique rank values that we have cards for
        unique_ranks = sorted(unique_ranks_cards.keys())

        if len(unique_ranks) < min_length:
             return []

        # Iterate through possible start ranks and lengths
        for length in range(min_length, len(unique_ranks) + 1): # Check lengths from min_length up to num unique ranks
            for i in range(len(unique_ranks) - length + 1):
                is_straight_seq = True
                potential_straight_ranks = unique_ranks[i : i + length]

                # Check for consecutive ranks in the sequence
                for j in range(length - 1):
                    if potential_straight_ranks[j+1] != potential_straight_ranks[j] + 1:
                        is_straight_seq = False
                        break

                if is_straight_seq:
                    # Construct the straight using the stored cards for these ranks
                    straight_cards = [unique_ranks_cards[rank_val] for rank_val in potential_straight_ranks]
                    straights.append(straight_cards)

        # Optional: Filter out shorter straights contained within longer ones if needed
        # Example: If found [3,4,5] and [3,4,5,6], only keep [3,4,5,6]?
        # Current logic finds all possible straights of min_length or more. This is usually correct.

        return straights


    def _create_mcts_state(self, env, player_id):
        """Tạo state cho MCTS từ env hiện tại."""
        # The MCTS state should represent the perspective of the player whose turn it is
        if player_id == 0:
            my_hand = env.player_hand.copy()
            opponent_hand_count = len(env.ai_hand) # Need count for simulation eval
            # opponent_hand = env.ai_hand.copy() # Full hand if perfect info MCTS
        else: # player_id == 1
            my_hand = env.ai_hand.copy()
            opponent_hand_count = len(env.player_hand)
            # opponent_hand = env.player_hand.copy() # Full hand if perfect info MCTS

        state = {
            "hand": my_hand,                        # Cards of the player whose turn it is in this MCTS state
            "opponent_hand_count": opponent_hand_count, # Opponent's card count is crucial
            "current_play": env.current_play.copy() if env.current_play else [],
            "consecutive_passes": env.consecutive_passes,
            "player_turn": player_id,               # ID of the player whose turn it IS in this state
            "turn_count": env.turn_count,           # Current turn number
            "last_player": env.last_player,         # ID of the player who last played cards (important for pass logic)
            "xam_declared": env.xam_declared,       # Which player declared Xam (if any)
            "done": False,                          # Game status within MCTS simulation
            "winner": None                          # Winner within MCTS simulation
            # Add opponent hand explicitly if needed for simulation logic (more complex but potentially more accurate)
            # "opponent_cards": opponent_hand # If needed for perfect information simulation
        }
        return state

    def _mcts_search(self, initial_mcts_state, valid_actions, player_id, iterations=1000, sim_depth=80):
        """Tìm nước đi tốt nhất bằng MCTS với cải tiến."""
        if not valid_actions:
            logger.warning(f"MCTS called for P{player_id} with no valid actions. Returning 'pass'.")
            return "pass" # Should ideally not happen if logic before MCTS handles this

        if len(valid_actions) == 1:
             logger.debug(f"MCTS for P{player_id}: Only one valid action ({self._format_action(valid_actions[0])}). Returning it.")
             return valid_actions[0] # No search needed

        # --- CẢI TIẾN: Áp dụng Heuristic NHANH trước MCTS ---
        # (These are simple checks before running the full search)
        # 1. Nếu có thể thắng trong một nước
        my_hand_size = len(initial_mcts_state["hand"])
        for action in valid_actions:
            if isinstance(action, list) and len(action) == my_hand_size:
                logger.info(f"MCTS Pre-Heuristic: Winning move found for P{player_id}. Action: {self._format_action(action)}")
                return action

        # --- Bắt đầu MCTS ---
        root = MCTSNode(initial_mcts_state)
        # Untried actions are specific to the root node's state and player
        root.untried_actions = valid_actions.copy()

        start_time = time.time()
        actual_iterations = 0
        for i in range(iterations):
            actual_iterations += 1
             # Check time limit if necessary
             # if time.time() - start_time > MAX_MCTS_TIME:
             #     logger.warning(f"MCTS timeout after {time.time() - start_time:.2f}s and {i} iterations.")
             #     break

            leaf = self._select(root)

            # If leaf is terminal, backpropagate its actual outcome
            # Use the node's own is_terminal method
            if leaf.is_terminal():
                final_state = leaf.state
                # Determine reward based on who won IN THE LEAF state
                # Perspective is relative to the player whose turn it is AT THE LEAF
                if len(final_state.get("hand", [])) == 0: # Player at leaf won
                    reward = 1.0
                elif final_state.get("opponent_hand_count", -1) == 0: # Opponent at leaf won
                    reward = -1.0
                else: # Draw or unfinished game ended by depth? Evaluate state
                    # Use a simple heuristic for terminal but non-win states (e.g., draw)
                    reward = 0.0 # Neutral outcome for draw/timeout in simulation
                self._backpropagate(leaf, reward)
            else:
                # If leaf can be expanded (has untried actions)
                if leaf.untried_actions:
                    child = self._expand(leaf)
                    if child: # Expansion successful
                        reward = self._simulate(child, depth=sim_depth)
                        self._backpropagate(child, reward)
                    else:
                         # Could not expand (e.g., _apply_action failed, no valid next state?)
                         # Backpropagate neutral reward for the parent leaf
                         logger.warning(f"MCTS: Expansion returned None for leaf. State: {leaf.state}. Backpropagating 0.")
                         self._backpropagate(leaf, 0.0)
                else:
                    # Node is not terminal, but has no untried actions and no children expanded yet.
                    # This suggests all paths from here were explored or led to dead ends?
                    # Or maybe _get_valid_actions_from_state failed for children.
                    # Backpropagate based on current state evaluation or neutral.
                    logger.warning(f"MCTS: Reached non-terminal leaf with no untried_actions and no children. State: {leaf.state}")
                    # Evaluate state heuristically? For now, backpropagate neutral.
                    self._backpropagate(leaf, 0.0) # Or use a state evaluation heuristic

        end_time = time.time()
        logger.debug(f"MCTS for P{player_id} took {end_time - start_time:.3f}s for {actual_iterations} iterations.")

        # Choose the best action based on visits (most robust) or value
        if not root.children:
             logger.warning(f"MCTS for P{player_id}: No children expanded after {actual_iterations} iterations. Root state: {root.state}. Valid actions: {self._format_action_list(valid_actions)}. Falling back to heuristic.")
             # Fallback: use the heuristic choice or random valid action
             # return random.choice(valid_actions) if valid_actions else "pass"
             return self._choose_heuristic_action(initial_mcts_state, valid_actions, player_id)

        # Select child with the highest visit count
        # Ensure visits > 0 to avoid choosing unvisited nodes if possible
        visited_children = [c for c in root.children if c.visits > 0]
        if not visited_children:
             # If no child was visited (e.g., very few iterations, immediate terminal states?)
             # Choose based on raw value or fallback heuristic
             logger.warning(f"MCTS for P{player_id}: No children visited. Choosing based on initial value or heuristic.")
             # best_child = max(root.children, key=lambda c: c.value) # Could use initial value guess? Risky.
             return self._choose_heuristic_action(initial_mcts_state, valid_actions, player_id)

        best_child = max(visited_children, key=lambda c: c.visits)

        # Log MCTS choice and stats
        if best_child and best_child.visits > 0: # Ensure best_child is valid and visited
            avg_value = best_child.value / best_child.visits
            logger.info(f"MCTS Choice for P{player_id}: {self._format_action(best_child.action)} (Visits: {best_child.visits}, AvgValue: {avg_value:.3f})")
            # Log other top options for comparison
            for child in sorted(visited_children, key=lambda c: c.visits, reverse=True)[:5]: # Log top 5 visited
                 if child != best_child:
                      child_avg_value = child.value / child.visits if child.visits > 0 else 0.0
                      logger.debug(f"  - Option: {self._format_action(child.action)} (Visits: {child.visits}, AvgValue: {child_avg_value:.3f})")
            return best_child.action
        else:
            # Should not happen if visited_children is not empty, but as a fallback
            logger.error(f"MCTS for P{player_id}: Could not select best visited child. Root children: {len(root.children)}. Falling back to heuristic.")
            return self._choose_heuristic_action(initial_mcts_state, valid_actions, player_id)

    # ==========================================================================
    # MCTS Helper Methods - Correctly indented
    # ==========================================================================

    def _select(self, node):
        """Lựa chọn node để mở rộng (Selection phase)."""
        current = node
        while not current.is_terminal():
            if current.untried_actions: # If there are actions we haven't tried from this node
                return current # This is the node to expand
            if not current.children: # If no children and no untried actions (shouldn't happen if not terminal?)
                 # This might indicate an issue where a non-terminal state has no valid moves
                 # Or the game ended unexpectedly during simulation/expansion
                 logger.warning(f"MCTS Select: Reached non-terminal node with no untried actions and no children. State: {current.state}")
                 return current # Return the node itself, backpropagation will handle it as terminal/evaluated
            # If all actions are tried, move to the best child according to UCT
            best_next_node = current.best_child()
            if best_next_node is None: # Safety check if best_child returns None (e.g., all children have 0 visits?)
                 logger.error(f"MCTS Select: best_child returned None for node with children. Parent State: {current.state}. Children: {len(current.children)}")
                 # What to do here? Return the parent? Choose random child?
                 # Returning parent might lead to infinite loop if it happens repeatedly.
                 # Let's return the parent, but log error. Backprop might resolve.
                 return current
            current = best_next_node

        # If the loop finishes, it means we reached a terminal node
        return current

    def _expand(self, node):
        """Mở rộng node với một nước đi ngẫu nhiên từ các nước chưa thử (Expansion phase)."""
        if not node.untried_actions:
            logger.warning(f"MCTS Expand called on node with no untried actions. Node state: {node.state}")
            return None # Cannot expand

        # Select an action to expand
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # Simulate the action to get the next state
        # The state in the node is from the perspective of node.state["player_turn"]
        try:
            next_state_dict = self._apply_action(deepcopy(node.state), action)
        except Exception as apply_err:
            logger.error(f"MCTS Expand: Error applying action '{self._format_action(action)}' to state {node.state}: {apply_err}")
            logger.error(traceback.format_exc())
            return None # Cannot create child if action fails

        # Create the child node
        child = node.expand(action, next_state_dict)

        # Initialize untried actions for the child node (for the *next* player)
        # This requires getting valid actions for the state *after* the action was applied.
        # If the child state is terminal, it won't have untried actions.
        if not child.is_terminal():
            try:
                child.untried_actions = self._get_valid_actions_from_state(child.state)
            except Exception as get_actions_err:
                 logger.error(f"MCTS Expand: Error getting valid actions for child state {child.state}: {get_actions_err}")
                 logger.error(traceback.format_exc())
                 child.untried_actions = [] # Set empty list if error occurs
        else:
             child.untried_actions = [] # Terminal nodes have no further actions

        return child

    def _simulate(self, node, depth=80):
        """Mô phỏng trò chơi từ node hiện tại đến khi kết thúc hoặc max depth (Simulation phase)."""
        # Simulate from the perspective of the player whose turn it is *in the node's state*
        # The reward should be calculated from the perspective of the player who *made the move* leading to this node (i.e., node.parent.state.player_turn)
        # Let's keep track of the player whose turn it was at the PARENT of the simulated node (the one who chose the action leading here)
        if node.parent:
            original_player_perspective = node.parent.state["player_turn"]
        else: # Simulating from the root
            original_player_perspective = node.state["player_turn"] # Perspective of the player at the root

        state = deepcopy(node.state)
        current_depth = 0

        while current_depth < depth:
            if self._is_terminal_state(state):
                break

            # Get valid actions for the *current* player in the simulated state
            try:
                valid_actions = self._get_valid_actions_from_state(state)
            except Exception as sim_get_err:
                logger.warning(f"MCTS Simulate: Error getting valid actions in simulation depth {current_depth}. State: {state}. Error: {sim_get_err}")
                # If error getting actions, assume forced pass? Or end simulation?
                state["done"] = True # End simulation on error
                state["winner"] = None # Error state
                break

            if not valid_actions:
                 # If a player has cards but no valid actions, maybe they must pass?
                 # This depends on game rules. Assume pass is always an option if board not empty.
                 # If board is empty and no actions, it's an error state or win?
                 if state["current_play"]: # Can pass if board not empty
                    action = "pass"
                    # logger.debug(f"MCTS Simulate (Depth {current_depth}): P{state['player_turn']} has no actions vs board, forcing pass.")
                 elif state["hand"]: # Board empty, has cards, but no actions? Error.
                    logger.warning(f"MCTS Simulate (Depth {current_depth}): P{state['player_turn']} has cards but no actions generated when board is empty! State: {state}. Ending sim.")
                    state["done"] = True
                    state["winner"] = None
                    break
                 else: # No hand, should be terminal already
                     logger.warning(f"MCTS Simulate (Depth {current_depth}): P{state['player_turn']} has no actions and no hand, should be terminal. State: {state}")
                     break # Already terminal
            else:
                # --- Simulation Strategy cải tiến (Simplified Random + Basic Heuristics) ---
                # Basic strategy: avoid passing if possible, otherwise random. More complex strategies here are costly.
                non_pass_actions = [a for a in valid_actions if a != "pass"]
                if non_pass_actions:
                    # Maybe slightly prefer non-pass?
                    if random.random() < 0.9 or not state["current_play"]: # 90% chance to play if possible, always play if board empty
                        action = random.choice(non_pass_actions)
                    elif "pass" in valid_actions:
                        action = "pass"
                    else: # Only non-pass actions available
                         action = random.choice(non_pass_actions)
                elif "pass" in valid_actions: # Only pass is available
                    action = "pass"
                else:
                    # No actions at all (should have been caught earlier)
                    logger.error(f"MCTS Simulate: Reached state with no valid_actions list. State: {state}")
                    break # Error

            # Apply the chosen action to get the next state
            try:
                state = self._apply_action(state, action)
            except Exception as sim_apply_err:
                 logger.warning(f"MCTS Simulate: Error applying action '{self._format_action(action)}' in simulation depth {current_depth}. State: {state}. Error: {sim_apply_err}")
                 state["done"] = True # End simulation on error
                 state["winner"] = None
                 break

            current_depth += 1
        # Loop ends due to terminal state or max depth

        # --- Evaluate Final State of Simulation ---
        # Return reward (+1 for win, -1 for loss, 0 for draw/timeout) from the perspective of 'original_player_perspective'

        final_winner = state.get("winner") # Winner determined by _apply_action or terminal check

        if final_winner is not None: # Win/Loss determined
             return 1.0 if final_winner == original_player_perspective else -1.0
        else: # Game likely ended due to depth limit or draw/error state
             # Use card count heuristic evaluation
            player_at_end_turn = state["player_turn"]

            # Get card counts relative to the original player's perspective
            if player_at_end_turn == original_player_perspective:
                my_cards_end = len(state.get("hand", []))
                opp_cards_end = state.get("opponent_hand_count", 10)
            else:
                my_cards_end = state.get("opponent_hand_count", 10)
                opp_cards_end = len(state.get("hand", []))

            # Simple evaluation: less cards is better
            if my_cards_end < opp_cards_end:
                return 0.5 # Moderate win signal
            elif opp_cards_end < my_cards_end:
                return -0.5 # Moderate loss signal
            else:
                return 0.0 # Draw


    def _backpropagate(self, node, reward):
        """Lan truyền reward ngược lên cây (Backpropagation phase)."""
        current = node
        # The reward is calculated from the perspective of the player whose *turn it was*
        # right *before* the action leading to the simulated node (or the node itself if terminal).
        # When updating a node, the reward needs to be added if the node represents the player
        # who received the reward, and subtracted if it represents the opponent.
        while current:
            current.visits += 1
            # Check whose turn it is *at the parent* node (the player who made the move leading to `current`)
            # The reward (+1/-1/0) is relative to the player perspective defined in _simulate
            # If the player whose turn it is at the *current* node is *different* from the player
            # who received the positive reward, we negate the reward for this node's value update.
            if current.parent: # Don't adjust reward sign for root based on perspective alone
                 # player_who_made_move = current.parent.state['player_turn'] # This is the perspective the reward is based on
                 # player_at_this_node = current.state['player_turn'] # Turn flips *after* the move usually
                 # If the turn flipped, the value update should reflect the outcome for the player *at the parent*
                 # So, we add the reward directly, as the value reflects the expected outcome *after* the parent plays.
                 current.value += reward
                 # Flip the reward for the next level up (parent node update)
                 reward *= -1.0
            else:
                 # Root node update: The reward is already from the perspective of the root player
                 current.value += reward

            current = current.parent

    def _apply_action(self, state, action):
        """Áp dụng action vào state và trả về state mới (cho MCTS simulation)."""
        # This needs to perfectly mimic the core logic of the game environment's step function
        # but operate on the MCTS state dictionary.
        new_state = state # Modify in place for efficiency? No, deepcopy happened before calling.
        player_id = new_state["player_turn"]
        opponent_id = 1 - player_id

        # Get current hand (must be mutable)
        current_hand = new_state["hand"] # This is already a copy if deepcopy was used before call

        # --- Apply Action ---
        if action == "pass":
            new_state["consecutive_passes"] += 1
            # Determine next player based on pass rules (Tien Len/Xam Loc specific)
            # Rule: If player A plays, B passes -> A plays again, board clears.
            # Rule: If player A plays, B plays, A passes -> B plays again, board stays.
            if new_state["last_player"] is None: # Passing on first turn? Should not happen unless no cards.
                 # Assume turn flips normally if passing when no one has played yet (unlikely)
                 new_state["player_turn"] = opponent_id
            elif new_state["last_player"] != player_id: # The other player played last (e.g. A played, current (B) passes)
                 new_state["player_turn"] = new_state["last_player"] # Player who played last leads again
                 new_state["current_play"] = [] # Board clears
                 new_state["consecutive_passes"] = 0 # Reset passes for new round
            else: # The current player also played last (e.g. A played, B played, current (A) passes)
                 new_state["player_turn"] = opponent_id # The other player (B) gets the turn
                 # Board does NOT clear

        elif action == "declare_xam":
            new_state["xam_declared"] = player_id
            new_state["consecutive_passes"] = 0 # Reset passes
            # Turn does not change yet, Xam player plays first card/combo
            new_state["last_player"] = player_id # Mark declarer as last player to start

        elif isinstance(action, list) and action: # Play cards
            # 1. Validate action against hand (should be pre-validated, but good practice)
            action_cards_tuples = set(tuple(card) for card in action)
            hand_cards_tuples = set(tuple(card) for card in current_hand)
            if not action_cards_tuples.issubset(hand_cards_tuples):
                 # This indicates a bug in valid_actions generation or state corruption
                 logger.error(f"MCTS Apply Action: Invalid action {self._format_action(action)} for hand {self._format_hand(current_hand)} in state {state}")
                 # How to handle? Return original state? Raise error? Treat as pass?
                 # Treating as pass might be safest to continue simulation
                 return self._apply_action(state, "pass") # Recursive call with pass

            # 2. Remove cards from hand
            new_hand = [card for card in current_hand if tuple(card) not in action_cards_tuples]
            new_state["hand"] = new_hand

            # 3. Update board state
            new_state["current_play"] = action # Action itself is the list of cards
            new_state["last_player"] = player_id
            new_state["consecutive_passes"] = 0 # Reset pass count

            # 4. Update opponent hand count - This state only tracks the count, it doesn't change here.

            # 5. Check win condition for current player
            if not new_state["hand"]:
                 new_state["done"] = True
                 new_state["winner"] = player_id
                 new_state["player_turn"] = opponent_id # Turn technically passes, but game is over
                 return new_state # Return immediately as game ended

            # 6. Switch turn if game not done
            new_state["player_turn"] = opponent_id

        else: # Invalid action type
            logger.error(f"MCTS Apply Action: Unknown action type '{action}' received. Treating as pass.")
            return self._apply_action(state, "pass") # Recursive call with pass


        # --- Check opponent win condition (opponent ran out of cards previously) ---
        # This check should ideally be part of is_terminal_state used before action
        # If opponent_hand_count reaches 0, the game should end.
        if new_state["opponent_hand_count"] == 0:
             new_state["done"] = True
             new_state["winner"] = opponent_id # The opponent already won

        # Increment turn count (optional, might not be needed for simulation state)
        new_state["turn_count"] = state.get("turn_count", 0) + 1

        return new_state

    def _get_valid_actions_from_state(self, state):
        """Lấy các nước đi hợp lệ từ state dictionary của MCTS."""
        # This needs opponent hand info only if rules depend on it (like Xam declaration requires full hand?)
        # For basic play, we need current hand, current play.
        hand = state["hand"]
        if not hand: return [] # No actions if no hand

        current_play = state["current_play"]
        player_id = state["player_turn"] # Player whose turn it is in this state

        # Create a mock 'env' or necessary context for the main _get_valid_actions
        # Use a simple object or dictionary that mimics necessary env attributes
        mock_env = type('MockEnv', (object,), {
            'current_play': current_play,
            # Pass empty lists for hands, as _get_valid_actions takes hand as separate arg
            'player_hand': [],
            'ai_hand': [],
            'current_player': player_id,
            'turn_count': state.get("turn_count", 0),
            'consecutive_passes': state.get("consecutive_passes", 0),
            'last_player': state.get("last_player", None),
            'xam_declared': state.get("xam_declared", None)
            # Add other attributes if _get_valid_actions requires them (e.g., betting_unit?)
        })()

        # Call the main validation logic, passing the hand from the state
        return self._get_valid_actions(mock_env, hand)

    def _is_terminal_state(self, state):
        """Kiểm tra xem state dictionary của MCTS có phải là trạng thái kết thúc không."""
        # Check based on the state dictionary structure
        if state.get("done", False): # Explicit done flag takes precedence
            return True
        if state.get("winner") is not None: # Explicit winner flag
             return True

        player_hand_empty = len(state.get("hand", [])) == 0
        # Check opponent hand count if available and reliable
        opponent_hand_empty = state.get("opponent_hand_count", -1) == 0

        return player_hand_empty or opponent_hand_empty


    def _is_straight(self, cards):
        """Kiểm tra xem một tập hợp bài có phải là sảnh hay không."""
        if not isinstance(cards, list) or len(cards) < 3:
            return False

        # Filter out non-card elements just in case
        card_objects = [c for c in cards if isinstance(c, (list, tuple)) and len(c) == 2 and isinstance(c[0], str)]
        if len(card_objects) != len(cards): return False # Invalid input format

        # Check for '2' - not allowed in straights
        if any(c[0] == '2' for c in card_objects):
            return False

        # Lấy và sắp xếp các giá trị rank
        try:
            ranks = sorted([RANK_VALUES.get(c[0], -1) for c in card_objects])
            if -1 in ranks: return False # Invalid rank found
        except:
            return False # Error getting ranks

        # Kiểm tra tính liên tục và không trùng rank
        if len(set(ranks)) != len(ranks): # Check for duplicate ranks (e.g., [[3s], [3d], [4h]] is not straight)
            return False
        for i in range(len(ranks) - 1):
            if ranks[i+1] != ranks[i] + 1:
                return False

        return True

    def _choose_heuristic_action(self, state, valid_actions, player_id):
        """Lựa chọn action dựa trên heuristic đơn giản (fallback for MCTS)."""
        logger.warning(f"P{player_id} falling back to heuristic action selection.")
        if not valid_actions:
            return "pass" # Should have cards if called here, so pass if board exists

        # 1. Win if possible
        my_hand = state["hand"]
        for action in valid_actions:
            if isinstance(action, list) and len(action) == len(my_hand):
                logger.debug(f"Heuristic Fallback: Winning move found: {self._format_action(action)}")
                return action

        current_play = state["current_play"]
        hand_size = len(my_hand)
        opponent_card_count = state.get("opponent_hand_count", 10) # Estimate if unknown

        # Prioritize non-pass actions
        play_actions = [a for a in valid_actions if a != "pass"]

        # 2. Board empty: Lead strategy
        if not current_play:
            # Try to lead with lowest combination (single < pair < straight < triple < 4kind)
            # Sort actions: singles first, then pairs, then straights, etc., then by lowest rank
            def sort_key(action):
                if not isinstance(action, list): return (10, 0) # Place 'pass' or others last
                a_type = get_combination_type(action)
                a_len = len(action)
                min_rank = min(RANK_VALUES.get(c[0], 100) for c in action) # Use 100 for errors

                if a_type == "single": return (0, min_rank)
                if a_type == "pair": return (1, min_rank)
                if a_type == "straight": return (2, a_len, min_rank) # Prioritize longer straights? No, lead low. (2, min_rank)
                if a_type == "three_of_a_kind": return (3, min_rank)
                if a_type == "four_of_a_kind": return (4, min_rank)
                return (5, min_rank) # Unknown type

            sorted_play_actions = sorted(play_actions, key=sort_key)
            if sorted_play_actions:
                chosen_action = sorted_play_actions[0]
                logger.debug(f"Heuristic Fallback (Lead): Choosing lowest action: {self._format_action(chosen_action)}")
                return chosen_action
            else: # Only 'pass' or 'declare_xam'?
                 return valid_actions[0] # Return the first (likely declare_xam or pass)


        # 3. Board has cards: Beat or Pass strategy
        else:
            if not play_actions: # Only option is pass
                return "pass"

            # Try to beat with the smallest possible combination that works
            def beat_sort_key(action):
                 if not isinstance(action, list): return (10, 0) # Should not happen here
                 max_rank = max(RANK_VALUES.get(c[0], -1) for c in action)
                 return max_rank # Simple: beat with lowest max rank card

            sorted_beat_actions = sorted(play_actions, key=beat_sort_key)

            chosen_action = sorted_beat_actions[0]
            logger.debug(f"Heuristic Fallback (Beat): Choosing smallest beating action: {self._format_action(chosen_action)}")
            return chosen_action

        # Absolute fallback if logic above fails somehow
        logger.error("Heuristic fallback reached end without choosing action. Returning random.")
        return random.choice(valid_actions)
    def analyze_game(self, env, player_hands_history, actions_history):
        """Phân tích ván đấu sau khi kết thúc để đưa ra đánh giá và gợi ý cải thiện."""
        analysis = []
        analysis.append('<span style="color: #1E90FF; font-weight: bold;">>>> PHÂN TÍCH VÁN ĐẤU <<<</span>')

        # 1. Phân tích tổng quan
        p0_hand_counts = [len(hand) for turn, pid, hand in player_hands_history if pid == 0]
        p1_hand_counts = [len(hand) for turn, pid, hand in player_hands_history if pid == 1]

        if not p0_hand_counts or not p1_hand_counts:
            analysis.append('<span style="color: #FF0000;">Không đủ dữ liệu để phân tích ván đấu.</span>')
            return analysis

        # Tính tốc độ giảm số lá
        p0_reduction_rate = (p0_hand_counts[0] - p0_hand_counts[-1]) / max(1, len(p0_hand_counts) - 1) if len(p0_hand_counts) > 1 else 0
        p1_reduction_rate = (p1_hand_counts[0] - p1_hand_counts[-1]) / max(1, len(p1_hand_counts) - 1) if len(p1_hand_counts) > 1 else 0

        analysis.append(f'<span style="color: #000000;">Số lượt chơi: {len(actions_history)}</span>')
        analysis.append(f'<span style="color: #000000;">Tốc độ giảm bài P0: {p0_reduction_rate:.2f} lá/lượt</span>')
        analysis.append(f'<span style="color: #000000;">Tốc độ giảm bài P1: {p1_reduction_rate:.2f} lá/lượt</span>')

        # 2. Phân tích các quyết định quan trọng
        analysis.append('<span style="color: #1E90FF; font-weight: bold;">Các quyết định quan trọng:</span>')

        # Thống kê loại bài đánh ra
        p0_actions = [act for act, pid, _, _, _ in actions_history if pid == 0 and act != "pass" and isinstance(act, list)]
        p1_actions = [act for act, pid, _, _, _ in actions_history if pid == 1 and act != "pass" and isinstance(act, list)]

        # Đếm số lần đánh lẻ, đôi, sảnh, etc.
        def count_action_types(actions):
            singles = sum(1 for a in actions if len(a) == 1)
            pairs = sum(1 for a in actions if len(a) == 2 and a[0][0] == a[1][0])
            triples = sum(1 for a in actions if len(a) == 3 and a[0][0] == a[1][0] == a[2][0])
            straights = sum(1 for a in actions if len(a) >= 3 and
                           all(RANK_VALUES.get(a[i][0], -1) + 1 == RANK_VALUES.get(a[i+1][0], -1) for i in range(len(a)-1)))
            four_kind = sum(1 for a in actions if len(a) == 4 and a[0][0] == a[1][0] == a[2][0] == a[3][0])
            return singles, pairs, triples, straights, four_kind

        try:
            p0_singles, p0_pairs, p0_triples, p0_straights, p0_four_kind = count_action_types(p0_actions)
            p1_singles, p1_pairs, p1_triples, p1_straights, p1_four_kind = count_action_types(p1_actions)

            analysis.append(f'<span style="color: #000000;">P0 đánh: {len(p0_actions)} lần - Lẻ: {p0_singles}, Đôi: {p0_pairs}, Bộ ba: {p0_triples}, Sảnh: {p0_straights}, Tứ quý: {p0_four_kind}</span>')
            analysis.append(f'<span style="color: #000000;">P1 đánh: {len(p1_actions)} lần - Lẻ: {p1_singles}, Đôi: {p1_pairs}, Bộ ba: {p1_triples}, Sảnh: {p1_straights}, Tứ quý: {p1_four_kind}</span>')
        except Exception as e:
            analysis.append(f'<span style="color: #FF0000;">Lỗi khi phân tích loại bài: {str(e)}</span>')

        # Phân tích số lần bỏ lượt
        p0_passes = sum(1 for act, pid, _, _, _ in actions_history if pid == 0 and act == "pass")
        p1_passes = sum(1 for act, pid, _, _, _ in actions_history if pid == 1 and act == "pass")

        analysis.append(f'<span style="color: #000000;">P0 bỏ lượt: {p0_passes} lần</span>')
        analysis.append(f'<span style="color: #000000;">P1 bỏ lượt: {p1_passes} lần</span>')

        # 3. Phân tích các cơ hội bỏ lỡ
        analysis.append('<span style="color: #1E90FF; font-weight: bold;">Cơ hội bỏ lỡ:</span>')

        missed_opportunities = []

        for turn_index, (action, player_id, hand, valid_actions, current_play) in enumerate(actions_history):
            turn_num = turn_index + 1  # Lượt hiển thị cho người dùng (1-based)

            try:
                # Kiểm tra bỏ lượt khi có thể đánh
                if action == "pass" and len(valid_actions) > 1:
                    non_pass_actions = [a for a in valid_actions if a != "pass"]
                    if non_pass_actions:
                        # Lọc ra các nước đánh tốt nhất
                        best_actions = []
                        # Ưu tiên đánh khi đối phương sắp hết bài
                        opponent_hand_count = p1_hand_counts[turn_index//2] if player_id == 0 else p0_hand_counts[turn_index//2]
                        if opponent_hand_count <= 3:
                            high_plays = [a for a in non_pass_actions if isinstance(a, list) and
                                         any(RANK_VALUES.get(c[0], -1) >= RANK_VALUES.get('J', -1) for c in a)]
                            if high_plays:
                                best_actions.append(("đánh cao khi đối thủ sắp hết bài", high_plays[0]))

                        # Bàn trống, nên đánh thay vì bỏ lượt
                        if not current_play:
                            pairs = [a for a in non_pass_actions if isinstance(a, list) and len(a) == 2
                                    and a[0][0] == a[1][0]]
                            if pairs:
                                low_pairs = [p for p in pairs if RANK_VALUES.get(p[0][0], -1) < RANK_VALUES.get('J', -1)]
                                if low_pairs:
                                    best_actions.append(("đánh đôi thấp khi bàn trống", min(low_pairs,
                                                       key=lambda p: RANK_VALUES.get(p[0][0], -1))))

                            straights = [a for a in non_pass_actions if isinstance(a, list) and len(a) >= 3
                                        and all(RANK_VALUES.get(a[i][0], -1) + 1 == RANK_VALUES.get(a[i+1][0], -1)
                                             for i in range(len(a)-1))]
                            if straights:
                                best_actions.append(("đánh sảnh dài khi bàn trống", max(straights, key=len)))

                        if best_actions:
                            reason, best_action = best_actions[0]  # Lấy cơ hội tốt nhất
                            missed_opportunities.append(f'<span style="color: #FFA500;">Lượt {turn_num}: P{player_id} bỏ lượt khi có thể {reason}: {self._format_action(best_action)}</span>')

                # Kiểm tra đánh lẻ khi có thể đánh đôi (chỉ khi bàn trống)
                if isinstance(action, list) and len(action) == 1 and not current_play:
                    # Tìm đôi có cùng giá trị
                    card_rank = action[0][0]
                    matching_cards = [c for c in hand if c[0] == card_rank and c != action[0]]

                    if matching_cards:  # Có thể đánh đôi
                        missed_opportunities.append(f'<span style="color: #FF0000;">Lượt {turn_num}: P{player_id} đánh lẻ {self._format_action(action)} thay vì đánh đôi {self._format_action([action[0], matching_cards[0]])}</span>')

                # Kiểm tra xé sảnh
                if isinstance(action, list) and len(action) >= 1:
                    # Tìm kiếm các sảnh tiềm năng trong bài
                    potential_straights = []
                    sorted_hand = sorted(hand, key=lambda c: RANK_VALUES.get(c[0], -1))

                    # Đơn giản hóa: kiểm tra xem card đã đánh có là một phần của sảnh tiềm năng không
                    rank_values = [RANK_VALUES.get(c[0], -1) for c in sorted_hand if c[0] != '2']  # Bỏ qua 2

                    action_ranks = [RANK_VALUES.get(c[0], -1) for c in action if c[0] != '2']

                    for rank in action_ranks:
                        if rank-1 in rank_values and rank+1 in rank_values:
                            # Card này có thể nằm giữa một sảnh
                            missed_opportunities.append(f'<span style="color: #FF8C00;">Lượt {turn_num}: P{player_id} có thể đã xé sảnh tiềm năng khi đánh {self._format_action(action)}</span>')
                            break
            except Exception as e:
                analysis.append(f'<span style="color: #FF0000;">Lỗi khi phân tích cơ hội bỏ lỡ ở lượt {turn_num}: {str(e)}</span>')

        # Hiển thị các cơ hội bỏ lỡ (giới hạn số lượng để không quá dài)
        max_opportunities = 5
        if missed_opportunities:
            for i, opportunity in enumerate(missed_opportunities[:max_opportunities]):
                analysis.append(opportunity)
            if len(missed_opportunities) > max_opportunities:
                analysis.append(f'<span style="color: #000000;">... và {len(missed_opportunities) - max_opportunities} cơ hội bỏ lỡ khác.</span>')
        else:
            analysis.append('<span style="color: #008000;">Không phát hiện cơ hội bỏ lỡ đáng kể.</span>')

        # 4. Phân tích người thắng/thua
        analysis.append('<span style="color: #1E90FF; font-weight: bold;">Phân tích kết quả:</span>')

        if env.winner == 0:
            analysis.append('<span style="color: #008000;">P0 (Người chơi) thắng!</span>')
            # Phân tích lý do thắng
            if p0_reduction_rate > p1_reduction_rate:
                analysis.append('<span style="color: #000000;">- P0 giảm bài nhanh hơn P1, đánh hiệu quả hơn.</span>')
            if p0_straights > p1_straights:
                analysis.append('<span style="color: #000000;">- P0 đánh nhiều sảnh hơn, giúp giảm nhanh số lá.</span>')
            if p0_pairs > p1_pairs:
                analysis.append('<span style="color: #000000;">- P0 tận dụng tốt các cặp đôi.</span>')

        elif env.winner == 1:
            analysis.append('<span style="color: #008000;">P1 (AI) thắng!</span>')
            # Phân tích lý do thắng
            if p1_reduction_rate > p0_reduction_rate:
                analysis.append('<span style="color: #000000;">- P1 giảm bài nhanh hơn P0, đánh hiệu quả hơn.</span>')
            if p1_straights > p0_straights:
                analysis.append('<span style="color: #000000;">- P1 đánh nhiều sảnh hơn, giúp giảm nhanh số lá.</span>')
            if p1_pairs > p0_pairs:
                analysis.append('<span style="color: #000000;">- P1 tận dụng tốt các cặp đôi.</span>')
            if p0_passes > p1_passes:
                analysis.append('<span style="color: #000000;">- P0 bỏ lượt nhiều hơn, mất cơ hội giảm bài.</span>')
        else:
            analysis.append('<span style="color: #FFA500;">Ván đấu kết thúc không rõ người thắng.</span>')

        # 5. Đề xuất cải thiện
        analysis.append('<span style="color: #1E90FF; font-weight: bold;">Đề xuất cải thiện:</span>')

        if p0_singles > p0_pairs and p0_singles > p0_straights:
            analysis.append('<span style="color: #000000;">1. P0 nên ưu tiên đánh đôi và sảnh hơn là đánh lẻ khi bàn trống.</span>')

        if p1_singles > p1_pairs and p1_singles > p1_straights:
            analysis.append('<span style="color: #000000;">1. P1 nên ưu tiên đánh đôi và sảnh hơn là đánh lẻ khi bàn trống.</span>')

        if p0_passes > 3:
            analysis.append('<span style="color: #000000;">2. P0 nên giảm việc bỏ lượt, tận dụng cơ hội đánh bài.</span>')

        if p1_passes > 3:
            analysis.append('<span style="color: #000000;">2. P1 nên giảm việc bỏ lượt, tận dụng cơ hội đánh bài.</span>')

        analysis.append('<span style="color: #000000;">3. Khi đối thủ còn ít bài (≤3 lá), nên ưu tiên đánh bài cao để ép đối phương.</span>')
        analysis.append('<span style="color: #000000;">4. Không đánh lẻ khi có thể đánh đôi, đặc biệt là khi bàn trống.</span>')
        analysis.append('<span style="color: #000000;">5. Tránh xé sảnh tiềm năng khi có lựa chọn khác.</span>')

        return analysis