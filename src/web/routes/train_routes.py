# -*- coding: utf-8 -*-
from flask import request, jsonify, render_template
import threading
import time
import os
import logging
import traceback
# Thêm import module queue
import queue
from src.ai.ai_training import XamLocSoloAssistant
from src.game.game_environment import XamLocSoloEnvironment
from ..app_config import training_queue, training_log, training_status

logger = logging.getLogger('xam_loc_solo')

def train():
    global training_log, training_status
    if request.method == 'POST':
        try:
            if training_status["running"]:
                return jsonify({"status": "Error", "message": "Đã có quá trình huấn luyện đang chạy. Vui lòng đợi hoặc làm mới trang."}), 400

            num_games = int(request.form.get('num_games', 500))
            betting_unit = int(request.form.get('bet_unit', 1))
            reset_stats = request.form.get('reset_stats', 'false').lower() in ['true', 'on', '1', 'yes']
            reset_epsilon = request.form.get('reset_epsilon', 'true').lower() in ['true', 'on', '1', 'yes']
            model_path = "xam_loc_solo_model.pth"

            training_log = []
            training_status = {
                "running": True,
                "progress": 0,
                "total_games": num_games,
                "start_time": time.time()
            }
            training_log.append(f"Bắt đầu huấn luyện: {num_games} ván, đơn vị cược: {betting_unit}, reset_stats: {reset_stats}, reset_epsilon: {reset_epsilon}")

            threading.Thread(target=train_task, args=(
                num_games, betting_unit, model_path, reset_stats, reset_epsilon
            ), daemon=True).start()

            return jsonify({"status": "Training started", "message": "Huấn luyện đang chạy ở chế độ nền. Kiểm tra log để theo dõi tiến trình."})
        except ValueError as e:
            logger.error(f"Lỗi giá trị đầu vào: {str(e)}")
            return jsonify({"status": "Error", "message": f"Giá trị không hợp lệ: {str(e)}"}), 400
        except Exception as e:
            logger.error(f"Lỗi không xác định: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"status": "Error", "message": f"Lỗi: {str(e)}"}), 500
    return render_template('train.html')

def train_task(num_games, betting_unit, model_path, reset_stats, reset_epsilon):
    global training_log, training_status
    try:
        logger.info(f"Bắt đầu huấn luyện: {num_games} ván, đơn vị cược: {betting_unit}, reset_stats: {reset_stats}, reset_epsilon: {reset_epsilon}")

        assistant = XamLocSoloAssistant(betting_unit=betting_unit, model_path=model_path)
        training_log.append("Đã khởi tạo AI Assistant")

        # Xóa sạch self.memory để loại bỏ dữ liệu cũ
        assistant.ai.memory.clear()
        training_log.append("Đã xóa sạch bộ nhớ replay để bắt đầu huấn luyện mới")

        if os.path.exists(model_path):
            training_log.append(f"Tìm thấy model hiện có: {model_path}")
            if reset_stats:
                assistant.ai.load(model_path, reset_stats=True, reset_epsilon=reset_epsilon)
                training_log.append(f"Đã reset thống kê cho model: {model_path}")
            elif reset_epsilon:
                assistant.ai.load(model_path, reset_stats=False, reset_epsilon=True)
                training_log.append(f"Đã reset epsilon cho model: {model_path}")
            else:
                assistant.ai.load(model_path, reset_stats=False, reset_epsilon=False)
                training_log.append(f"Đã tải model hiện có mà không reset: {model_path}")
        else:
            training_log.append(f"Không tìm thấy model: {model_path}. Bắt đầu từ đầu với model mới.")
            if reset_epsilon:
                assistant.ai.epsilon = 1.0
                training_log.append("Đã thiết lập epsilon=1.0 cho model mới")

        sg = assistant.ai.games_played + 1
        eg = sg + num_games - 1
        training_log.append(f"Bắt đầu huấn luyện từ ván {sg} đến ván {eg}")
        logger.info(f"Bắt đầu huấn luyện: {num_games} ván ({sg} -> {eg}), reset_epsilon={reset_epsilon}")

        training_status["progress"] = 0
        training_status["total_games"] = num_games

        for ep in range(sg, eg + 1):
            training_status["progress"] = ep - sg
            env = XamLocSoloEnvironment(betting_unit=betting_unit)
            state_dict = env.reset()
            done = False
            turn_count = 0

            while not done and turn_count < env.MAX_TURNS_PENALTY_THRESHOLD:
                state = env._get_state()
                current_player = state["player_turn"]
                hand = env.ai_hand if current_player == 1 else env.player_hand

                try:
                    valid_actions = assistant.ai.get_valid_actions(state, env)
                    filtered_valid_actions = []
                    for action in valid_actions:
                        if action == "pass" or action == "declare_xam":
                            filtered_valid_actions.append(action)
                        elif isinstance(action, list) and all(card in hand for card in action):
                            filtered_valid_actions.append(action)

                    if not filtered_valid_actions:
                        if current_player == 1:
                            filtered_valid_actions = ["pass"]
                        elif hand:
                            filtered_valid_actions = [[hand[0]]]

                    action = assistant.ai.predict_action(state, filtered_valid_actions, env)
                    action_index = assistant.ai.action_to_index(action, filtered_valid_actions)
                    training_log.append(f"Ván {ep}, lượt {turn_count}: Action = {action}, Action Index = {action_index}")
                    next_state, reward, done, info = env.step(action)
                    assistant.ai.remember(state, action_index, reward, next_state, done)

                    if len(assistant.ai.memory) > assistant.ai.batch_size * 5:
                        assistant.ai.replay()

                    turn_count += 1

                except Exception as e:
                    logger.error(f"Lỗi trong ván {ep}, lượt {turn_count}: {str(e)}\n{traceback.format_exc()}")
                    training_log.append(f"Lỗi trong ván {ep}: {str(e)}")
                    done = True

            assistant.ai.games_played += 1
            assistant.ai.experience_log["money_history"].append(env.money_earned)
            assistant.ai.experience_log["turns_history"].append(env.turn_count)
            assistant.ai.experience_log["pass_history"].append(env.pass_count)
            assistant.ai.experience_log["win_rate"]["games"] += 1

            if env.winner == 1:
                assistant.ai.experience_log["win_rate"]["wins"] += 1

            if env.xam_declared is not None:
                if env.xam_declared == 1:
                    assistant.ai.experience_log["xam_stats"]["declared_ai"] += 1
                    if env.winner == 1:
                        assistant.ai.experience_log["xam_stats"]["success_ai"] += 1
                else:
                    assistant.ai.experience_log["xam_stats"]["declared_opp"] += 1

            assistant.ai.update_target_model()

            if (ep - sg + 1) % 50 == 0 or ep == eg:
                assistant.ai.save(model_path)
                current_win_rate = assistant.ai.experience_log["win_rate"]["wins"] / assistant.ai.experience_log["win_rate"]["games"] * 100
                training_log.append(f"Ván {ep-sg+1}/{num_games}: Tỷ lệ thắng AI = {current_win_rate:.2f}%, epsilon = {assistant.ai.epsilon:.4f}")
                logger.info(f"Ván {ep-sg+1}/{num_games} đã hoàn thành. Tỷ lệ thắng: {current_win_rate:.2f}%")

        training_status["progress"] = num_games
        win_rate = assistant.ai.experience_log["win_rate"]["wins"] / max(assistant.ai.experience_log["win_rate"]["games"], 1) * 100
        avg_money = sum(assistant.ai.experience_log["money_history"]) / max(len(assistant.ai.experience_log["money_history"]), 1)

        result = {
            "status": "Completed",
            "games_played": assistant.ai.games_played,
            "win_rate": win_rate,
            "avg_money": avg_money,
            "epsilon": assistant.ai.epsilon
        }

        elapsed_time = time.time() - training_status["start_time"]
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        training_log.append(f"Hoàn thành huấn luyện {num_games} ván trong {time_str}")
        training_log.append(f"Tỷ lệ thắng AI: {win_rate:.2f}%")
        training_log.append(f"Trung bình tiền/ván: {avg_money:.2f}")
        training_log.append(f"Epsilon cuối cùng: {assistant.ai.epsilon:.6f}")

        training_status["running"] = False
        training_queue.put(result)
        logger.info(f"Huấn luyện hoàn tất: {result}")

    except Exception as e:
        error_msg = f"Lỗi huấn luyện: {str(e)}"
        logger.error(error_msg + "\n" + traceback.format_exc())
        training_log.append(error_msg)
        training_status["running"] = False
        training_queue.put({"status": "Error", "message": error_msg})

def train_status():
    global training_log, training_status
    try:
        result = None
        try:
            result = training_queue.get_nowait()
        except queue.Empty:
            pass

        response = {
            "status": result if result else training_status,
            "log": training_log,
            "progress": training_status
        }

        if training_status["running"] and training_status["start_time"]:
            elapsed_time = time.time() - training_status["start_time"]
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            response["elapsed_time"] = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

            if training_status["progress"] > 0 and training_status["total_games"] > 0:
                progress_ratio = training_status["progress"] / training_status["total_games"]
                if progress_ratio > 0:
                    estimated_total = elapsed_time / progress_ratio
                    remaining = estimated_total - elapsed_time
                    r_hours, r_remainder = divmod(remaining, 3600)
                    r_minutes, r_seconds = divmod(r_remainder, 60)
                    response["remaining_time"] = f"{int(r_hours):02d}:{int(r_minutes):02d}:{int(r_seconds):02d}"

        return jsonify(response)
    except Exception as e:
        logger.error(f"Lỗi khi lấy trạng thái huấn luyện: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "status": {"error": str(e)},
            "log": training_log + [f"Lỗi: {str(e)}"],
            "progress": training_status
        })

def reset_stats():
    try:
        model_path = "xam_loc_solo_model.pth"
        from ..ai.dqn_model import XamLocSoloAI
        tmp_ai = XamLocSoloAI()
        if os.path.exists(model_path):
            if tmp_ai.load(model_path, reset_stats=True, reset_epsilon=False):
                tmp_ai.save(model_path)
                return jsonify({"status": "Completed", "message": "Đã reset thống kê thành công"})
            else:
                return jsonify({"status": "Error", "message": f"Không thể tải model {model_path}"}), 500
        else:
            return jsonify({"status": "Error", "message": f"Không tìm thấy file model {model_path}"}), 404
    except Exception as e:
        logger.error(f"Lỗi reset thống kê: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "Error", "message": str(e)}), 500