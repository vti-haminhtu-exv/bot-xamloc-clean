# -*- coding: utf-8 -*-
from flask import request, render_template, jsonify
import logging
from ...ai.ai_training import XamLocSoloAssistant
from ...assistant.move_suggestion import parse_cards

# Sử dụng logger chung
logger = logging.getLogger('xam_loc_solo')

def suggest():
    if request.method == 'POST':
        try:
            hand_str = request.form.get('hand', '')
            current_play_str = request.form.get('current_play', '')
            opponent_cards = int(request.form.get('opponent_cards', 10))
            xam_declared = request.form.get('xam_declared', None)
            last_player = request.form.get('last_player', None)
            consecutive_passes = int(request.form.get('consecutive_passes', 0))
            model_path = "xam_loc_solo_model.pth"
            player_hand = parse_cards(hand_str)
            current_play = parse_cards(current_play_str)
            if not player_hand:
                raise ValueError("Bài trên tay không hợp lệ")
            if opponent_cards < 0 or opponent_cards > 10:
                raise ValueError("Số lá bài đối thủ phải từ 0 đến 10")
            xam_declared = int(xam_declared) if xam_declared in ['0', '1'] else None
            last_player = int(last_player) if last_player in ['0', '1'] else None
            assistant = XamLocSoloAssistant(model_path=model_path)
            suggestion = assistant.suggest_move(
                player_hand=player_hand,
                current_play=current_play,
                opponent_cards=opponent_cards,
                xam_declared=xam_declared,
                last_player=last_player,
                consecutive_passes=consecutive_passes
            )
            return jsonify({"status": "Completed", "suggestion": suggestion})
        except ValueError as e:
            return jsonify({"status": "Error", "message": str(e)}), 400
        except Exception as e:
            logger.error(f"Lỗi gợi ý: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"status": "Error", "message": str(e)}), 500
    return render_template('suggest.html')