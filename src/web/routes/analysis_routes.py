# -*- coding: utf-8 -*-
from flask import jsonify
import logging
import os
from ...ai.dqn_model import XamLocSoloAI
from ...ai.ai_interface import load, save

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze():
    try:
        model_path = "xam_loc_solo_model.pth"
        tmp_ai = XamLocSoloAI()
        analysis = []
        if os.path.exists(model_path):
            if load(tmp_ai, model_path, reset_stats=False, reset_epsilon=False):
                analysis = tmp_ai.analyze_learning()
                tmp_ai.plot_money_history()
            else:
                analysis.append(f"Error loading {model_path}")
        else:
            analysis.append(f"File not found: {model_path}")
        return jsonify({"status": "Completed", "analysis": analysis})
    except Exception as e:
        logger.error(f"Lỗi phân tích: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "Error", "message": str(e)}), 500