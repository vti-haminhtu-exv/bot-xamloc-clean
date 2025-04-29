# -*- coding: utf-8 -*-
from flask import Blueprint, request, jsonify, render_template, current_app
import logging
import traceback
import os

# Import mới - đảm bảo sử dụng Demo class thay vì Assistant
from ...assistant.demo import XamLocSoloDemo

# Khởi tạo logger
logger = logging.getLogger('xam_loc_solo')

# Biến global để lưu trữ instance demo
demo_manager = None

# Tạo blueprint
demo_routes = Blueprint('demo_routes', __name__)

# Hàm demo cho backwards compatibility - main.py có thể import hàm này
def demo():
    if request.method == 'POST':
        try:
            betting_unit = int(request.form.get('bet_unit', 1))
            model_path = "xam_loc_solo_model.pth"

            # Khởi tạo demo manager
            global demo_manager
            demo_manager = XamLocSoloDemo(betting_unit=betting_unit, model_path=model_path)

            # Chạy demo game
            demo_result = demo_manager.play_demo_game()

            # Kiểm tra kết quả
            if not isinstance(demo_result, dict):
                return jsonify({
                    "status": "Completed",
                    "log": demo_result if isinstance(demo_result, list) else []
                })

            return jsonify({
                "status": "Completed",
                "log": demo_result.get("log", []),
                "winner": demo_result.get("winner"),
                "money_earned": demo_result.get("money_earned", 0)
            })

        except ValueError:
            return jsonify({"status": "Error", "message": "Đơn vị cược không hợp lệ."}), 400
        except Exception as e:
            logger.error(f"Lỗi demo: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"status": "Error", "message": str(e)}), 500

    return render_template('demo.html')

# Các routes sử dụng blueprint
@demo_routes.route('/demo_bp', methods=['GET'])
def show_demo_page():
    """Hiển thị trang demo."""
    return render_template('demo.html')

@demo_routes.route('/demo_bp', methods=['POST'])
def run_demo():
    """Chạy một ván demo và trả về kết quả."""
    global demo_manager

    try:
        # Lấy thông tin đơn vị cược
        bet_unit = int(request.form.get('bet_unit', 1))
        if bet_unit <= 0:
            return jsonify({"status": "Error", "message": "Đơn vị cược phải lớn hơn 0"}), 400

        # Lấy đường dẫn đến model từ cấu hình ứng dụng
        model_path = current_app.config.get('MODEL_PATH', 'xam_loc_solo_model.pth')

        # Kiểm tra xem model có tồn tại không
        if not os.path.exists(model_path):
            logger.warning(f"Model không tồn tại tại đường dẫn: {model_path}")
            # Vẫn tiếp tục chạy demo với model mặc định

        # Khởi tạo demo manager
        logger.info(f"Khởi tạo demo với betting_unit={bet_unit}, model_path={model_path}")
        demo_manager = XamLocSoloDemo(betting_unit=bet_unit, model_path=model_path)

        # Chạy demo game
        logger.info("Bắt đầu chạy demo game")
        demo_result = demo_manager.play_demo_game()

        # Kiểm tra kết quả
        if not isinstance(demo_result, dict):
            return jsonify({
                "status": "Completed",
                "log": demo_result if isinstance(demo_result, list) else []
            })

        return jsonify({
            "status": "Completed",
            "log": demo_result.get("log", []),
            "winner": demo_result.get("winner"),
            "money_earned": demo_result.get("money_earned", 0)
        })

    except ValueError:
        logger.error("Lỗi giá trị đầu vào không hợp lệ")
        return jsonify({"status": "Error", "message": "Đơn vị cược không hợp lệ."}), 400
    except Exception as e:
        logger.error(f"Lỗi demo: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "Error", "message": str(e)}), 500

@demo_routes.route('/demo/status', methods=['GET'])
def get_demo_status():
    """Trả về trạng thái hiện tại của demo."""
    global demo_manager

    if demo_manager is None:
        return jsonify({"status": "NotInitialized"})

    try:
        # Trả về thông tin cơ bản về demo
        return jsonify({
            "status": "Ready",
            "betting_unit": getattr(demo_manager, 'betting_unit', 1),
            "model_loaded": os.path.exists(getattr(demo_manager, 'model_path', ''))
        })
    except Exception as e:
        logger.error(f"Lỗi khi lấy trạng thái demo: {str(e)}")
        return jsonify({"status": "Error", "message": str(e)}), 500