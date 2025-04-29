# -*- coding: utf-8 -*-
from flask import Flask
import logging
from logging.handlers import RotatingFileHandler
from .routes.train_routes import train, train_status, reset_stats
from .routes.suggest_routes import suggest
from .routes.demo_routes import demo
from .routes.analysis_routes import analyze

# Cấu hình logging
logger = logging.getLogger('xam_loc_solo')
logger.setLevel(logging.DEBUG)

# Tạo formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Handler ghi log ra console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Handler ghi log vào file (RotatingFileHandler để giới hạn kích thước file)
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10*1024*1024, backupCount=5)  # Giới hạn 10MB, giữ 5 file backup
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = Flask(__name__)

# Đăng ký các route
app.add_url_rule('/', view_func=train, methods=['GET'])
app.add_url_rule('/train', view_func=train, methods=['GET', 'POST'])
app.add_url_rule('/train_status', view_func=train_status)
app.add_url_rule('/reset_stats', view_func=reset_stats, methods=['POST'])
app.add_url_rule('/suggest', view_func=suggest, methods=['GET', 'POST'])
app.add_url_rule('/demo', view_func=demo, methods=['GET', 'POST'])
app.add_url_rule('/analyze', view_func=analyze)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)