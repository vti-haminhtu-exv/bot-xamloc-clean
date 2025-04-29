# -*- coding: utf-8 -*-
import queue
import logging

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hàng đợi lưu trữ kết quả huấn luyện
training_queue = queue.Queue()

# Biến toàn cục để lưu log huấn luyện
training_log = []

# Trạng thái huấn luyện
training_status = {"running": False, "progress": 0, "total_games": 0, "start_time": None}