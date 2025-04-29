import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import traceback
import importlib.util
from typing import List, Tuple, Dict, Any, Optional, Union
import random
import queue

# Phương thức import động - sẽ tìm và import file chính
def import_source_file(file_path="main.py"):
    """Import file nguồn động"""
    try:
        if not os.path.exists(file_path):
            # Tìm kiếm file trong thư mục hiện tại
            possible_files = [f for f in os.listdir() if f.endswith('.py') and 'xam' in f.lower()]
            if possible_files:
                file_path = possible_files[0]
                print(f"Đã tìm thấy file nguồn: {file_path}")
            else:
                raise FileNotFoundError(f"Không tìm thấy file nguồn {file_path}")

        # Import file nguồn động
        module_name = os.path.basename(file_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        print(f"Đã nạp thành công file nguồn: {file_path}")
        return module
    except Exception as e:
        print(f"Lỗi khi nhập file nguồn: {e}")
        traceback.print_exc()
        messagebox.showerror("Lỗi Import", f"Không thể nạp file nguồn: {e}")
        return None

# Nhập mô-đun nguồn
source_module = import_source_file("main.py")

# Nếu import thành công, lấy các lớp và hằng số cần thiết
if source_module:
    XamLocSoloEnvironment = source_module.XamLocSoloEnvironment
    XamLocSoloAI = source_module.XamLocSoloAI
    XamLocSoloAssistant = source_module.XamLocSoloAssistant
    SUITS = source_module.SUITS
    RANKS = source_module.RANKS
    RANK_VALUES = source_module.RANK_VALUES
    CARDS_PER_PLAYER = source_module.CARDS_PER_PLAYER
    parse_cards = source_module.parse_cards
else:
    # Backup: Nếu không thể import, hiển thị thông báo và thoát
    messagebox.showerror("Lỗi Import", "Không thể nạp file nguồn. Vui lòng đảm bảo main.py nằm trong cùng thư mục.")
    # Tạo dummy classes để tránh lỗi khi compile
    class DummyClass: pass
    XamLocSoloEnvironment = XamLocSoloAI = XamLocSoloAssistant = DummyClass
    SUITS = RANKS = []
    RANK_VALUES = {}
    CARDS_PER_PLAYER = 10
    def parse_cards(card_str): return []

class RedirectText:
    def __init__(self, text_widget):
        self.output = text_widget

    def write(self, string):
        self.output.configure(state='normal')
        self.output.insert(tk.END, string)
        self.output.see(tk.END)
        self.output.configure(state='disabled')

    def flush(self):
        pass

class XamLocSoloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Xâm Lốc Solo AI - Giao diện")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        # Biến lưu trữ trạng thái
        self.model_file = "xam_loc_solo_model.pth"
        self.assistant = None
        self.training_thread = None
        self.is_training = False
        self.demo_log_queue = queue.Queue()

        print("Bắt đầu tạo giao diện...")  # Debug
        # Tạo giao diện
        try:
            self.create_widgets()
            print("Giao diện đã được tạo.")  # Debug
        except Exception as e:
            print(f"Lỗi khi tạo giao diện: {e}")
            traceback.print_exc()
            messagebox.showerror("Lỗi", f"Không thể tạo giao diện: {e}")
            return

        # Khởi tạo Assistant
        print("Khởi tạo Assistant...")  # Debug
        try:
            self.assistant = XamLocSoloAssistant(model_path=self.model_file)
            self.log(f"Đã tải mô hình từ: {self.model_file}\n")
        except Exception as e:
            self.log(f"Không thể tải mô hình: {e}\n")
            print(f"Lỗi khi khởi tạo Assistant: {e}")
            traceback.print_exc()

        # Bắt đầu xử lý hàng đợi log
        print("Bắt đầu xử lý hàng đợi log...")  # Debug
        self.process_demo_log_queue()

    def create_widgets(self):
        # Tạo notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Huấn luyện
        self.tab_train = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text="Huấn luyện")
        self.create_train_tab()

        # Tab 2: Demo
        self.tab_demo = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_demo, text="Demo")
        self.create_demo_tab()

        # Tab 3: Gợi ý
        self.tab_suggest = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_suggest, text="Gợi ý")
        self.create_suggest_tab()

        # Tab 4: Phân tích
        self.tab_analyze = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analyze, text="Phân tích")
        self.create_analyze_tab()

        # Frame dưới cùng để hiển thị log
        self.log_frame = ttk.LabelFrame(self.root, text="Log")
        self.log_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

        # Vùng hiển thị log
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.configure(state='disabled')

        # Chuyển hướng stdout để hiển thị trong log_text
        self.stdout_redirect = RedirectText(self.log_text)
        sys.stdout = self.stdout_redirect

        # Thanh trạng thái
        self.status_var = tk.StringVar(value="Sẵn sàng")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_train_tab(self):
        frame = ttk.Frame(self.tab_train)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame cấu hình
        config_frame = ttk.LabelFrame(frame, text="Cấu hình huấn luyện")
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        # Số lượng trò chơi
        ttk.Label(config_frame, text="Số lượng trò chơi:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.games_var = tk.StringVar(value="500")
        ttk.Entry(config_frame, textvariable=self.games_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # Đơn vị cược
        ttk.Label(config_frame, text="Đơn vị cược:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.bet_var = tk.StringVar(value="1")
        ttk.Entry(config_frame, textvariable=self.bet_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # Đặt lại thống kê
        self.reset_stats_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Đặt lại thống kê trước khi huấn luyện", variable=self.reset_stats_var).grid(
            row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

        # File mô hình
        ttk.Label(config_frame, text="File mô hình:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        model_frame = ttk.Frame(config_frame)
        model_frame.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        self.model_path_var = tk.StringVar(value=self.model_file)
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=30).pack(side=tk.LEFT)
        ttk.Button(model_frame, text="...", width=3, command=self.browse_model).pack(side=tk.LEFT, padx=2)

        # Frame điều khiển
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=5, pady=10)

        # Nút bắt đầu huấn luyện
        self.train_button = ttk.Button(control_frame, text="Bắt đầu huấn luyện", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=5)

        # Nút dừng huấn luyện
        self.stop_button = ttk.Button(control_frame, text="Dừng huấn luyện", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Nút đặt lại thống kê
        ttk.Button(control_frame, text="Đặt lại thống kê", command=self.reset_stats).pack(side=tk.LEFT, padx=5)

        # Thanh tiến trình
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, padx=5, pady=5)

        # Frame hiển thị thông tin huấn luyện
        info_frame = ttk.LabelFrame(frame, text="Thông tin huấn luyện")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.train_info_text = scrolledtext.ScrolledText(info_frame)
        self.train_info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.train_info_text.configure(state='disabled')

    def create_demo_tab(self):
        frame = ttk.Frame(self.tab_demo)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame cấu hình demo
        config_frame = ttk.LabelFrame(frame, text="Cấu hình Demo")
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        # Đơn vị cược
        ttk.Label(config_frame, text="Đơn vị cược:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.demo_bet_var = tk.StringVar(value="1")
        ttk.Entry(config_frame, textvariable=self.demo_bet_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # Nút bắt đầu demo
        ttk.Button(config_frame, text="Bắt đầu Demo (AI vs AI)", command=self.start_demo).grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        # Frame hiển thị trò chơi
        game_frame = ttk.LabelFrame(frame, text="Diễn biến trò chơi")
        game_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.demo_text = scrolledtext.ScrolledText(game_frame)
        self.demo_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.demo_text.configure(state='disabled')

    def create_suggest_tab(self):
        frame = ttk.Frame(self.tab_suggest)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame đầu vào
        input_frame = ttk.LabelFrame(frame, text="Thông tin bàn chơi")
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        # Bài của bạn
        ttk.Label(input_frame, text="Bài của bạn (ví dụ: 3s,4c,Jd...):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.hand_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.hand_var, width=40).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # Bài trên bàn
        ttk.Label(input_frame, text="Bài trên bàn (để trống nếu không có):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.table_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.table_var, width=40).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # Chế độ bài đối thủ
        ttk.Label(input_frame, text="Chế độ bài đối thủ:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.opp_mode_var = tk.StringVar(value="Nhập số lượng")
        opp_mode_combo = ttk.Combobox(input_frame, textvariable=self.opp_mode_var, width=15)
        opp_mode_combo['values'] = ('Nhập số lượng', 'Random')
        opp_mode_combo.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        opp_mode_combo.bind('<<ComboboxSelected>>', self.toggle_opp_cards_input)

        # Số bài của đối thủ
        self.opp_cards_label = ttk.Label(input_frame, text=f"Số bài của đối thủ (0-{CARDS_PER_PLAYER}):")
        self.opp_cards_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.opp_cards_var = tk.StringVar(value=str(CARDS_PER_PLAYER))
        self.opp_cards_entry = ttk.Entry(input_frame, textvariable=self.opp_cards_var, width=5)
        self.opp_cards_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        # Người chơi cuối cùng
        ttk.Label(input_frame, text="Người chơi cuối cùng:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.last_player_var = tk.StringVar(value="None")
        last_player_combo = ttk.Combobox(input_frame, textvariable=self.last_player_var, width=10)
        last_player_combo['values'] = ('None', 'Bạn (0)', 'Đối thủ (1)')
        last_player_combo.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)

        # Số lượt bỏ
        ttk.Label(input_frame, text="Số lượt bỏ liên tiếp:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.passes_var = tk.StringVar(value="0")
        ttk.Entry(input_frame, textvariable=self.passes_var, width=5).grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)

        # Xâm đã báo
        ttk.Label(input_frame, text="Xâm đã báo:").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        self.xam_var = tk.StringVar(value="None")
        xam_combo = ttk.Combobox(input_frame, textvariable=self.xam_var, width=10)
        xam_combo['values'] = ('None', 'Bạn (0)', 'Đối thủ (1)')
        xam_combo.grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)

        # Nút gợi ý
        ttk.Button(input_frame, text="Lấy gợi ý", command=self.get_suggestion).grid(row=7, column=0, columnspan=2, padx=5, pady=5)

        # Frame hiển thị gợi ý
        suggestion_frame = ttk.LabelFrame(frame, text="Gợi ý của AI")
        suggestion_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.suggestion_text = scrolledtext.ScrolledText(suggestion_frame)
        self.suggestion_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.suggestion_text.configure(state='disabled')

    def toggle_opp_cards_input(self, event=None):
        """Bật/tắt trường nhập số bài đối thủ dựa trên chế độ"""
        if self.opp_mode_var.get() == "Random":
            self.opp_cards_entry.configure(state='disabled')
            self.opp_cards_label.configure(text="Số bài của đối thủ: (Random)")
        else:
            self.opp_cards_entry.configure(state='normal')
            self.opp_cards_label.configure(text=f"Số bài của đối thủ (0-{CARDS_PER_PLAYER}):")

    def random_opponent_cards(self, player_hand, table_cards, max_cards=CARDS_PER_PLAYER):
        """Sinh ngẫu nhiên bài đối thủ, không trùng với bài người chơi và bài trên bàn"""
        try:
            # Tạo bộ bài đầy đủ
            full_deck = [(r, s) for r in RANKS for s in SUITS]
            # Loại bỏ bài của người chơi và bài trên bàn
            used_cards = set(player_hand) | set(table_cards or [])
            available_deck = [card for card in full_deck if card not in used_cards]

            if not available_deck:
                return None, 0  # Không đủ bài để random

            # Random số lượng bài đối thủ (1 đến max_cards)
            num_cards = random.randint(1, min(max_cards, len(available_deck)))
            # Random các lá bài
            opponent_cards = random.sample(available_deck, num_cards)
            return opponent_cards, num_cards
        except Exception as e:
            self.log(f"Lỗi khi random bài đối thủ: {e}\n")
            return None, 0

    def create_analyze_tab(self):
        frame = ttk.Frame(self.tab_analyze)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame điều khiển
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(control_frame, text="Phân tích dữ liệu", command=self.analyze_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Xem biểu đồ", command=self.show_chart).pack(side=tk.LEFT, padx=5)

        # Frame kết quả phân tích
        analyze_result_frame = ttk.LabelFrame(frame, text="Kết quả phân tích")
        analyze_result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Chia làm 2 phần: thông tin phân tích và biểu đồ
        self.analyze_paned = ttk.PanedWindow(analyze_result_frame, orient=tk.HORIZONTAL)
        self.analyze_paned.pack(fill=tk.BOTH, expand=True)

        # Phần thông tin phân tích
        info_frame = ttk.Frame(self.analyze_paned)
        self.analyze_text = scrolledtext.ScrolledText(info_frame)
        self.analyze_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.analyze_text.configure(state='disabled')
        self.analyze_paned.add(info_frame, weight=40)

        # Phần biểu đồ
        self.chart_frame = ttk.Frame(self.analyze_paned)
        self.analyze_paned.add(self.chart_frame, weight=60)

        # Tạo một figure cho matplotlib
        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        """Ghi log vào vùng log_text"""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

    def update_status(self, message):
        """Cập nhật thanh trạng thái"""
        self.status_var.set(message)

    def browse_model(self):
        """Hiển thị hộp thoại chọn file mô hình"""
        file_path = filedialog.askopenfilename(
            title="Chọn file mô hình",
            filetypes=[("PyTorch Model", "*.pth"), ("Tất cả files", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
            self.model_file = file_path

    def start_training(self):
        """Bắt đầu huấn luyện AI trong một luồng riêng"""
        if self.is_training:
            messagebox.showwarning("Đang huấn luyện", "AI đang trong quá trình huấn luyện.")
            return

        try:
            num_games = int(self.games_var.get())
            bet_unit = int(self.bet_var.get())
            reset_stats = self.reset_stats_var.get()
            model_path = self.model_path_var.get()

            # Xác nhận nếu reset_stats = True
            if reset_stats and not messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn đặt lại thống kê trước khi huấn luyện không?"):
                return

            # Xóa thông tin cũ
            self.train_info_text.configure(state='normal')
            self.train_info_text.delete(1.0, tk.END)
            self.train_info_text.configure(state='disabled')

            # Cập nhật trạng thái
            self.is_training = True
            self.train_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.update_status(f"Đang huấn luyện: 0/{num_games} trò chơi...")

            # Tạo và bắt đầu luồng huấn luyện
            self.training_thread = threading.Thread(
                target=self._training_thread_func,
                args=(num_games, bet_unit, reset_stats, model_path)
            )
            self.training_thread.daemon = True
            self.training_thread.start()

        except ValueError as e:
            messagebox.showerror("Lỗi đầu vào", f"Vui lòng nhập số hợp lệ: {e}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi bắt đầu huấn luyện: {e}")
            traceback.print_exc()

    def _training_thread_func(self, num_games, bet_unit, reset_stats, model_path):
        """Hàm chạy trong luồng huấn luyện"""
        try:
            # Tạo một assistant mới
            assistant = XamLocSoloAssistant(model_path=model_path, betting_unit=bet_unit)

            # Đặt lại thống kê nếu được yêu cầu
            if reset_stats:
                assistant.ai.load(model_path, reset_stats=True)

            # Thiết lập để cập nhật GUI
            original_train = assistant.train

            # Custom training function để cập nhật GUI
            def custom_train(assistant_self, num_episodes=200, verbose=True, save_interval=100):
                env = XamLocSoloEnvironment(assistant_self.betting_unit)
                sg = assistant_self.ai.games_played + 1
                eg = sg + num_episodes - 1
                current_game = 0

                self.add_train_info(f"--- Bắt đầu huấn luyện: {num_episodes} trò chơi ({sg} -> {eg}) ---\n")
                self.add_train_info(f"Device: {assistant_self.ai.device}\n")

                for ep in range(sg, eg + 1):
                    if not self.is_training:  # Kiểm tra nếu đã dừng
                        self.add_train_info("Huấn luyện bị dừng bởi người dùng.\n")
                        break

                    state_dict = env.reset()
                    done = False
                    is_to = False
                    is_le = False
                    ls_ai = None
                    la_ai = None
                    last_reward = 0

                    while not done:
                        cpid = env.current_player
                        sai_p = env._get_state()

                        if cpid == 1:
                            sv_act = sai_p
                        else:
                            sv_act = {
                                "hand": env.player_hand.copy(),
                                "current_play": env.current_play.copy(),
                                "opponent_cards_count": len(env.ai_hand),
                                "player_turn": 0,
                                "consecutive_passes": env.consecutive_passes,
                                "xam_declared": env.xam_declared,
                                "last_player": env.last_player,
                                "turn_count": env.turn_count,
                                "pass_count": env.pass_count
                            }

                        vacts = assistant_self.ai.get_valid_actions(sv_act, env)

                        if not vacts:
                            act = "pass"
                            rew = -500
                            done = True
                            is_le = True
                            info = {"error": "No valid actions"}
                            self.add_train_info(f"!!! Lỗi: Không có hành động hợp lệ! Trò chơi {ep}.\n")
                        else:
                            st = assistant_self.ai.encode_state(sai_p)
                            act = assistant_self.ai.predict_action(st, vacts)

                        if cpid == 1:
                            ls_ai = sai_p
                            la_ai = act

                        try:
                            next_s, rew, done_step, info = env.step(act)
                            done = done_step
                            last_reward = rew
                            if info.get("error", "").startswith("Severe Logic Error"):
                                self.add_train_info(f"!!! Lỗi: {info['error']} | Trò chơi {ep}.\n")
                                done = True
                                is_le = True
                                last_reward = -500
                        except Exception as e:
                            self.add_train_info(f"!!! Lỗi khi thực thi bước: {str(e)} | Trò chơi {ep}.\n")
                            rew = -1000
                            done = True
                            is_le = True
                            info = {"error": f"Runtime error: {e}"}
                            last_reward = rew

                        if cpid == 1 and ls_ai is not None:
                            assistant_self.ai.remember(ls_ai, la_ai, rew, next_s, done, is_timeout=False, is_logic_error=is_le)

                        if not done and env.turn_count > 50:
                            self.add_train_info(f"Cảnh báo: Trò chơi {ep} kết thúc vì quá nhiều lượt ({env.turn_count}).\n")
                            done = True
                            is_to = True
                            info = info if info else {}
                            info["warning"] = "Timeout"
                            final_s = env._get_state()
                            final_s["winner"] = None

                            if ls_ai is not None and la_ai is not None:
                                assistant_self.ai.remember(ls_ai, la_ai, -30, final_s, done, is_timeout=True, is_logic_error=False)

                            last_reward = -30

                        if len(assistant_self.ai.memory) > assistant_self.ai.batch_size * 5 and env.turn_count % 4 == 0:
                            assistant_self.ai.replay()

                    # End of Episode - Logging
                    final_turns = env.turn_count
                    final_passes = env.pass_count
                    final_winner = env.winner
                    final_money = env.money_earned

                    money_hist_reward = 0
                    if not is_to and not is_le and final_winner is not None:
                        money_hist_reward = final_money if final_winner == 1 else -final_money
                    elif is_to:
                        money_hist_reward = -30
                    elif is_le:
                        money_hist_reward = last_reward

                    log = assistant_self.ai.experience_log
                    log.setdefault("money_history", []).append(money_hist_reward)
                    log.setdefault("turns_history", []).append(final_turns)
                    log.setdefault("pass_history", []).append(final_passes)

                    if final_winner is not None and not is_to and not is_le:
                        wl = log.setdefault("win_rate", {"games": 0, "wins": 0})
                        wl["games"] += 1
                        if final_winner == 1:
                            wl["wins"] += 1

                    final_xam_declared = env.xam_declared
                    xs = log.setdefault("xam_stats", {"declared_ai": 0, "success_ai": 0, "declared_opp": 0})
                    last_known_xam = ls_ai.get("xam_declared") if ls_ai else final_xam_declared

                    if last_known_xam == 1:
                        xs["declared_ai"] += 1
                        if final_winner == 1 and not is_to and not is_le:
                            xs["success_ai"] += 1
                    elif last_known_xam == 0:
                        xs["declared_opp"] += 1

                    if is_to:
                        log["timeout_games"] = log.get("timeout_games", 0) + 1
                    if is_le:
                        log["logic_error_games"] = log.get("logic_error_games", 0) + 1

                    assistant_self.ai.games_played += 1
                    current_game += 1

                    # Cập nhật GUI
                    progress_percent = (current_game / num_episodes) * 100
                    self.root.after(0, self.update_progress, progress_percent)
                    self.root.after(0, self.update_status, f"Đang huấn luyện: {current_game}/{num_episodes} trò chơi...")

                    if len(assistant_self.ai.memory) > assistant_self.ai.batch_size * 5:
                        assistant_self.ai.replay()

                    assistant_self.ai.update_target_model()

                    if verbose and (ep % 10 == 0 or ep == eg):
                        outcome = f"Timeout({final_turns})" if is_to else f"Error({final_turns})" if is_le else f"P{final_winner} thắng ({final_turns} lượt, {final_passes} bỏ lượt)"
                        log_msg = f"Trò chơi {ep}/{eg}: Kết quả={outcome}, Phần thưởng AI={money_hist_reward:.0f}, Eps={assistant_self.ai.epsilon:.4f}, Mem={len(assistant_self.ai.memory)}\n"
                        self.root.after(0, self.add_train_info, log_msg)

                    if ep % save_interval == 0 or ep == eg:
                        analysis = assistant_self.ai.analyze_learning()
                        analysis_text = "\n--- Phân tích @Trò chơi " + str(ep) + " ---\n"
                        for i in analysis:
                            analysis_text += f"- {i}\n"
                        analysis_text += "---------------------------\n\n"
                        self.root.after(0, self.add_train_info, analysis_text)

                        assistant_self.ai.save(assistant_self.model_path)
                        try:
                            assistant_self.ai.plot_money_history()
                            self.root.after(0, self.add_train_info, "Đã lưu biểu đồ tiền thưởng.\n")
                        except Exception as plot_err:
                            self.root.after(0, self.add_train_info, f"Lỗi khi vẽ biểu đồ: {str(plot_err)}\n")

                self.root.after(0, self.add_train_info, f"--- Huấn luyện kết thúc @trò chơi {assistant_self.ai.games_played}. ---\n")
                assistant_self.ai.save(assistant_self.model_path)
                try:
                    assistant_self.ai.plot_money_history()
                except Exception as plot_err:
                    self.root.after(0, self.add_train_info, f"Lỗi khi vẽ biểu đồ cuối cùng: {str(plot_err)}\n")

                return assistant_self.ai.experience_log

            # Thay thế phương thức train
            assistant.train = custom_train.__get__(assistant, type(assistant))

            # Huấn luyện
            assistant.train(num_episodes=num_games, verbose=True, save_interval=100)

            # Cập nhật assistant chính nếu đã tồn tại
            if self.assistant:
                self.assistant = assistant

            # Kết thúc huấn luyện
            self.root.after(0, self._training_finished)
        except Exception as e:
            self.root.after(0, lambda: self._training_error(str(e)))
            traceback.print_exc()

    def _training_finished(self):
        """Gọi khi huấn luyện kết thúc"""
        self.is_training = False
        self.train_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.update_status("Huấn luyện hoàn tất.")
        messagebox.showinfo("Hoàn tất", "Huấn luyện AI đã hoàn tất!")

    def _training_error(self, error_msg):
        """Gọi khi có lỗi trong quá trình huấn luyện"""
        self.is_training = False
        self.train_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.update_status("Lỗi huấn luyện!")
        messagebox.showerror("Lỗi huấn luyện", f"Lỗi: {error_msg}")

    def stop_training(self):
        """Dừng quá trình huấn luyện"""
        if not self.is_training:
            return

        if messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn dừng huấn luyện không?"):
            self.is_training = False
            self.update_status("Đang dừng huấn luyện...")

    def update_progress(self, value):
        """Cập nhật thanh tiến trình"""
        self.progress_var.set(value)

    def add_train_info(self, text):
        """Thêm thông tin vào vùng thông tin huấn luyện"""
        self.train_info_text.configure(state='normal')
        self.train_info_text.insert(tk.END, text)
        self.train_info_text.see(tk.END)
        self.train_info_text.configure(state='disabled')

    def reset_stats(self):
        """Đặt lại thống kê mà không huấn luyện"""
        if not messagebox.askyesno("Xác nhận", f"Bạn có chắc chắn muốn đặt lại thống kê trong '{self.model_file}' không?\nLưu ý: Điều này sẽ giữ nguyên trọng số mô hình."):
            return

        try:
            tmp_ai = XamLocSoloAI()
            if tmp_ai.load(self.model_file, reset_stats=True):
                tmp_ai.save(self.model_file)
                messagebox.showinfo("Thành công", "Đã đặt lại thống kê thành công!")
            else:
                messagebox.showerror("Lỗi", f"Không thể tải {self.model_file} để đặt lại thống kê.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi đặt lại thống kê: {e}")
            traceback.print_exc()

    def start_demo(self):
        """Bắt đầu chơi demo giữa hai AI"""
        try:
            # Lấy đơn vị cược
            bet_unit = int(self.demo_bet_var.get())

            # Xóa thông tin demo cũ
            self.demo_text.configure(state='normal')
            self.demo_text.delete(1.0, tk.END)
            self.demo_text.configure(state='disabled')

            # Tạo một luồng riêng để chạy demo
            threading.Thread(target=self._run_demo, args=(bet_unit,), daemon=True).start()

        except ValueError:
            messagebox.showerror("Lỗi đầu vào", "Vui lòng nhập đơn vị cược hợp lệ.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi bắt đầu demo: {e}")
            traceback.print_exc()

    def _run_demo(self, bet_unit):
        """Chạy trò chơi demo trong một luồng riêng"""
        try:
            # Redirect stdout để thu thập thông tin
            original_stdout = sys.stdout

            class DemoRedirect:
                def __init__(self, gui):
                    self.gui = gui
                    self.buffer = ""

                def write(self, string):
                    self.buffer += string
                    if '\n' in string:
                        self.gui.demo_log_queue.put(self.buffer)
                        self.buffer = ""

                def flush(self):
                    if self.buffer:
                        self.gui.demo_log_queue.put(self.buffer)
                        self.buffer = ""

            sys.stdout = DemoRedirect(self)

            self.demo_log_queue.put(f"=== BẮT ĐẦU DEMO (AI vs AI) ===\n")
            self.demo_log_queue.put(f"Đơn vị cược: {bet_unit}\n")
            self.demo_log_queue.put(f"Mô hình: {self.model_file}\n\n")
            self.update_status("Đang chạy demo...")

            # Chuẩn bị môi trường
            try:
                env = XamLocSoloEnvironment(betting_unit=bet_unit)
                env.reset()
                assistant = XamLocSoloAssistant(model_path=self.model_file, betting_unit=bet_unit)

                # Lưu epsilon hiện tại và đặt về 0 để tránh ngẫu nhiên
                eps = assistant.ai.epsilon
                assistant.ai.epsilon = 0.0

                self.demo_log_queue.put(f"P0 (Người chơi): {assistant._format_action(env.player_hand)}\n")
                self.demo_log_queue.put(f"P1 (AI): {assistant._format_action(env.ai_hand)}\n")
                self.demo_log_queue.put(f"Lượt đầu: P{env.current_player}\n")
                self.demo_log_queue.put(f"{'-'*40}\n")

                done = False
                info = {}
                err = False

                while not done:
                    pid = env.current_player
                    self.demo_log_queue.put(f"\n--- Lượt {env.turn_count} ---\n")
                    pname = f"P{pid}"
                    self.demo_log_queue.put(f"Lượt: {pname}\n")
                    self.demo_log_queue.put(f"Bàn: {assistant._format_action(env.current_play) if env.current_play else '(Trống)'}\n")

                    state_ai = env._get_state()
                    if pid == 1:
                        h = env.ai_hand
                        oc = len(env.player_hand)
                        sv = state_ai
                    else:
                        h = env.player_hand
                        oc = len(env.ai_hand)
                        sv = {
                            "hand": h.copy(),
                            "current_play": env.current_play.copy(),
                            "opponent_cards_count": oc,
                            "player_turn": 0,
                            "consecutive_passes": env.consecutive_passes,
                            "xam_declared": env.xam_declared,
                            "last_player": env.last_player,
                            "turn_count": env.turn_count,
                            "pass_count": env.pass_count
                        }

                    self.demo_log_queue.put(f"{pname} Bài ({len(h)}): {assistant._format_action(h)} | Đối thủ: {oc}\n")

                    valid = assistant.ai.get_valid_actions(sv, env)
                    if not valid:
                        self.demo_log_queue.put(f"!!! LỖI DEMO: {pname} không còn nước đi hợp lệ!\n")
                        act = "pass"
                        err = True
                        info = {"error": "No valid actions"}
                        done = True
                    else:
                        st = assistant.ai.encode_state(state_ai)
                        act = assistant.ai.predict_action(st, valid)
                        self.demo_log_queue.put(f"{pname} đánh: {assistant._format_action(act)}\n")

                    try:
                        _, _, done_step, info_step = env.step(act)
                        done = done_step
                        info = info_step

                        if info.get("error", "").startswith("Severe"):
                            err = True
                            done = True
                            self.demo_log_queue.put(f" -> LỖI NGHIÊM TRỌNG: {info['error']}\n")
                        elif info.get("error"):
                            self.demo_log_queue.put(f" -> Thông báo lỗi: {info['error']}\n")

                        if info.get("message"):
                            self.demo_log_queue.put(f" -> {info['message']}\n")

                    except Exception as e:
                        self.demo_log_queue.put(f"!!! Lỗi bước: {e}\n")
                        traceback.print_exc()
                        done = True
                        err = True
                        info = {"error": f"Runtime error: {e}"}

                    # Tránh treo máy với demo quá dài
                    if not done and env.turn_count > 100:
                        self.demo_log_queue.put(f"!!! HẾT GIỜ DEMO ở lượt {env.turn_count}\n")
                        err = True
                        done = True
                        info["warning"] = "Demo timeout"

                # Kết thúc game
                self.demo_log_queue.put("\n" + "="*18 + " KẾT THÚC DEMO " + "="*18 + "\n")

                if env.game_over and not err and env.winner is not None:
                    win = f"P{env.winner}"
                    los = f"P{1-env.winner}"
                    self.demo_log_queue.put(f"Người thắng: {win}\n")

                    los_h = env.player_hand if env.winner == 1 else env.ai_hand
                    self.demo_log_queue.put(f"Người thua ({los}) Bài ({len(los_h)}): {assistant._format_action(los_h)}\n")
                    self.demo_log_queue.put("-"*20 + "\n")

                    self.demo_log_queue.put(f"Tiền: {los} trả {env.money_earned} cho {win}\n")
                    self.demo_log_queue.put("Chi tiết điểm:\n")

                    # Đảm bảo các giá trị luôn tồn tại
                    score_keys = {
                        "base_penalty_calc": "Không tính được",
                        "thoi_2_penalty_calc": "Không có",
                        "thoi_4_penalty_calc": "Không có",
                        "xam_result_note": "Không có Xâm",
                        "final_amount_calc": str(env.money_earned)
                    }
                    for k, default in score_keys.items():
                        value = info.get(k, default)
                        title = k.replace('_calc', '').replace('note', '').replace('_', ' ').title()
                        self.demo_log_queue.put(f"  {title}: {value}\n")

                elif err:
                    self.demo_log_queue.put("Trò chơi kết thúc vì lỗi.\n")
                    if info.get("error"):
                        self.demo_log_queue.put(f"Chi tiết lỗi: {info['error']}\n")
                else:
                    self.demo_log_queue.put(f"Trò chơi kết thúc không như dự kiến. Thông tin: {info}\n")

                # Khôi phục epsilon
                assistant.ai.epsilon = eps

                # Xác định và hiển thị kết quả
                result = env.money_earned if env.winner == 1 else (-env.money_earned if env.winner == 0 else 0)
                msg = f"\n=== KẾT QUẢ DEMO ===\nKết quả cho AI: {result}\n"
                self.demo_log_queue.put(msg)

            except Exception as e:
                self.demo_log_queue.put(f"Lỗi khởi tạo demo: {e}\n")
                traceback.print_exc()

            self.root.after(0, self.update_status, "Demo hoàn tất.")

        except Exception as e:
            self.demo_log_queue.put(f"Lỗi demo: {e}\n")
            traceback.print_exc()
        finally:
            # Khôi phục stdout
            sys.stdout = original_stdout

    def process_demo_log_queue(self):
        """Xử lý hàng đợi log demo để đảm bảo thứ tự"""
        try:
            while not self.demo_log_queue.empty():
                text = self.demo_log_queue.get_nowait()
                self.add_demo_text(text)
        except queue.Empty:
            pass
        self.root.after(50, self.process_demo_log_queue)

    def add_demo_text(self, text):
        """Thêm văn bản vào vùng demo"""
        self.demo_text.configure(state='normal')
        self.demo_text.insert(tk.END, text)
        self.demo_text.see(tk.END)
        self.demo_text.configure(state='disabled')

    def get_suggestion(self):
        """Lấy gợi ý từ AI cho tình huống đã cho"""
        try:
            # Phân tích dữ liệu đầu vào
            hand_str = self.hand_var.get()
            player_hand = parse_cards(hand_str)

            if not player_hand:
                messagebox.showerror("Lỗi đầu vào", "Bạn phải nhập ít nhất một lá bài hợp lệ!")
                return

            table_str = self.table_var.get()
            current_play = parse_cards(table_str) if table_str else None

            # Xử lý bài đối thủ
            if self.opp_mode_var.get() == "Random":
                opponent_cards, opp_cards_count = self.random_opponent_cards(player_hand, current_play)
                if opponent_cards is None:
                    messagebox.showerror("Lỗi", "Không đủ bài để random cho đối thủ!")
                    return
            else:
                opp_cards_input = self.opp_cards_var.get()
                if not opp_cards_input.isdigit():
                    messagebox.showerror("Lỗi đầu vào", "Số bài đối thủ phải là số nguyên!")
                    return
                opp_cards_count = int(opp_cards_input)
                if not (0 <= opp_cards_count <= CARDS_PER_PLAYER):
                    messagebox.showerror("Lỗi đầu vào", f"Số bài đối thủ phải từ 0 đến {CARDS_PER_PLAYER}!")
                    return
                opponent_cards = None

            # Phân tích người chơi cuối cùng
            last_player_str = self.last_player_var.get()
            last_player = 0 if last_player_str == "Bạn (0)" else (1 if last_player_str == "Đối thủ (1)" else None)

            # Phân tích số lượt bỏ
            passes = int(self.passes_var.get()) if self.passes_var.get().isdigit() else 0

            # Phân tích trạng thái xâm
            xam_str = self.xam_var.get()
            xam_declared = 0 if xam_str == "Bạn (0)" else (1 if xam_str == "Đối thủ (1)" else None)

            # Xóa thông tin gợi ý cũ
            self.suggestion_text.configure(state='normal')
            self.suggestion_text.delete(1.0, tk.END)
            self.suggestion_text.configure(state='disabled')

            # Tạo một luồng riêng để lấy gợi ý
            threading.Thread(
                target=self._get_suggestion_thread,
                args=(player_hand, current_play, opp_cards_count, xam_declared, last_player, passes, opponent_cards),
                daemon=True
            ).start()

            self.update_status("Đang tính toán gợi ý...")

        except ValueError:
            messagebox.showerror("Lỗi đầu vào", "Vui lòng nhập các giá trị số hợp lệ.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi lấy gợi ý: {e}")
            traceback.print_exc()

    def _get_suggestion_thread(self, player_hand, current_play, opp_cards_count, xam_declared, last_player, passes, opponent_cards=None):
        """Lấy gợi ý từ AI trong một luồng riêng"""
        try:
            # Ghi log thông tin đầu vào
            self.root.after(0, self.add_suggestion_text, "Đang tìm gợi ý với thông tin sau:\n")
            self.root.after(0, self.add_suggestion_text, f"- Bài của bạn: {', '.join([f'{r}{s}' for r,s in player_hand])}\n")
            if current_play:
                self.root.after(0, self.add_suggestion_text, f"- Bài trên bàn: {', '.join([f'{r}{s}' for r,s in current_play])}\n")
            else:
                self.root.after(0, self.add_suggestion_text, "- Bài trên bàn: (không có)\n")
            if opponent_cards:
                self.root.after(0, self.add_suggestion_text, f"- Bài đối thủ (Random): {', '.join([f'{r}{s}' for r,s in opponent_cards])} ({opp_cards_count} lá)\n")
            else:
                self.root.after(0, self.add_suggestion_text, f"- Số lá đối thủ: {opp_cards_count}\n")
            self.root.after(0, self.add_suggestion_text, f"- Xâm đã báo: {xam_declared if xam_declared is not None else 'Không có'}\n")
            self.root.after(0, self.add_suggestion_text, f"- Người chơi gần nhất: {last_player if last_player is not None else 'Không có'}\n")
            self.root.after(0, self.add_suggestion_text, f"- Số lượt bỏ liên tiếp: {passes}\n\n")
            self.root.after(0, self.add_suggestion_text, "Đang tính toán, vui lòng đợi...\n\n")

            # Lấy gợi ý
            assistant = XamLocSoloAssistant(model_path=self.model_file, betting_unit=1)
            suggestion = assistant.suggest_move(
                player_hand=player_hand,
                current_play=current_play,
                opponent_cards=opp_cards_count,
                xam_declared=xam_declared,
                last_player=last_player,
                consecutive_passes=passes
            )

            # Hiển thị gợi ý
            self.root.after(0, self.add_suggestion_text, f"KẾT QUẢ GỢI Ý:\n{suggestion}\n")
            self.root.after(0, self.update_status, "Đã tính toán gợi ý.")

        except Exception as e:
            error_msg = f"Lỗi khi tìm gợi ý: {str(e)}\n"
            self.root.after(0, self.add_suggestion_text, error_msg)
            self.root.after(0, self.update_status, "Lỗi khi tính toán gợi ý")
            traceback.print_exc()

    def add_suggestion_text(self, text):
        """Thêm văn bản vào vùng gợi ý"""
        self.suggestion_text.configure(state='normal')
        self.suggestion_text.insert(tk.END, text)
        self.suggestion_text.configure(state='disabled')

    def analyze_data(self):
        """Phân tích dữ liệu huấn luyện"""
        try:
            self.update_status("Đang phân tích dữ liệu...")

            # Tạo một AI mới và tải mô hình
            tmp_ai = XamLocSoloAI()

            if tmp_ai.load(self.model_file):
                # Lấy thông tin phân tích
                analysis = tmp_ai.analyze_learning()

                # Xóa thông tin phân tích cũ
                self.analyze_text.configure(state='normal')
                self.analyze_text.delete(1.0, tk.END)

                # Hiển thị thông tin phân tích
                self.analyze_text.insert(tk.END, "=== PHÂN TÍCH DỮ LIỆU HUẤN LUYỆN ===\n\n")
                self.analyze_text.insert(tk.END, f"Đường dẫn mô hình: {self.model_file}\n\n")

                for item in analysis:
                    self.analyze_text.insert(tk.END, f"{item}\n")

                # Thêm thông tin tổng quan
                self.analyze_text.insert(tk.END, "\n=== THÔNG TIN CHI TIẾT ===\n")

                # Thông tin về Epsilon
                self.analyze_text.insert(tk.END, f"Epsilon hiện tại: {tmp_ai.epsilon:.6f}\n")
                self.analyze_text.insert(tk.END, f"Epsilon min: {tmp_ai.epsilon_min:.6f}\n")
                self.analyze_text.insert(tk.END, f"Epsilon decay: {tmp_ai.epsilon_decay:.6f}\n\n")

                # Thông tin về bộ nhớ
                self.analyze_text.insert(tk.END, f"Kích thước bộ nhớ: {len(tmp_ai.memory)}/{tmp_ai.memory.maxlen}\n")
                self.analyze_text.insert(tk.END, f"Kích thước batch: {tmp_ai.batch_size}\n\n")

                # Thông tin về model
                self.analyze_text.insert(tk.END, f"Model device: {tmp_ai.device}\n")
                self.analyze_text.insert(tk.END, f"Tổng số tham số: {sum(p.numel() for p in tmp_ai.model.parameters())}\n")

                self.analyze_text.configure(state='disabled')
                self.update_status("Đã phân tích dữ liệu. Nhấn 'Xem biểu đồ' để hiển thị biểu đồ tiền thưởng.")

                # Tự động vẽ biểu đồ luôn
                self.show_chart()
            else:
                messagebox.showerror("Lỗi", f"Không thể tải {self.model_file} để phân tích.")
                self.update_status("Lỗi khi tải dữ liệu.")

        except Exception as e:
            messagebox.showerror("Lỗi phân tích", f"Lỗi: {e}")
            self.update_status(f"Lỗi phân tích: {str(e)}")
            traceback.print_exc()

    def show_chart(self):
        """Hiển thị biểu đồ tiền thưởng"""
        try:
            self.update_status("Đang vẽ biểu đồ...")

            # Tạo một AI mới và tải mô hình
            tmp_ai = XamLocSoloAI()

            if tmp_ai.load(self.model_file):
                # Lấy dữ liệu lịch sử
                hist = tmp_ai.experience_log.get("money_history", [])
                turn_hist = tmp_ai.experience_log.get("turns_history", [])
                pass_hist = tmp_ai.experience_log.get("pass_history", [])

                if len(hist) < 2:
                    messagebox.showinfo("Thông báo", "Không đủ dữ liệu để vẽ biểu đồ.")
                    self.update_status("Không đủ dữ liệu.")
                    return

                # Xóa biểu đồ cũ
                self.ax.clear()

                # Vẽ biểu đồ chính - Tiền thưởng
                gl = len(hist)
                x = list(range(1, gl + 1))
                self.ax.plot(x, hist, label='Tiền/Trò chơi', alpha=0.6, lw=1, color='dodgerblue')

                # Vẽ đường trung bình động
                window = min(50, gl)
                if window > 1:
                    mv = np.convolve(hist, np.ones(window) / window, mode='valid')
                    xm = list(range(window, gl + 1))
                    self.ax.plot(xm, mv, label=f'Trung bình ({window} trò chơi)', color='red', lw=2)

                # Vẽ đường break-even
                self.ax.axhline(y=0, color='grey', ls='--', alpha=0.7, label='Hòa vốn')

                # Cấu hình biểu đồ
                self.ax.set_title(f'Huấn luyện AI ({gl} trò chơi đã ghi nhận)')
                self.ax.set_xlabel('Trò chơi (Đã ghi nhận)')
                self.ax.set_ylabel('Tiền thắng/thua (Góc nhìn AI)')
                self.ax.legend()
                self.ax.grid(True, ls=':', alpha=0.6)

                # Cập nhật canvas
                self.fig.tight_layout()
                self.canvas.draw()

                # Thêm thông tin thống kê
                if self.analyze_text:
                    self.analyze_text.configure(state='normal')
                    self.analyze_text.insert(tk.END, "\n\n=== THỐNG KÊ BIỂU ĐỒ ===\n")
                    self.analyze_text.insert(tk.END, f"Tổng số trò chơi: {gl}\n")

                    if hist:
                        avg_money = sum(hist)/len(hist)
                        avg_money_50 = sum(hist[-50:])/min(50, len(hist)) if len(hist) >= 50 else avg_money
                        self.analyze_text.insert(tk.END, f"Tiền trung bình/trò chơi: {avg_money:.2f}\n")
                        if len(hist) >= 50:
                            self.analyze_text.insert(tk.END, f"Tiền trung bình 50 trò chơi gần nhất: {avg_money_50:.2f}\n")

                    if turn_hist:
                        avg_turns = sum(turn_hist)/len(turn_hist)
                        self.analyze_text.insert(tk.END, f"Lượt trung bình/trò chơi: {avg_turns:.1f}\n")

                    if pass_hist:
                        avg_passes = sum(pass_hist)/len(pass_hist)
                        self.analyze_text.insert(tk.END, f"Bỏ lượt trung bình/trò chơi: {avg_passes:.1f}\n")

                    self.analyze_text.configure(state='disabled')

                self.update_status("Đã hiển thị biểu đồ.")
            else:
                messagebox.showerror("Lỗi", f"Không thể tải {self.model_file} để vẽ biểu đồ.")
                self.update_status("Lỗi khi tải dữ liệu.")

        except Exception as e:
            messagebox.showerror("Lỗi biểu đồ", f"Lỗi: {e}")
            self.update_status(f"Lỗi biểu đồ: {str(e)}")
            traceback.print_exc()

def main():
    """Hàm chính để chạy ứng dụng"""
    root = tk.Tk()
    app = XamLocSoloGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()