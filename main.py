# -*- coding: utf-8 -*-
from src.assistant.demo import play_demo_game
from src.assistant.ai_training import XamLocSoloAssistant, train_xam_loc_AI

if __name__ == "__main__":
    # Chạy demo game
    assistant = XamLocSoloAssistant(betting_unit=1)
    demo_log = []
    play_demo_game(assistant, betting_unit=1, model_path="xam_loc_solo_model.pth", demo_log=demo_log)
    for line in demo_log:
        print(line)
    # Ví dụ: Huấn luyện AI
    # ai, log, summary = train_xam_loc_AI(num_episodes=100)
    # for line in summary:
    #     print(line)