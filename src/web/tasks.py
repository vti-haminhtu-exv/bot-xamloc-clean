# -*- coding: utf-8 -*-
import logging
from src.ai.ai_training import XamLocSoloAssistant
from src.game.game_environment import XamLocSoloEnvironment, XamLocGameState

# Kiểm tra xem thư viện mctspy có sẵn không
try:
    from mctspy.tree.search import MonteCarloTreeSearch as MCTS
    from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    logger = logging.getLogger('xam_loc_solo')
    logger.warning("Thư viện 'mctspy' không có sẵn. Sử dụng DQN để huấn luyện thay thế.")

logger = logging.getLogger('xam_loc_solo')

def train_task(games, betting_unit):
    try:
        assistant = XamLocSoloAssistant(betting_unit=betting_unit)
        env = XamLocSoloEnvironment(betting_unit=betting_unit)  # Sử dụng môi trường thay vì AI
        ai = assistant.ai  # AI để xử lý DQN

        for episode in range(games):
            state = env.reset()
            game_state = XamLocGameState(env)
            done = False
            while not done:
                if MCTS_AVAILABLE:
                    # Tạo node gốc từ trạng thái hiện tại
                    root_node = TwoPlayersGameMonteCarloTreeSearchNode(
                        state=game_state,
                        parent=None
                    )
                    # Khởi tạo MCTS với node gốc
                    mcts = MCTS(root_node)
                    # Sử dụng MCTS để chọn hành động tối ưu
                    action = mcts.best_action(simulation_time=1.0, max_iterations=1000)
                    if action is None:
                        action = "pass"  # Nếu MCTS không tìm được hành động, bỏ lượt
                else:
                    # Sử dụng DQN nếu MCTS không có sẵn
                    valid_actions = ai.get_valid_actions(state, env)
                    action = ai.predict_action(state, valid_actions, env)
                next_state, reward, done, info = env.step(action)
                # Lưu trữ trải nghiệm vào bộ nhớ DQN
                valid_actions = ai.get_valid_actions(state, env)
                action_idx = ai.action_to_index(action, valid_actions)
                ai.remember(state, action_idx, reward, next_state, done)
                state = next_state
                game_state = XamLocGameState(env)
                # Huấn luyện DQN dựa trên dữ liệu
                ai.replay()
            ai.update_target_model()
            if episode % 100 == 0:
                logger.info(f"Đã huấn luyện {episode}/{games} ván")
        ai.save("xam_loc_solo_model.pth")
        return f"Huấn luyện hoàn tất: {games} ván."
    except Exception as e:
        logger.error(f"Lỗi trong train_task: {str(e)}")
        return None