# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATE_SIZE = 47  # Cập nhật: 13 (hand encoding) + 13 (hand ranks) + 13 (current play) + 6 (game features) + 1 (has_two) + 1 (hand_diff)
ACTION_SIZE = 15  # 13 ranks + pass + declare_xam

class XamLocSoloModel(nn.Module):
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE):
        super(XamLocSoloModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_value = nn.Linear(128, 1)
        self.fc_policy = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc_value(x)
        policy = self.fc_policy(x)
        return value, policy