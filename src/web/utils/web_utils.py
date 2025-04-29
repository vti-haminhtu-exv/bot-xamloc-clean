# -*- coding: utf-8 -*-
from ..game.game_rules import SUITS, RANKS

def parse_cards(cards_str):
    """Chuyển đổi chuỗi bài thành danh sách các lá bài."""
    if not cards_str:
        return []
    cards = []
    for card in cards_str.split(','):
        card = card.strip()
        if len(card) < 2:
            continue
        rank = card[:-1]
        suit = card[-1]
        if rank in RANKS and suit in SUITS:
            cards.append((rank, suit))
    return cards