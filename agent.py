from halite import *


def agent(obs, config):
    b = MyBoard(obs, config)
    opening_actions(b)
    defend_shipyards(b)
    endspiel_actions(b)
    ship_converts(b)
    ship_moves(b)
    shipyards_actions(b)
    leave_one_ship_in_a_shipyard(b)
    return b.me.next_actions
