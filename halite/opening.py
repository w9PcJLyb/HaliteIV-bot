from kaggle_environments.envs.halite.helpers import Ship
from . import MyBoard


def opening_actions(board: MyBoard):
    if board.step > 0:
        return

    if board.my_shipyards or len(board.my_ships) != 1:
        return

    ship = board.my_ships[0]
    if ship.halite + board.my_halite < 500:
        return

    _convert_ship_into_shipyard(board, ship, may_choose=False)


def _convert_ship_into_shipyard(board: MyBoard, ship: Ship, may_choose: bool = False):
    position = ship.position
    cell_halite = board.position_to_halite[position]
    if not may_choose or cell_halite == 0:
        board.create_shipyard(ship)
        return

    # find the best position
    action, halite = None, 0
    for _p, _action in board.get_neighbor_positions(position, add_self=False):
        cell_halite = board.position_to_halite[position]
        if cell_halite > 0:
            continue

        _halite = board.environment_reserves(_p)
        if _halite > halite:
            action, halite = _action, _halite

    if action is None:
        board.create_shipyard(ship)
        return

    board.move_ship(ship, action)
