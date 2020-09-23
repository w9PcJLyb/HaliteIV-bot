import random
from typing import List, Union
from .board import MyBoard, ALL_SHIP_ACTIONS
from kaggle_environments.envs.halite.helpers import Point, Ship, Player, Shipyard


def endspiel_actions(board: MyBoard):
    if board.moves_left <= 14:
        for ship in board.free_ships:
            if ship.halite > 0:
                _park_ship(board, ship)

        for ship in board.free_ships:
            position = ship.position
            shipyard = board.position_to_shipyard.get(position)
            if shipyard and position not in board.taken_cells:
                # leave this ship for protection
                board.move_ship(ship, None)

        if board.num_my_ships > 2:
            for ship in board.free_ships:
                _endspiel_attack_actions(board, ship)

    if board.moves_left == 1:
        _last_step_actions(board)


def _park_ship(board: MyBoard, ship: Ship):
    """ Send our ship to the nearest shipyard """
    moves_left = board.moves_left
    shipyard_to_distance = _get_shipyard_to_distance(board, ship.position, board.me)
    shipyards = [s for s, d in shipyard_to_distance.items() if d <= moves_left]
    if not shipyards:
        # create new one, if there are enough halite
        if (
            ship.halite > board.configuration.convert_cost
            and not board.is_danger_position(ship.position, ship)
        ):
            board.create_shipyard(ship)
        return

    shipyards = sorted(shipyards, key=lambda x: shipyard_to_distance[x])

    for sy in shipyards:
        if _sent_ship_to_shipyard(board, ship, sy):
            return

    for sy in shipyards:
        if shipyard_to_distance[sy] < moves_left - 1:
            if _sent_ship_to_shipyard(board, ship, sy, can_wait=True):
                return

    for sy in shipyards:
        if _sent_ship_to_shipyard(board, ship, sy, allow_collision=True):
            return


def _sent_ship_to_shipyard(
    board: MyBoard,
    ship: Ship,
    shipyard: Shipyard,
    can_wait: bool = False,
    allow_collision: bool = False,
) -> bool:
    actions, distance = board.dirs_to(ship.position, shipyard.position)
    if distance == 0:
        return True

    target_moves = set(actions)
    stalemate_and_bad_moves = board.avoid_moves(ship, avoid_stalemate=True)
    taken_moves = board.taken_moves(ship)

    action_set = target_moves - stalemate_and_bad_moves - taken_moves

    if action_set:
        board.move_ship(ship, random.choice(list(action_set)))
        return True

    if can_wait:
        if None not in stalemate_and_bad_moves and None not in taken_moves:
            board.move_ship(ship, None)
            return True

        other_actions = (
            ALL_SHIP_ACTIONS - {None} - stalemate_and_bad_moves - taken_moves
        )
        if other_actions:
            board.move_ship(ship, random.choice(list(other_actions)))
            return True

    if allow_collision and target_moves - stalemate_and_bad_moves:
        board.move_ship(
            ship, random.choice(list(target_moves - stalemate_and_bad_moves))
        )
        return True

    return False


def _endspiel_attack_actions(board: MyBoard, ship: Ship):
    """ Attack enemy shipyards """
    moves_left = board.moves_left
    shipyard_to_distance = _get_shipyard_to_distance(board, ship.position, board.opponents)
    shipyards = [s for s, d in shipyard_to_distance.items() if d <= moves_left]
    if not shipyards:
        return

    for sy in sorted(shipyards, key=lambda x: shipyard_to_distance[x]):
        actions, _ = board.dirs_to(ship.position, sy.position)

        target_moves = set(actions)
        taken_moves = board.taken_moves(ship)

        action_set = target_moves - taken_moves
        if action_set:
            board.move_ship(ship, random.choice(list(action_set)))
            return


def _last_step_actions(board: MyBoard):
    for ship in board.my_ships:
        if ship.halite == 0:
            continue

        shipyard, distance = board.nearest_shipyard(ship.position, player=board.me)

        if distance == 1:
            actions, _ = board.dirs_to(ship.position, shipyard.position)
            board.move_ship(ship, actions[0])
        elif ship.halite > board.configuration.convert_cost:
            board.create_shipyard(ship)


def _get_shipyard_to_distance(
    board: MyBoard, position: Point, player: Union[Player, List[Player]]
):
    if not isinstance(player, list):
        player = [player]
    player_ids = {x.id for x in player}
    shipyard_to_distance = {}
    for p, sy in board.position_to_shipyard.items():
        if sy.player_id in player_ids:
            shipyard_to_distance[sy] = board.distance(p, position)
    return shipyard_to_distance
