import numpy as np
from typing import List, Optional, Tuple
from collections import defaultdict
from kaggle_environments.envs.halite.helpers import Ship
from .board import MyBoard, ALL_SHIP_ACTIONS
from .logger import logger


def ship_converts(board: MyBoard):
    """ Convert our ships into shipyards """
    if board.step == 0 or board.moves_left < 20:
        return

    if not board.num_my_shipyards:
        is_final_part = board.moves_left <= 40
        _create_shipyard(board, is_final_part)

    for ship in board.free_ships:
        # CHECK if in danger without escape, convert if h > 500
        if ship.halite <= board.configuration.convert_cost:
            continue

        avoid_moves = board.avoid_moves(ship)
        if ALL_SHIP_ACTIONS - avoid_moves:
            continue

        logger.warning(
            f"Ship {ship.id} at {ship.position}: Can't run away, converting."
        )
        board.create_shipyard(ship)

    # Generate a shipyard from the best ship
    min_score = 1000
    if board.num_my_shipyards < 2:
        min_score = 400
    if (
        board.num_my_shipyards <= 3
        and board.num_my_ships > 10 + board.num_my_shipyards * 5
        and board.moves_left > 100
    ):
        available_ships = [x for x in board.free_ships if _can_convert_ship(board, x)]
        if available_ships:
            ship, score = _choice_ship_to_convert(board, available_ships)
            if ship is not None and score > min_score:
                logger.info(
                    f"Ship {ship.id} at {ship.position}: Create a shipyard, cell score = {score}."
                )
                board.create_shipyard(ship)


def _can_convert_ship(board: MyBoard, ship: Ship) -> bool:
    """ Is this the good place for a shipyard? """
    pos = ship.position
    if pos in board.position_to_shipyard:
        return False

    if (
        ship.halite + board.my_halite < board.configuration.convert_cost
        or board.is_danger_position(pos, ship)
    ):
        return False

    num_my_shipyards = sum(
        1 for x in board.my_shipyards if board.distance(x.position, pos) <= 2
    )
    if num_my_shipyards > 0:
        return False

    num_my_ships = sum(
        1 for x in board.my_ships if board.distance(x.position, pos) <= 1
    )
    if num_my_ships < 1:
        return False

    min_distance_to_enemy_ship = min(
        board.distance(x.position, pos)
        for x in board.ships.values()
        if x.player_id != board.me.id
    )
    if min_distance_to_enemy_ship <= 2:
        return False

    return True


def _create_shipyard(board: MyBoard, is_final_part: bool = False):
    """ What we do if we haven't shipyards """
    if is_final_part:
        # the end of the game, convert one ship if it makes sense
        ship_to_halite = defaultdict(int)

        available_ships = [
            x
            for x in board.my_ships
            if x.halite + board.my_halite >= board.configuration.convert_cost
        ]
        for ship in available_ships:
            distance_to_enemy_ship = board.distance_to_enemy_ship(ship.position, board.me)
            distance_to_enemy_ship = distance_to_enemy_ship or board.size
            if distance_to_enemy_ship < 3:
                # an enemy vessel nearby, can't convert
                continue

            max_my_ship_distance = min(distance_to_enemy_ship, board.moves_left)
            for other_ship in board.my_ships:
                if board.distance(ship.position, other_ship.position) < max_my_ship_distance:
                    ship_to_halite[ship] += other_ship.halite

        if not ship_to_halite:
            return

        max_halite = max(ship_to_halite.values())
        if max_halite > board.configuration.convert_cost:
            # it makes sense to convert, choose one
            ship = [s for s, h in ship_to_halite.items() if h == max_halite][0]
            board.create_shipyard(ship)

    else:
        # meddle of the game, we have to create a shipyard
        logger.warning("No shipyards! We must create at least one!")

        available_ships = [
            x
            for x in board.my_ships
            if x.halite + board.my_halite >= board.configuration.convert_cost
        ]
        if not available_ships:
            logger.warning("Can't create a shipyard, not enough halite! Keep mining.")
            return

        if (
            len(available_ships) == 1
            and board.my_halite + available_ships[0].halite
            < board.configuration.convert_cost + board.configuration.spawn_cost
        ):
            logger.warning("Can't create a shipyard, not enough halite! Keep mining.")
            return

        ship, _ = _choice_ship_to_convert(board, available_ships)
        if ship:
            board.create_shipyard(ship)


def _choice_ship_to_convert(
    board: MyBoard, ships: List[Ship]
) -> Tuple[Optional[Ship], float]:
    assert len(ships) > 0

    ship, score = None, -np.inf
    for _ship in ships:
        pos = _ship.position
        if pos in board.position_to_shipyard:
            # shipyard here
            continue

        _score = board.environment_reserves(pos)

        _score -= board.position_to_halite[pos]

        if _score > score:
            ship, score = _ship, _score

    return ship, score
