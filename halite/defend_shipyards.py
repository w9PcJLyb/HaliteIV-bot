import numpy as np
from .board import MyBoard
from .logger import logger


def defend_shipyards(board: MyBoard):
    for sy in board.my_shipyards:
        attacker_cargo = board.attacker_cargo(sy.position, defender=board.me)
        if attacker_cargo is None:
            continue

        # looking help nearby
        my_id = board.me.id
        taken_ships = board.taken_ships
        ship, cargo = None, -np.inf
        for _p, _ in board.get_neighbor_positions(sy.position, add_self=False):
            _ship = board.position_to_ship.get(_p)
            if not _ship or _ship.player_id != my_id or _ship in taken_ships:
                continue

            _cargo = _ship.halite

            if _ship.halite > attacker_cargo:
                continue

            if _ship.halite > cargo:
                ship, cargo = _ship, _cargo

        if ship:
            logger.info(
                f"Ship {ship.id} at {ship.position}: Need to enter the shipyard to protect it."
            )
            action, distance = board.dirs_to(ship.position, sy.position)
            assert len(action) == 1 and action is not None and distance == 1
            board.move_ship(ship, action[0])
            continue

        # may be there is a ship in the shipyard
        ship = board.position_to_ship.get(sy.position)
        if ship and ship not in board.taken_ships:
            logger.info(
                f"Ship {ship.id} at {ship.position}: Stay at the shipyard to protect it."
            )
            board.move_ship(ship, None)
            continue

        # nothing works, turn on the active defense
        if (60 > board.moves_left >= 30 and board.num_my_shipyards > 1) or (
            30 > board.moves_left and board.num_my_ships > 0
        ):
            # Some players send all their ships to the enemy shipyards at the end of the game,
            # trying to eliminate the opponents.
            # We don't want to spend a lot of halite at the end of the game to create the active defense.
            continue

        if board.my_halite >= board.configuration.spawn_cost:
            logger.info(
                f"Shipyard {sy.id} at {sy.position}: Need to create a ship to protect yourself."
            )
            board.create_ship(sy)
        else:
            logger.warning(
                f"Shipyard {sy.id} at {sy.position}: "
                "Need to create a ship to protect yourself. But we have not enough halite."
            )


def leave_one_ship_in_a_shipyard(board: MyBoard):
    if board.step < 50:
        # not need to defend shipyards at the beginning of the game
        return

    for sy in board.my_shipyards:
        ship = board.position_to_ship.get(sy.position)
        if not ship:
            continue

        if sy.position not in board.taken_cells:
            board.move_ship(ship, None)
