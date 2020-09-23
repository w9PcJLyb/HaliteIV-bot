from .board import MyBoard
from .logger import logger


def shipyards_actions(board: MyBoard):
    """ Create more ships """
    if not board.my_shipyards:
        return

    if board.my_halite < board.configuration.spawn_cost:
        # not enough halite
        return

    ships_needed = _compute_max_ships(board)
    if board.num_my_ships >= ships_needed:
        return

    free_shipyards = board.free_shipyards
    if not free_shipyards:
        logger.warning("Need to create more ships, but we don't have free shipyards.")
        return

    free_shipyards = sorted(
        free_shipyards, key=lambda x: -board.environment_reserves(x.position)
    )

    for sy in free_shipyards:
        if (
            board.num_my_ships < ships_needed
            and board.my_halite >= board.configuration.spawn_cost
        ):
            board.create_ship(sy)


def _compute_max_ships(board: MyBoard):
    # This needs to be tuned, perhaps based on amount of halite left
    my_shipyards = board.my_shipyards
    if not my_shipyards:
        # oh, it's bad
        return 0

    def shipyard_distance(p):
        return min(board.distance(p, sy.position) for sy in my_shipyards)

    num_cells_with_halite = sum(
        1
        for p, h in board.position_to_halite.items()
        if h > 50 and shipyard_distance(p) <= 5
    )

    return min(num_cells_with_halite, _max_ships_from_step(board.step))


def _max_ships_from_step(step):
    if step < 180:
        return 35
    elif step < 300:
        return 10
    elif step < 350:
        return 8
    elif step < 370:
        return 5
    elif step < 380:
        return 4
    elif step < 390:
        return 2
    else:
        return 1
