import math
import random
import numpy as np
from typing import Dict, List, Optional, Set
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from kaggle_environments.envs.halite.helpers import Point, Ship, Shipyard, ShipAction
from .board import MyBoard, ALL_SHIP_ACTIONS, reverse_action
from .logger import logger


# turns_optimal[CH ratio, round_trip_travel] for mining
# See notebook on optimal mining https://www.kaggle.com/solverworld/optimal-mining
TURNS_OPTIMAL = np.array(
    [
        [0, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8],
        [0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7],
        [0, 0, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7],
        [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6],
        [0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6],
        [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


def ship_moves(board: MyBoard):
    """ Ship movement strategy """
    _attack_shipyards(board)
    _attack_ships(board)
    _move_ships_without_escape(board)
    # _move_ships_from_shipyards(board)
    _mine_halite(board)


def _mine_halite(board: MyBoard):
    ships = [x for x in board.my_ships if x.next_action != ShipAction.CONVERT]
    ship_to_mining_actions = _find_targets(board, ships)

    for ship in sorted(board.free_ships, key=lambda x: -x.halite):

        # to prevent self collision
        for other_ship in board.free_ships:
            bad_moves = board.avoid_moves(other_ship, avoid_stalemate=False)
            possible_moves = (
                ALL_SHIP_ACTIONS - bad_moves - board.taken_moves(other_ship)
            )
            if len(possible_moves) == 1:
                action, *_ = possible_moves
                board.move_ship(other_ship, action)

        if ship in board.taken_ships:
            continue

        mining_actions = ship_to_mining_actions.get(ship, set())
        attack_actions = board.attack_moves(
            ship, add_moves_when_an_enemy_can_escape=False
        )
        hunters_stalemate_and_bad_moves = board.avoid_moves(
            ship, avoid_stalemate=True, avoid_hunters=True
        )
        stalemate_and_bad_moves = board.avoid_moves(
            ship, avoid_stalemate=True, avoid_hunters=False
        )
        bad_moves = board.avoid_moves(ship, avoid_stalemate=False, avoid_hunters=False)
        taken_moves = board.taken_moves(ship)
        hunters_stalemate_bad_and_taken_moves = (
            hunters_stalemate_and_bad_moves | taken_moves
        )
        stalemate_bad_and_taken_moves = stalemate_and_bad_moves | taken_moves
        bad_and_taken_moves = bad_moves | taken_moves
        logger.debug(
            f"Ship {ship.id} at {ship.position}: "
            f"mining_actions={_to_str(mining_actions)}, "
            f"attack_actions={_to_str(attack_actions)}, "
            f"hunters_stalemate_bad_and_taken_moves={_to_str(hunters_stalemate_bad_and_taken_moves)}, "
            f"stalemate_and_bad_moves={_to_str(stalemate_and_bad_moves)}, "
            f"bad_moves={_to_str(bad_moves)}, "
            f"taken_moves={_to_str(taken_moves)}."
        )

        action_set = (
            mining_actions & attack_actions - hunters_stalemate_bad_and_taken_moves
            or mining_actions - hunters_stalemate_bad_and_taken_moves
            or attack_actions - hunters_stalemate_bad_and_taken_moves
            or ALL_SHIP_ACTIONS - hunters_stalemate_bad_and_taken_moves
            or mining_actions & attack_actions - stalemate_bad_and_taken_moves
            or mining_actions - stalemate_bad_and_taken_moves
            or attack_actions - stalemate_bad_and_taken_moves
            or ALL_SHIP_ACTIONS - stalemate_bad_and_taken_moves
            or mining_actions & attack_actions - bad_and_taken_moves
            or mining_actions - bad_and_taken_moves
            or attack_actions - bad_and_taken_moves
            or ALL_SHIP_ACTIONS - bad_and_taken_moves
            or mining_actions & attack_actions - taken_moves
            or mining_actions - taken_moves
            or attack_actions - taken_moves
            or ALL_SHIP_ACTIONS - taken_moves
        )

        if not action_set:
            logger.error(
                f"Ship {ship.id} at {ship.position}: Can't do anything! All positions are already taken!"
            )
            if (
                ship.halite + board.me.halite > board.configuration.convert_cost
                and board.moves_left > 100
            ):
                logger.info(
                    f"Ship {ship.id} at {ship.position}: Converting to avoid destruction."
                )
                board.create_shipyard(ship)
            continue

        if (
            {None} in action_set
            and len(action_set) > 1
            and board.position_to_halite[ship.position] > 0
        ):
            action_set -= {None}

        board.move_ship(ship, random.choice(list(action_set)))


def _to_str(a):
    return "{" + ", ".join(map(str, a)) + "}"


def _move_ships_from_shipyards(board: MyBoard):
    for shipyard in board.my_shipyards:
        pos = shipyard.position
        ship = board.position_to_ship.get(pos)
        if not ship:
            continue

        if ship in board.taken_ships:
            continue

        action, cargo, other_ship = None, 0, None
        for _pos, _action in board.get_neighbor_positions(pos, add_self=False):
            _other_ship = board.position_to_ship.get(_pos)
            if (
                not _other_ship
                or _other_ship.player_id != board.me.id
                or _other_ship in board.taken_ships
            ):
                continue

            _cargo = _other_ship.halite
            if _cargo > cargo:
                action, cargo, other_ship = _action, _cargo, _other_ship

        if action and other_ship:
            board.move_ship(ship, action)
            board.move_ship(other_ship, reverse_action(action))


class TargetCell:
    def __init__(
        self,
        board: MyBoard,
        position: Point,
        is_prey: bool = False,
        is_my_shipyard: bool = False,
    ):
        assert not (is_prey and is_my_shipyard)
        self._board = board
        self.position = position
        self.is_prey = is_prey
        self.is_my_shipyard = is_my_shipyard
        self.halite = board.position_to_halite[position]
        self.enemy_halite = self._enemy_halite()
        self.distance_to_my_shipyard = self._distance_to_shipyard(board.me)
        self.distance_to_enemy_shipyard = self._distance_to_shipyard(board.opponents)
        self.num_ships_around = self._num_ships_around()
        self.enemy_ships_around = self._enemy_ships_around()

    def num_enemy_hunters_around(self, cargo=0):
        return sum(1 for s in self.enemy_ships_around if s.halite <= cargo)

    def _distance_to_shipyard(self, player):
        sy, d = self._board.nearest_shipyard(self.position, player)
        return d if sy else None

    def _enemy_ships_around(self):
        return [
            ship
            for pos, ship in self._board.position_to_ship.items()
            if ship.player_id != self._board.me.id
            and self._board.distance(pos, self.position) <= 2
        ]

    def _num_ships_around(self):
        return sum(
            1
            for pos, ship in self._board.position_to_ship.items()
            if self._board.distance(pos, self.position) <= 2
        )

    def _enemy_halite(self):
        ship = self._board.position_to_ship.get(self.position)
        if not ship or ship.player_id == self._board.me.id:
            return
        return ship.halite


def _find_targets(
    board: MyBoard, ships: List[Ship]
) -> Dict[Ship, Set[Optional[ShipAction]]]:
    # https://www.kaggle.com/solverworld/optimus-mine-agent
    if not ships:
        return {}

    all_targets = []
    for pos in _mining_cells(board):
        target = TargetCell(board, pos)
        if target.num_enemy_hunters_around() < 4:
            all_targets.append(TargetCell(board, pos))

    for sy in board.my_shipyards:
        for i in range(4):
            all_targets.append(TargetCell(board, sy.position, is_my_shipyard=True))

    for ship in board.ships.values():
        if ship.halite > 0 and ship.player_id != board.me.id:
            num_my_hunters_around = board.num_hunters_around(
                ship.position, board.me, max_distance=3
            )
            if num_my_hunters_around < 3:
                continue
            for pos, _ in board.get_neighbor_positions(ship.position, add_self=True):
                target = TargetCell(board, pos, is_prey=True)
                target.enemy_halite = ship.halite
                all_targets.append(target)

    mine_matrix = np.zeros((len(ships), len(all_targets)))

    for i, ship in enumerate(ships):
        available_actions = (
            ALL_SHIP_ACTIONS
            - board.avoid_moves(ship, avoid_stalemate=False, avoid_hunters=True)
            - board.taken_moves(ship)
        )
        my_halite = ship.halite
        for j, target in enumerate(all_targets):
            if not target.is_my_shipyard and not target.is_prey:
                actions, d1 = board.dirs_to(ship.position, target.position)
                d2 = target.distance_to_my_shipyard or 1
                d3 = target.distance_to_enemy_shipyard

                # in the mining section
                mining_score, mined = _halite_per_turn(my_halite, target.halite, d1 + d2)

                # do not send ships where there are already many ships
                num_ships_around = target.num_ships_around
                if d1 <= 2:
                    num_ships_around -= 1
                mining_score -= 0.1 * num_ships_around * abs(mining_score)

                if board.step < 40:
                    # more aggressive mining near enemy shipyards at the beginning of the game
                    if d3 and d2 >= d3:
                        mining_score *= 1.1
                        if target.position == ship.position:
                            mining_score *= 100
                elif board.moves_left > 50:
                    # careful mining at the middle
                    pass
                else:
                    # at the end, mining only near our shipyards
                    if d1 + d2 > board.moves_left + 1:
                        mining_score -= 1000

            elif target.is_my_shipyard:
                # in the direct to shipyard section
                actions, d1 = board.dirs_to(ship.position, target.position)
                if d1 > 0:
                    mining_score = ship.halite / d1
                else:
                    # we are at a shipyard
                    mining_score = 0

            elif target.is_prey:
                actions, d1 = board.dirs_to(ship.position, target.position)
                if ship.halite == 0 and 0 < d1 <= 3:
                    dx, dy = ship.position - target.position
                    mining_score = (
                        1000 + target.enemy_halite + 500
                        if (dx == 0 or dy == 0)
                        else -500
                    )
                else:
                    mining_score = -10000

            else:
                raise ValueError("Unknown target.")

            if not set(actions) & available_actions:
                # path blocked
                mining_score -= 1000

            mine_matrix[i, j] = mining_score

    # Compute the optimal assignment
    row, col = linear_sum_assignment(mine_matrix, maximize=True)
    # so ship row[i] is assigned to target col[j]
    ship_to_mining_actions = {}
    ship_id_to_target = {}
    for r, c in zip(row, col):
        ship = ships[r]
        target = all_targets[c]
        if target.is_prey:
            logger.info(
                f"Ship {ship.id} at {ship.position}: I'm a hunter, my target - {target.position}."
            )
        actions, _ = board.dirs_to(ship.position, target.position)
        actions = set(actions)
        ship_to_mining_actions[ship] = actions
        ship_id_to_target[ship] = target.position

    return ship_to_mining_actions


def _mining_cells(board: MyBoard):
    """ Where do we need to mine? """
    return [p for p, h in board.position_to_halite.items() if h >= 50]


def _num_turns_to_mine(cargo: float, cell_halite: float, rt_travel: int):
    """ How many turns should we plan on mining? """
    if not cargo:
        ch = 0
    elif not cell_halite:
        ch = TURNS_OPTIMAL.shape[0] - 1
    else:
        ch = int(math.log(cargo / cell_halite) * 2.5 + 5.5)
        ch = min(max(ch, 0), TURNS_OPTIMAL.shape[0] - 1)

    rt_travel = min(max(rt_travel, 0), TURNS_OPTIMAL.shape[1] - 1)
    return TURNS_OPTIMAL[ch, rt_travel]


def _halite_per_turn(cargo: float, halite: float, travel: int, min_mine: int = 1):
    """
    Compute return from going to a cell containing halite, using optimal number of mining steps
    returns halite mined per turn, optimal number of mining steps
    Turns could be zero, meaning it should just return to a shipyard (subject to min_mine)
    """
    turns = _num_turns_to_mine(cargo, halite, travel)
    if turns < min_mine:
        turns = min_mine
    mined = cargo + (1 - 0.75 ** turns) * halite
    return mined / (travel + turns), mined


def _attack_shipyards(board: MyBoard):
    """ Send our ships to enemy shipyards """
    for ship in board.free_ships:
        for _pos, _ in board.get_neighbor_positions(ship.position, add_self=False):
            _sy = board.position_to_shipyard.get(_pos)
            if not _sy or _sy.player_id == board.me.id:
                continue

            if _do_attack_shipyard(board, ship, _sy):
                continue


def _do_attack_shipyard(
    board: MyBoard,
    my_ship: Ship,
    shipyard: Shipyard,
    max_ship_cargo: float = 0,
    min_num_my_ships: int = 10,
) -> bool:
    """
    If True, we send our ship to attack the enemy shipyard, our ship will be destroyed!
    Glory to the heroes!!!
    """
    if shipyard.position in board.taken_cells:
        # another ship already attacking the shipyard
        return False

    if my_ship.halite > max_ship_cargo or board.num_my_ships <= min_num_my_ships:
        # we can't sacrifice this ship
        return False

    action, distance = board.dirs_to(my_ship.position, shipyard.position)
    assert distance == 1
    action = action[0]

    enemy = shipyard.player
    assert enemy.id != board.me.id

    if enemy.halite >= board.configuration.spawn_cost:
        # The enemy can create a new ship and protect the shipyard
        return False

    enemy_support = _shipyard_support(board, shipyard)
    if enemy_support is not None and enemy_support <= my_ship.halite:
        # the enemy has a support ship
        return False

    logger.info(
        f"Ship {my_ship.id} at {my_ship.position} -> "
        f"Attack not protected enemy shipyard at {shipyard.position}."
    )
    board.move_ship(my_ship, action)
    return True


def _shipyard_support(board: MyBoard, shipyard: Shipyard):
    cargo = []
    for pos, _ in board.get_neighbor_positions(shipyard.position, add_self=True):
        ship = board.position_to_ship.get(pos)
        if ship and ship.player_id != board.me.id:
            cargo.append(ship.halite)
    return min(cargo) if cargo else None


def _move_ships_without_escape(board: MyBoard):
    """ Trying to save our ships from a desperate situation """
    for ship in board.free_ships:
        avoid_moves = board.avoid_moves(ship, avoid_stalemate=True)
        if ALL_SHIP_ACTIONS - avoid_moves:
            # can escape
            continue

        logger.warning(
            f"Ship {ship.id} at {ship.position}: Can't run away, looking for the least dangerous position."
        )

        action_to_probability = {}
        for escape_position, action in board.get_neighbor_positions(
            ship.position, add_self=False
        ):
            p = board.estimate_probability_to_survive(ship, escape_position)
            logger.info(
                f"Ship {ship.id} at {ship.position}: Probability to survive in the {action} = {p}"
            )
            action_to_probability[action] = p

        best_prob = max(action_to_probability.values())
        best_actions = [a for a, p in action_to_probability.items() if p == best_prob]
        best_action = sorted(
            best_actions,
            key=lambda x: board.nearest_shipyard(
                ship.position.translate(x.to_point(), board.size), player=board.me
            )[1],
        )[0]

        logger.info(
            f"Ship {ship.id} at {ship.position}: Save ourselves by moving to the {best_action}."
        )
        board.move_ship(ship, best_action)


def _split_ships_into_groups(
    board: MyBoard, ships: List[Ship]
) -> Dict[int, List[Ship]]:
    """
    Split ships into groups
    The movement of each ship in a group affects the possible movement of all other ships in this group
    """
    group_to_ships = defaultdict(list)
    cell_to_group = {}

    next_group_id_id = 0
    for ship in ships:
        taken_moves = board.taken_moves(ship)
        ship_cells = []
        for pos, action in board.get_neighbor_positions(ship.position, add_self=True):
            if action not in taken_moves:
                ship_cells.append(pos)

        cell_groups = {cell_to_group[x] for x in ship_cells if x in cell_to_group}
        num_groups = len(cell_groups)

        if num_groups == 0:
            # create new group
            next_group_id_id += 1
            group_id = next_group_id_id
        elif num_groups == 1:
            # use it
            group_id, *_ = cell_groups
        else:
            # combine groups
            group_id, *other_groups = cell_groups
            for cell, d in cell_to_group.items():
                if d in other_groups:
                    cell_to_group[cell] = group_id
            for d in other_groups:
                group_to_ships[group_id] += group_to_ships.pop(d)

        for cell in ship_cells:
            cell_to_group[cell] = group_id
        group_to_ships[group_id].append(ship)

    return group_to_ships


def _attack_ships(board: MyBoard):
    for player in board.opponents:
        for enemy_vessel in player.ships:
            if board.can_escape(enemy_vessel, murderer=None, is_stalemate_danger=False):
                continue

            logger.info(
                f"Enemy vessel {enemy_vessel.id} at {enemy_vessel.position} can escape, press it."
            )

            ship_to_action = {}
            for press_pos, _ in board.get_neighbor_positions(
                enemy_vessel.position, add_self=False
            ):
                if press_pos in board.taken_cells:
                    continue

                for pos, action in board.get_neighbor_positions(
                    press_pos, add_self=True
                ):
                    ship = board.position_to_ship.get(pos)
                    if (
                        not ship
                        or ship.player_id != board.me.id
                        or ship in board.taken_ships
                        or ship.halite >= enemy_vessel.halite
                        or board.is_danger_position(press_pos, ship)
                    ):
                        continue

                    ship_to_action[ship] = reverse_action(action)
                    break

            if not ship_to_action:
                continue

            if all(a is None for a in ship_to_action.values()):
                # all ships can't just wait, we send the heaviest ship to attack
                ship = sorted(ship_to_action, key=lambda x: -x.halite)[0]
                actions, distance = board.dirs_to(ship.position, enemy_vessel.position)
                assert distance == 1 and len(actions) == 1
                logger.info(
                    f"Ship {ship.id} at {ship.position}: Attack the enemy vessel {enemy_vessel.id}"
                )
                board.move_ship(ship, actions[0])
                continue

            for ship, action in ship_to_action.items():
                logger.info(
                    f"Ship {ship.id} at {ship.position}: Press the enemy vessel {enemy_vessel.id}"
                )
                board.move_ship(ship, action)
