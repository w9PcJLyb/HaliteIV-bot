import warnings
import numpy as np
from typing import List, Optional, Tuple, Union, Dict
from functools import lru_cache
from collections import defaultdict
from kaggle_environments.envs.halite.helpers import (
    Ship,
    Board,
    Point,
    Player,
    Shipyard,
    ShipAction,
    ShipyardAction,
)
from .logger import logger

ALL_SHIP_ACTIONS = {None, *ShipAction.moves()}
REVERS_ACTION = {
    None: None,
    ShipAction.WEST: ShipAction.EAST,
    ShipAction.EAST: ShipAction.WEST,
    ShipAction.NORTH: ShipAction.SOUTH,
    ShipAction.SOUTH: ShipAction.NORTH,
}


class MyBoard(Board):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.step == 0:
            logger.info(f"Start game: {self.configuration}.")

        logger.info(f"Step {self.step + 1}.")

        self.size = self.configuration.size
        self.max_steps = self.configuration.episode_steps
        self.me = self.players[self.current_player_id]
        self.my_ships = self.me.ships
        self.my_shipyards = self.me.shipyards
        self.my_halite = self.me.halite
        self.num_my_ships = len(self.my_ships)
        self.num_my_shipyards = len(self.my_shipyards)
        self.moves_left = self.configuration.episode_steps - 1 - self.step

        self.position_to_ship: Dict[Point, Ship] = {
            x.position: x for x in self.ships.values()
        }
        self.position_to_shipyard: Dict[Point, Shipyard] = {
            x.position: x for x in self.shipyards.values()
        }
        self.position_to_halite: Dict[Point, float] = {
            p: c.halite for p, c in self.cells.items()
        }
        self._player_to_occupation_matrix = self._get_occupation_matrix()

        self.ship_to_next_position: Dict[Ship, Point] = {}

        self._player_occupation_matrix_mem = {}

    @property
    def taken_ships(self) -> List[Ship]:
        moved_ships = list(self.ship_to_next_position.keys())
        converted_ships = [
            x for x in self.my_ships if x.next_action == ShipAction.CONVERT
        ]
        return moved_ships + converted_ships

    @property
    def taken_cells(self) -> List[Point]:
        by_ships = list(self.ship_to_next_position.values())
        by_shipyards = [
            x.position
            for x in self.my_shipyards
            if x.next_action == ShipyardAction.SPAWN
        ]
        return by_ships + by_shipyards

    @property
    def free_ships(self) -> List[Ship]:
        return [x for x in self.my_ships if x not in self.taken_ships]

    @property
    def free_shipyards(self) -> List[Shipyard]:
        return [
            x
            for x in self.my_shipyards
            if x.position not in self.taken_cells and not x.next_action
        ]

    def _get_occupation_matrix(self):
        """
        Return a dict: player_id -> np.array (board.size x board.size)
        Values - the minimum possible value of ship's cargo in this cell at the next step.
        """
        player_cargo = defaultdict(
            lambda: defaultdict(list)
        )  # player_id -> position -> list of cargo

        for pos, sy in self.position_to_shipyard.items():
            player_id = sy.player_id

            # a shipyard can create new ship
            player_cargo[player_id][pos].append(0)

        for pos, ship in self.position_to_ship.items():
            player_id = ship.player_id
            ship_halite = ship.halite

            # a ship can mine halite
            player_cargo[player_id][pos].append(ship_halite)

            # a ship can move
            for nest_pos, _ in self.get_neighbor_positions(pos, add_self=False):
                player_cargo[player_id][nest_pos].append(ship_halite)

        out = {}
        for player_id, pos_to_cargo in player_cargo.items():
            m = np.zeros((self.size, self.size), dtype=np.float32)
            m[:] = np.nan
            for (x, y), cargo_list in pos_to_cargo.items():
                m[x, y] = min(cargo_list)
            out[player_id] = m

        return out

    def player_occupation_matrix(
        self, player: Union[Player, List[Player]]
    ) -> Optional[np.array]:
        if isinstance(player, Player):
            ids = (player.id,)
        else:
            ids = tuple(x.id for x in player)

        if ids in self._player_occupation_matrix_mem:
            return self._player_occupation_matrix_mem[ids]

        m = [
            self._player_to_occupation_matrix[x]
            for x in ids
            if x in self._player_to_occupation_matrix
        ]
        if m:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
                m = np.nanmin(m, axis=0)
        else:
            m = None

        self._player_occupation_matrix_mem[ids] = m
        return m

    def player_occupation(
        self, position, player: Union[Player, List[Player]]
    ) -> Optional[float]:
        occupation_matrix = self.player_occupation_matrix(player)
        if occupation_matrix is None:
            return
        x, y = position
        occupation = occupation_matrix[x, y]
        return int(occupation) if not np.isnan(occupation) else None

    def get_neighbor_positions(
        self, p: Point, add_self: bool = False
    ) -> List[Tuple[Point, Optional[ShipAction]]]:
        """ List of neighbors """
        return _get_neighbors(p, self.size, add_self)

    def environment_reserves(self, position: Point) -> float:
        """ Return amount of halite in the nearest cells """
        halite = 0
        for p, cell_halite in self.position_to_halite.items():
            if p != position and self.distance(p, position) <= 2:
                halite += cell_halite
        return halite

    def is_win(self) -> bool:
        """ Are we winning the game? """
        our_score = self.player_score(self.me)
        best_enemy_score = max(self.player_score(x) for x in self.opponents)
        return our_score > best_enemy_score

    @staticmethod
    def player_score(player: Player) -> float:
        """ The amount of player's halite """
        if not player.ships and not player.shipyards:
            return -np.inf
        return player.halite

    def attacker_cargo(
        self, position: Point, defender: Player, murderer: Optional[Player] = None
    ) -> Optional[float]:
        """
        Take all ships that can move to thу cell and return the minimum value of the cargo of these ships
        If there are no such ships, return None
        """
        if murderer:
            assert defender.id != murderer.id
            murderers = [murderer]
        else:
            murderers = [x for x in self.players.values() if x != defender]

        return self.player_occupation(position, murderers)

    def is_danger_position(
        self,
        position: Point,
        ship: Ship,
        is_stalemate_danger: bool = True,
        is_hunters_danger: bool = False,
        murderer: Optional[Player] = None,
    ) -> bool:
        """ Is this cell a danger to the ship? """
        player = ship.player
        shipyard = self.position_to_shipyard.get(position)
        if shipyard and shipyard.player_id != player.id:
            # an enemy shipyard is a dangerous place
            return True

        if is_hunters_danger:
            num_hunters = 0
            for other_ship in self.ships.values():
                if (
                    other_ship.player_id != ship.player_id
                    and other_ship.halite < ship.halite
                    and self.distance(other_ship.position, position) <= 2
                ):
                    num_hunters += 1
            if num_hunters > 2:
                return True

        attacker_cargo = self.attacker_cargo(
            position, defender=player, murderer=murderer
        )
        if attacker_cargo is None:
            return False

        if is_stalemate_danger:
            return attacker_cargo <= ship.halite
        else:
            return attacker_cargo < ship.halite

    def create_ship(self, shipyard: Shipyard):
        """ Create a ship by the shipyard """
        logger.debug(f"Shipyard {shipyard.id} at {shipyard.position} -> Create a ship.")
        shipyard.next_action = ShipyardAction.SPAWN
        self.num_my_ships += 1
        self.my_halite -= self.configuration.spawn_cost

    def create_shipyard(self, ship: Ship):
        """ Create a shipyard from the ship """
        logger.debug(f"Ship {ship.id} at {ship.position} -> Create a shipyard.")
        ship.next_action = ShipAction.CONVERT
        self.num_my_shipyards += 1
        self.num_my_ships -= 1
        self.my_halite -= self.configuration.convert_cost

    def move_ship(self, ship: Ship, action: Optional[ShipAction]):
        """ Apply the action to the ship """
        if action is None:
            in_shipyard = ship.position in self.position_to_shipyard
            logger.debug(
                f"Ship {ship.id} at {ship.position} -> "
                f"{'MINE' if not in_shipyard else 'PROTECT'}."
            )
            ship.next_action = None
            self.ship_to_next_position[ship] = ship.position
        elif action in ShipAction.moves():
            logger.debug(f"Ship {ship.id} at {ship.position} -> {action}.")
            ship.next_action = action
            self.ship_to_next_position[ship] = ship.position.translate(
                action.to_point(), self.size
            )
        else:
            raise ValueError(f"Unknown action '{action}'.")

    def distance(self, a: Point, b: Point) -> int:
        """ Minimum manhattan distance between two cells """
        _, d = self.dirs_to(a, b)
        return d

    def dirs_to(self, a: Point, b: Point) -> Tuple[list, int]:
        """ Actions you should take to go from one cell to another """
        x, y = b - a
        return _dirs_to(x, y, self.size)

    def nearest_shipyard(
        self, position: Point, player: Union[Player, List[Player]]
    ) -> Tuple[Optional[Shipyard], float]:
        """ Return the closest shipyard to the cell and the distance to it """
        if not isinstance(player, list):
            player = [player]
        player_ids = {x.id for x in player}
        sy, d = None, np.inf
        for _sy in self.shipyards.values():
            if _sy.player_id not in player_ids:
                continue
            _d = self.distance(_sy.position, position)
            if _d > d:
                continue
            sy, d = _sy, _d
        return sy, d

    def attack_moves(
        self, ship: Ship, add_moves_when_an_enemy_can_escape: bool = True
    ) -> set:
        """ Return a set of actions for this ship that can lead to the destruction of an enemy vessel """
        moves = set()
        for p, action in self.get_neighbor_positions(ship.position, add_self=False):
            other_ship = self.position_to_ship.get(p)
            other_shipyard = self.position_to_shipyard.get(p)
            if (
                other_ship
                and not other_shipyard
                and other_ship.player_id != ship.player_id
                and other_ship.halite > ship.halite
            ):
                if add_moves_when_an_enemy_can_escape or not self.can_escape(
                    other_ship
                ):
                    moves.add(action)
        return moves

    def avoid_moves(
        self, ship: Ship, avoid_stalemate: bool = True, avoid_hunters: bool = False
    ) -> set:
        """ Return a set of actions for this ship that can lead to the loss of the vessel """
        moves = set()
        for p, action in self.get_neighbor_positions(ship.position, add_self=True):
            if self.is_danger_position(
                p,
                ship,
                is_hunters_danger=avoid_hunters,
                is_stalemate_danger=avoid_stalemate,
            ):
                moves.add(action)
        return moves

    def taken_moves(self, ship: Ship) -> set:
        """ Return set of actions that lead to the occupied cells """
        moves = set()
        taken_cells = self.taken_cells
        for p, action in self.get_neighbor_positions(ship.position, add_self=True):
            if p in taken_cells:
                moves.add(action)
        return moves

    def can_escape(
        self,
        ship: Ship,
        murderer: Optional[Player] = None,
        is_stalemate_danger: bool = True,
    ) -> bool:
        """ If True the ship can move to another cell """
        for p, _ in self.get_neighbor_positions(ship.position):
            if not self.is_danger_position(
                p, ship, murderer=murderer, is_stalemate_danger=is_stalemate_danger
            ):
                return True
        return False

    def estimate_probability_to_survive(self, ship: Ship, position: Point) -> float:
        """ The probability that the ship will survive in the cell """
        p = 1
        for pos, action in self.get_neighbor_positions(position, add_self=True):
            action = reverse_action(action)
            other_ship = self.position_to_ship.get(pos)
            if (
                not other_ship
                or other_ship.player_id == ship.player.id
                or other_ship.halite > ship.halite
            ):
                continue

            avoid_moves = self.avoid_moves(other_ship)
            if action in avoid_moves:
                continue

            attack_actions = self.attack_moves(other_ship)
            if action in attack_actions:
                p *= 0.01
                continue

            num_moves = len(ALL_SHIP_ACTIONS - avoid_moves)
            assert num_moves > 0

            # The probability that other_ship will not be in the cell on the next turn
            p *= 1 - 1 / num_moves

        return p

    def num_hunters_around(
        self, position: Point, murderer: Union[Player, List[Player]], max_distance: int
    ) -> int:
        """ Number of ships without cargo around the cell """
        if max_distance == 0:
            return 0
        if not isinstance(murderer, list):
            murderer = [murderer]
        player_ids = {x.id for x in murderer}
        num_hunters = 0
        for ship in self.ships.values():
            if (
                ship.player_id in player_ids
                and ship.halite == 0
                and self.distance(ship.position, position) <= max_distance
            ):
                num_hunters += 1
        return num_hunters

    def distance_to_enemy_ship(self, position: Point, player: Player) -> Optional[int]:
        """ Distance to the nearest enemy vessel, if no enemy ships return None """
        min_distance = np.inf
        for ship_pos, ship in self.position_to_ship.items():
            if ship.player_id != player.id:
                min_distance = min(min_distance, self.distance(ship_pos, position))
        return min_distance if not np.isinf(min_distance) else None


@lru_cache(maxsize=1024)
def _dirs_to(x: int, y: int, size: int) -> Tuple[List[Optional[ShipAction]], int]:
    if x == 0 and y == 0:
        return [None], 0

    if abs(x) > size / 2:
        x -= np.sign(x) * size
    if abs(y) > size / 2:
        y -= np.sign(y) * size

    ret = []
    if x > 0:
        ret.append(ShipAction.EAST)
    elif x < 0:
        ret.append(ShipAction.WEST)
    if y > 0:
        ret.append(ShipAction.NORTH)
    elif y < 0:
        ret.append(ShipAction.SOUTH)

    return ret, abs(x) + abs(y)


@lru_cache(maxsize=1024)
def _get_neighbors(
    point: Point, size: int, add_self: bool
) -> List[Tuple[Point, Optional[ShipAction]]]:
    out = []
    for action in ShipAction.moves():
        out.append((point.translate(action.to_point(), size), action))
    if add_self:
        out.append((point, None))
    return out


def reverse_action(action: Optional[ShipAction]) -> Optional[ShipAction]:
    return REVERS_ACTION[action]
