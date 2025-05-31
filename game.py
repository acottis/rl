from enum import Enum
from typing import Protocol, SupportsIndex, TypeVar, final, override

State = TypeVar("State")
Action = TypeVar("Action", bound=SupportsIndex)
Reward = float


class Game(Protocol[State, Action]):
    def actions(self) -> list[Action]:
        """Returns list of valid actions."""
        ...

    def reset(self) -> State:
        """Reset Game."""
        ...

    def step(self, action: Action) -> tuple[State, Reward]:
        """One turn of the game and the result."""
        ...

    def done(self) -> bool:
        """Is the game over."""
        ...

    def one_hot(self, state: State) -> int:
        """Convert to one hot index."""
        ...


@final
class MoveRight(Game[int, int]):
    class Actions(Enum):
        LEFT = 0
        RIGHT = 1

    _state: int
    _done: bool
    _reward: float
    _goal: int

    def __init__(self, goal: int):
        self._goal = goal
        self._state = 0
        self._done = False
        self._reward = 1.0

    @override
    def actions(self) -> list[int]:
        return [action.value for action in self.Actions]

    @override
    def reset(self) -> int:
        self._state = 0
        self._done = False
        return self._state

    @override
    def step(self, action: int) -> tuple[int, Reward]:
        if action == 0:
            self._state = max(0, self._state - 1)
        else:
            self._state = min(4, self._state + 1)

        self._done = self._state == self._goal
        reward = self._reward if self._done else -0.2
        return self._state, reward

    @override
    def done(self) -> bool:
        return self._done


@final
class GridWorld(Game[tuple[int, int], int]):
    class Actions(Enum):
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3

    grid_size: int
    state: tuple[int, int]
    goal: tuple[int, int]
    obstacles: set[tuple[int, int]]
    _done: bool

    def __init__(self, grid_size: int = 5, goal=(4, 4), obstacles=None):
        self.grid_size = grid_size
        self.goal = goal
        self.obstacles = obstacles or {(1, 1), (2, 2), (3, 3)}
        self.state = (0, 0)
        self._done = False

    def states(self) -> int:
        return self.grid_size * self.grid_size

    @override
    def actions(self) -> list[int]:
        return [a.value for a in self.Actions]

    @override
    def reset(self) -> tuple[int, int]:
        self.state = (0, 0)
        self._done = False
        return self.state

    @override
    def step(self, action: int) -> tuple[tuple[int, int], Reward]:
        x, y = self.state
        if action == self.Actions.UP.value:
            y = max(0, y - 1)
        elif action == self.Actions.DOWN.value:
            y = min(self.grid_size - 1, y + 1)
        elif action == self.Actions.LEFT.value:
            x = max(0, x - 1)
        elif action == self.Actions.RIGHT.value:
            x = min(self.grid_size - 1, x + 1)

        next_state = (x, y)
        if next_state in self.obstacles:
            next_state = self.state  # blocked, stay put

        self.state = next_state

        if self.state == self.goal:
            self._done = True
            return self.state, 1.0
        return self.state, -0.1

    @override
    def __str__(self) -> str:
        rows: list[str] = []
        for y in range(self.grid_size):
            row = ""
            for x in range(self.grid_size):
                pos = (x, y)
                if pos == self.state:
                    row += " A "  # Agent
                elif pos == self.goal:
                    row += " G "  # Goal
                elif pos in self.obstacles:
                    row += " # "  # Obstacle
                else:
                    row += " . "  # Empty cell
            rows.append(row)
        return "\n".join(rows)

    @override
    def done(self) -> bool:
        return self._done

    @override
    def one_hot(self, state: tuple[int, int]):
        x, y = state
        return y * self.grid_size + x
