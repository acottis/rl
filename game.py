from enum import Enum
from typing import Protocol, SupportsIndex, SupportsInt, TypeVar, final, override

State = TypeVar("State", bound=SupportsIndex)
Action = TypeVar("Action", bound=SupportsIndex)
Reward = float


class Game(Protocol[State, Action]):
    def actions(self) -> list[Action]:
        """Returns list of valid actions."""
        ...

    def reset(self) -> State:
        """Reset Game."""
        ...

    def step(self, action: Action) -> tuple[Action, Reward]:
        """One turn of the game and the result."""
        ...


@final
class MoveRight(Game[int, int]):
    class Actions(Enum):
        LEFT = 0
        RIGHT = 1

    state: int = 0
    done: bool = False
    reward: float = 1.0
    goal: int

    def __init__(self, goal: int):
        self.goal = goal

    @override
    def actions(self) -> list[int]:
        return [action.value for action in self.Actions]

    @override
    def reset(self) -> int:
        self.state = 0
        self.done = False
        return self.state

    @override
    def step(self, action: int) -> tuple[int, Reward]:
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(4, self.state + 1)

        self.done = self.state == self.goal
        reward = self.reward if self.done else -0.2
        return self.state, reward
