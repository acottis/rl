from enum import Enum
from typing import Generic, SupportsIndex, TypeVar
import numpy as np


class ActionEnum(int, Enum):
    pass


StateType = TypeVar("StateType", bound=SupportsIndex)
ActEnumType = TypeVar("ActEnumType", bound=ActionEnum)
ActType = TypeVar("ActType")


class Sarsa(Generic[StateType, ActEnumType]):
    q: np.typing.NDArray[np.float64]
    # exploration rate
    epsilon: float
    # how quickly we stop exploring
    epsilon_decay: float
    # For metrics
    training_loss: list[float] = []
    actions: type[ActEnumType]
    alpha: float
    gamma: float

    def __init__(
        self,
        actions: type[ActEnumType],
        epsilon: float,
        epsilon_decay: float,
        gamma: float,
        alpha: float,
    ):
        self.actions = actions
        self.q = np.zeros((5, len(self.actions)))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma

    def action(self, state: StateType) -> ActEnumType:
        """Choose action based on current Q Table and epsilon."""
        if np.random.rand() < self.epsilon:
            return self.actions(np.random.choice(list(self.actions)))
        else:
            # The index of the highest value
            return self.actions(np.argmax(self.q[state]))

    def update(
        self,
        # Is the game over
        done: bool,
        state: StateType,
        action: ActEnumType,
        reward: float,
        next_state: StateType,
        next_action: ActEnumType,
    ):
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q[next_state, next_action]

        loss = target - self.q[state, action]

        self.q[state, action] += self.alpha * loss

        self.training_loss.append(loss)

    def decay_epsilon(self):
        self.epsilon -= self.epsilon_decay


class Q(Generic[StateType, ActEnumType]):
    # exploration rate
    epsilon: float
    # how quickly we stop exploring
    epsilon_decay: float
    # learning rate
    alpha: float
    # discount factor - How much earlier actions matter. High == Long term thinking
    gamma: float
    # List of valid actions
    actions: type[ActEnumType]
    # The Q table - For a given state what action should we do
    q: np.typing.NDArray[np.float64]
    # For metrics
    training_loss: list[float] = []

    def __init__(
        self,
        actions: type[ActEnumType],
        epsilon: float,
        epsilon_decay: float,
        gamma: float,
        alpha: float,
    ):
        assert gamma < 1.0, "If gamma is 1.0 or above we never lower bad scores"
        assert gamma > 0.0, "Gamme must be over 0.0 or we dont get rewarded"

        self.actions = actions
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.q = np.zeros((5, len(actions)))

    def action(self, state: StateType) -> ActEnumType:
        """Choose action based on current Q Table and epsilon."""
        if np.random.rand() < self.epsilon:
            return self.actions(np.random.choice(list(self.actions)))
        else:
            # The index of the highest value
            return self.actions(np.argmax(self.q[state]))

    def update(
        self,
        state: StateType,
        action: ActEnumType,
        reward: float,
        next_state: StateType,
    ) -> None:
        # Get the score for the last executed action
        previous_q_value = self.q[state, action.value]
        # Get the score of the biggest next state (how important is this decision to the next score)
        next_max_q_value = np.max(self.q[next_state])
        loss = reward + self.gamma * next_max_q_value - previous_q_value
        self.training_loss.append(loss)

        # Update the last execution actions score based on our hyperparams
        self.q[state, action] += self.alpha * loss

    def decay_epsilon(self):
        self.epsilon -= self.epsilon_decay
