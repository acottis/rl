from typing import Generic
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from game import Action, Game, State


class Sarsa(Generic[State, Action]):
    q: np.typing.NDArray[np.float64]
    # exploration rate
    epsilon: float
    # how quickly we stop exploring
    epsilon_decay: float
    # For metrics
    training_loss: list[float] = []
    alpha: float
    gamma: float

    game: Game[State, Action]

    def __init__(
        self,
        game: Game[State, Action],
        epsilon: float,
        epsilon_decay: float,
        gamma: float,
        alpha: float,
    ):
        self.game = game
        self.q = np.zeros((5, len(game.actions())))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma

    def action(self, state: State) -> Action:
        """Choose action based on current Q Table and epsilon."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.array(self.game.actions()))
        else:
            # The index of the highest value
            return np.argmax(self.q[state])

    def update(
        self,
        # Is the game over
        done: bool,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_action: Action,
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


class Q(Generic[State, Action]):
    # exploration rate
    epsilon: float
    # how quickly we stop exploring
    epsilon_decay: float
    # learning rate
    alpha: float
    # discount factor - How much earlier actions matter. High == Long term thinking
    gamma: float
    # List of valid actions
    game: Game[State, Action]
    # The Q table - For a given state what action should we do
    q: np.typing.NDArray[np.float64]
    # For metrics
    training_loss: list[float] = []

    def __init__(
        self,
        game: Game[State, Action],
        epsilon: float,
        epsilon_decay: float,
        gamma: float,
        alpha: float,
    ):
        assert gamma < 1.0, "If gamma is 1.0 or above we never lower bad scores"
        assert gamma > 0.0, "Gamme must be over 0.0 or we dont get rewarded"

        self.game = game
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.q = np.zeros((5, len(self.game.actions())))

    def action(self, state: State) -> Action:
        """Choose action based on current Q Table and epsilon."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.array(self.game.actions()))
        else:
            # The index of the highest value
            return np.argmax(self.q[state])

    def update(
        self,
        done: bool,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
    ) -> None:
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q[next_state])

        loss = target - self.q[state, action]
        self.training_loss.append(loss)

        # Update the last execution actions score based on our hyperparams
        self.q[state, action] += self.alpha * loss

    def decay_epsilon(self):
        self.epsilon -= self.epsilon_decay


class QNet(Generic[State, Action]):
    q: nn.Linear
    epsilon: float
    epsilon_decay: float
    game: Game[State, Action]
    gamma: float
    alpha: float
    training_loss: list[float] = []

    loss_fn: nn.MSELoss

    def __init__(
        self,
        game: Game[State, Action],
        epsilon: float,
        epsilon_decay: float,
        gamma: float,
        alpha: float,
    ):
        self.game = game
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.alpha = alpha

        # Neural Network
        self.q = nn.Linear(5, 2)
        self.optimizer = torch.optim.SGD(self.q.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

    def action(self, state: State) -> Action:
        """Choose action based on current Q Table and epsilon."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.array(self.game.actions()))
        else:
            with torch.no_grad():
                table = self.q(self.one_hot(state))
                return torch.argmax(table).item()

    def one_hot(self, state: State) -> torch.Tensor:
        return F.one_hot(torch.tensor(state), num_classes=5).float().unsqueeze(0)

    def update(
        self,
        done: bool,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
    ) -> float:
        q_values = self.q(self.one_hot(state)).squeeze(0)
        with torch.no_grad():
            next_q_values = self.q(self.one_hot(next_state)).squeeze(0)

        if done:
            target = torch.tensor(reward)
        else:
            target = reward + self.gamma * torch.max(next_q_values)

        prediction = q_values[action]

        loss = self.loss_fn(prediction, target)
        self.training_loss.append(loss.item())

        # Update the last execution actions score based on our hyperparams
        self.optimizer.zero_grad()
        loss.backward()
        _ = self.optimizer.step()
        return loss

    def decay_epsilon(self):
        self.epsilon -= self.epsilon_decay
