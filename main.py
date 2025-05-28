from agents import Q, ActionEnum, Sarsa
import numpy as np


class Game:
    class Action(ActionEnum):
        LEFT = 0
        RIGHT = 1

    state: int = 0
    done: bool = False
    reward: float = 1.0
    goal: int

    def __init__(self, goal: int):
        self.goal = goal

    def reset(self) -> int:
        self.state = 0
        self.done = False
        return self.state

    def step(self, action: Action) -> tuple[int, float]:
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(4, self.state + 1)

        self.done = self.state == self.goal
        reward = self.reward if self.done else -0.2
        return self.state, reward


def qlearn() -> list[float]:
    game = Game(4)
    episodes = 100
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_final = 0.01
    epsilon_decay = (epsilon - epsilon_final) / episodes
    agent = Q[int, Game.Action](
        Game.Action,
        epsilon,
        epsilon_decay,
        gamma,
        alpha,
    )
    # Training loop
    for ep in range(episodes):
        state = game.reset()
        while not game.done:
            action = agent.action(state)
            next_state, reward = game.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

        agent.decay_epsilon()

        print(f"\nEpisode {ep + 1}")
        print(agent.q)

    return agent.training_loss


def sarsa() -> list[float]:
    game = Game(4)

    episodes = 2000
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_final = 0.01
    epsilon_decay = (epsilon - epsilon_final) / episodes

    agent = Sarsa[int, Game.Action](
        Game.Action,
        epsilon,
        epsilon_decay,
        gamma,
        alpha,
    )

    # Training loop
    for ep in range(episodes):
        state = game.reset()
        action = agent.action(state)

        while not game.done:
            next_state, reward = game.step(action)
            next_action = agent.action(next_state)
            agent.update(game.done, state, action, reward, next_state, next_action)
            state, action = next_state, next_action

        agent.decay_epsilon()

        print(f"\nEpisode {ep + 1}")
        print(agent.q)

    return agent.training_loss


def main():
    loss = sarsa()
    plot(loss)


def plot(training_loss: list[float]):
    import matplotlib.pyplot as plt

    window_size = max(1, int(len(training_loss) / 10))
    smoothed = np.convolve(
        training_loss, np.ones(window_size) / window_size, mode="valid"
    )

    _ = plt.plot(smoothed)
    plt.show()


main()
