from agents import Q, QNet, Sarsa
import numpy as np

from game import GridWorld, MoveRight


def qlearn_grid() -> list[float] | None:
    # game = MoveRight(4)
    game = GridWorld()
    episodes = 1000
    alpha = 0.05
    gamma = 0.99
    epsilon = 1.0
    epsilon_final = 0.01
    epsilon_decay = (epsilon - epsilon_final) / episodes
    agent = Q[tuple[int, int], int](
        game,
        epsilon,
        epsilon_decay,
        gamma,
        alpha,
    )
    # Training loop
    for ep in range(episodes):
        state = game.reset()
        loss = 0
        turns = 0
        while not game.done():
            action = agent.action(state)
            next_state, reward = game.step(action)
            loss += agent.update(game.done(), state, action, reward, next_state)
            state = next_state
            turns += 1

        agent.decay_epsilon()

        print(f"\nEpisode {ep + 1}: loss {loss}, turns {turns}")
    return agent.training_loss


def qnet() -> list[float] | None:
    # game = MoveRight(4)
    game = GridWorld()
    episodes = 500
    alpha = 0.05
    gamma = 0.99
    epsilon = 1.0
    epsilon_final = 0.01
    epsilon_decay = (epsilon - epsilon_final) / episodes
    agent = QNet[tuple[int, int], int](
        game,
        epsilon,
        epsilon_decay,
        gamma,
        alpha,
    )
    # Training loop
    for ep in range(episodes):
        state = game.reset()
        loss = 0
        turns = 0
        while not game.done():
            action = agent.action(state)
            next_state, reward = game.step(action)
            loss += agent.update(game.done(), state, action, reward, next_state)
            state = next_state
            turns += 1

        agent.decay_epsilon()

        print(f"\nEpisode {ep + 1}: loss {loss}, turns {turns}")
    return agent.training_loss


def qlearn() -> list[float]:
    game = MoveRight(4)
    episodes = 100
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_final = 0.01
    epsilon_decay = (epsilon - epsilon_final) / episodes
    agent = Q(
        game,
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
            agent.update(game.done, state, action, reward, next_state)
            state = next_state

        agent.decay_epsilon()

        print(f"\nEpisode {ep + 1}")
        print(agent.q)

    return agent.training_loss


def sarsa() -> list[float]:
    game = MoveRight(4)

    episodes = 200
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_final = 0.01
    epsilon_decay = (epsilon - epsilon_final) / episodes

    agent = Sarsa(
        game,
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
    if loss := qlearn_grid():
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
