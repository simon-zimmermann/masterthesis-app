import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Q-Learning Parameter
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1.0
exploration_decay = 0.99
episodes = 500

# Umgebung
size = 8
start = (0, 0)
goal = (7, 5)
obstacles = [(3, 3), (3, 4), (4, 3), (4, 4)]

# Q-Tabelle initialisieren
q_table = np.zeros((size, size, 4))  # 4 Aktionen: hoch, runter, links, rechts


def is_valid(state):
    """Prüft, ob ein Zustand (Feld) gültig ist (nicht außerhalb der Grenzen und kein Hindernis)."""
    if state[0] < 0 or state[0] >= size or state[1] < 0 or state[1] >= size:
        return False
    if state in obstacles:
        return False
    return True


def get_next_state(state, action):
    """Ermittelt den nächsten Zustand basierend auf der aktuellen Position und der Aktion."""
    next_state = list(state)
    if action == 0:  # hoch
        next_state[0] -= 1
    elif action == 1:  # runter
        next_state[0] += 1
    elif action == 2:  # links
        next_state[1] -= 1
    elif action == 3:  # rechts
        next_state[1] += 1
    next_state = tuple(next_state)
    if is_valid(next_state):
        return next_state
    else:
        return state


def choose_action(state):
    """Wählt eine Aktion basierend auf der epsilon-greedy Strategie."""
    if np.random.rand() < exploration_rate:
        return np.random.randint(4)  # zufällige Aktion
    else:
        return np.argmax(q_table[state[0], state[1]])  # beste Aktion


def train():
    global exploration_rate
    for episode in range(episodes):
        state = start
        step_count = 0
        print(f"Episode {episode + 1}/{episodes} gestartet.")
        while state != goal:
            action = choose_action(state)
            next_state = get_next_state(state, action)
            reward = 1 if next_state == goal else -0.1
            q_table[state[0], state[1], action] = q_table[state[0], state[1], action] + \
                learning_rate * (reward + discount_factor *
                                 np.max(q_table[next_state[0], next_state[1]]) - q_table[state[0], state[1], action])
            state = next_state
            step_count += 1
            #print(f"  Schritt {step_count}: Zustand {state}, Aktion {action}")

        print(f"Episode {episode + 1} abgeschlossen. Schritte: {step_count}")
        exploration_rate *= exploration_decay


def plot_path():
    state = start
    path = [state]
    while state != goal:
        action = np.argmax(q_table[state[0], state[1]])
        state = get_next_state(state, action)
        path.append(state)

    # Erstelle das Grid
    grid = np.zeros((size, size))
    for obstacle in obstacles:
        grid[obstacle] = -1
    grid[goal] = 2

    # Pfad zeichnen
    path_x, path_y = zip(*path)
    plt.plot(path_y, path_x, marker='o')

    # Grid zeichnen
    cmap = ListedColormap(['white', 'black', 'red'])
    plt.imshow(grid, cmap=cmap, origin='upper', extent=[-0.5, size-0.5, -0.5, size-0.5])
    plt.gca().invert_yaxis()
    plt.xticks(range(size))
    plt.yticks(range(size))
    plt.grid()
    plt.title("Gitter-Umgebung mit Pfad")
    plt.show()


# Training
train()

# Ergebnis anzeigen
plot_path()
