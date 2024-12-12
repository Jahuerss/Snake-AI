import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
import pandas as pd

# Parámetros del entorno y el juego
screen_size = 400
block_size = 20
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Parámetros del aprendizaje por refuerzos
gamma = 0.95  # Aumentar la influencia de recompensas futuras
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.0005  # Reducir para un aprendizaje más estable
batch_size = 64  # Aumentar el tamaño del batch para una mejor generalización
replay_buffer_size = 5000  # Aumentar el tamaño del buffer para más diversidad

# Dispositivo para entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Incrementar número de neuronas
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_state(snake, food, direction, board_size):
    head_x, head_y = snake[-1]
    food_x, food_y = food

    state = [
        (food_x - head_x) / board_size,
        (food_y - head_y) / board_size,
        direction[0],
        direction[1],
    ]

    for dx, dy in directions:
        next_x = head_x + dx * block_size
        next_y = head_y + dy * block_size
        if next_x < 0 or next_x >= screen_size or next_y < 0 or next_y >= screen_size:
            state.append(1)
        elif [next_x, next_y] in snake[:-1]:
            state.append(1)
        else:
            state.append(0)

    return np.array(state, dtype=np.float32)

def choose_action(state, model, epsilon):
    if np.random.rand() < epsilon:
        return random.randint(0, 3)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
    return torch.argmax(q_values).item()

def train_model(model, target_model, replay_buffer, optimizer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    minibatch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = model(states).gather(1, actions)
    next_q_values = target_model(next_states).max(1)[0].detach()
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    loss = nn.MSELoss()(q_values.squeeze(), target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def save_model(model, filename="snake_dqn.pth"):
    torch.save(model.state_dict(), filename)

def load_model(model, filename="snake_dqn.pth"):
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))

def save_to_excel(data, filename="training_data.xlsx"):
    df = pd.DataFrame(data, columns=["Episodio", "Recompensa", "Longitud"])
    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates()
    df.to_excel(filename, index=False)

def generate_food(snake, board_size, block_size):
    while True:
        food_x = random.randint(0, board_size - 1) * block_size
        food_y = random.randint(0, board_size - 1) * block_size
        food = [food_x, food_y]
        # Verifica que la comida no esté dentro de la serpiente
        if food not in snake:
            # Verifica que la comida no esté demasiado cerca de la cabeza
            head_x, head_y = snake[-1]
            distance_to_head = abs(food_x - head_x) + abs(food_y - head_y)
            if distance_to_head >= block_size:  # Cambia según la distancia mínima deseada
                return food

def game_loop(model, target_model, optimizer, replay_buffer, epsilon, episode, data):
    snake = [[screen_size // 2, screen_size // 2]]
    direction = directions[0]
    food = generate_food(snake, screen_size // block_size, block_size)

    total_reward = 0
    done = False

    while not done:
        state = get_state(snake, food, direction, board_size=screen_size // block_size)
        action = choose_action(state, model, epsilon)

        direction = directions[action]
        head_x, head_y = snake[-1]
        new_head = [head_x + direction[0] * block_size, head_y + direction[1] * block_size]
        snake.append(new_head)

        if (new_head in snake[:-1]) or (new_head[0] < 0 or new_head[0] >= screen_size or
                                        new_head[1] < 0 or new_head[1] >= screen_size):
            done = True
            reward = -50  # Reducir penalización por colisión para evitar aprendizaje extremo
        elif new_head == food:
            reward = 100
            food = [random.randint(0, (screen_size // block_size) - 1) * block_size,
                    random.randint(0, (screen_size // block_size) - 1) * block_size]
        else:
            reward = -1
            snake.pop(0)

        total_reward += reward
        next_state = get_state(snake, food, direction, board_size=screen_size // block_size)

        replay_buffer.append((state, action, reward, next_state, done))
        train_model(model, target_model, replay_buffer, optimizer, batch_size, gamma)

    data.append([episode, total_reward, len(snake)])

    if episode % 10 == 0:
        save_to_excel(data)

    return total_reward, len(snake)


def train_ai():
    model = DQNetwork(input_size=8, output_size=4).to(device)
    target_model = DQNetwork(input_size=8, output_size=4).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    replay_buffer = deque(maxlen=replay_buffer_size)
    epsilon = 1.0
    epsilon_min = 10 ** -6  # Valor final muy cercano a 0
    epsilon_decay = 0.9995395890030878  # Ajustado para 15,000 episodios
    episodes = 300000  # Asegúrate de que el total de episodios sea al menos este valor
    data = []

    for episode in range(episodes):
        total_reward, length = game_loop(model, target_model, optimizer, replay_buffer, epsilon, episode, data)

        # Actualizar epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episodio: {episode}, Recompensa: {total_reward}, Longitud: {length}")

        if episode % 100 == 0:
            save_model(model)

    save_model(model)


train_ai()
