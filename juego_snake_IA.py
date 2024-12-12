import pygame
import torch
import numpy as np
import random
import time
from collections import deque
import torch.nn as nn  # Aseguramos esta importación
import tkinter as tk
from tkinter import filedialog


# Parámetros del entorno y juego
screen_size = 400
block_size = 20
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Parámetros de colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def select_model_file():
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal
    model_path = filedialog.askopenfilename(
        title="Select Model File",
        filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")]
    )
    if not model_path:
        print("No file selected. Exiting...")
        exit()  # Salir si no selecciona un archivo
    return model_path

# Cargar el modelo seleccionado
def load_selected_model(model, filename):
    model.load_state_dict(torch.load(filename))
    print(f"Model loaded from: {filename}")

# Función para jugar con el modelo
def play_game_with_ai(model):
    # Tu función play_game_with_ai aquí
    pass

# Función principal
def main():
    pygame.init()

    # Seleccionar archivo del modelo mediante ventana emergente
    model_path = select_model_file()

    # Crear y cargar modelo
    model = DQNetwork(input_size=8, output_size=4).to(device)
    load_selected_model(model, model_path)

    # Configurar el modelo para evaluación
    model.eval()

    # Jugar con el modelo cargado
    play_game_with_ai(model)
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


def choose_action(state, model):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
    return torch.argmax(q_values).item()


def game_over_screen(screen, font, score, time_elapsed):
    # Fondo de pantalla de "Game Over"
    screen.fill((50, 50, 50))  # Gris oscuro

    # Mensajes de texto
    game_over_text = font.render("GAME OVER", True, RED)
    score_text = font.render(f"Score: {score}", True, WHITE)
    time_text = font.render(f"Time: {int(time_elapsed)} seconds", True, WHITE)
    retry_text = font.render("Press R to Retry or Q to Quit", True, WHITE)

    # Posiciones centradas
    screen_width, screen_height = screen.get_size()
    center_x = screen_width // 2
    center_y = screen_height // 2

    # Dibujar los textos centrados
    screen.blit(game_over_text, (center_x - game_over_text.get_width() // 2, center_y - 100))
    screen.blit(score_text, (center_x - score_text.get_width() // 2, center_y - 50))
    screen.blit(time_text, (center_x - time_text.get_width() // 2, center_y))
    screen.blit(retry_text, (center_x - retry_text.get_width() // 2, center_y + 50))

    # Actualizar pantalla
    pygame.display.flip()

    # Espera 2 segundos antes de continuar
    pygame.time.wait(2000)



def play_game_with_ai(model):
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size + 50))
    pygame.display.set_caption("Snake AI")
    clock = pygame.time.Clock()  # Controlador de tiempo
    font = pygame.font.SysFont(None, 36)

    while True:
        snake = [[screen_size // 2, screen_size // 2]]
        direction = directions[0]
        food = [random.randint(0, (screen_size // block_size) - 1) * block_size,
                random.randint(0, (screen_size // block_size) - 1) * block_size]
        score = 0
        start_time = time.time()
        running = True

        while running:
            screen.fill(BLACK)
            for x in range(0, screen_size, block_size):
                for y in range(0, screen_size, block_size):
                    pygame.draw.rect(screen, GRAY if (x + y) // block_size % 2 == 0 else WHITE,
                                     (x, y, block_size, block_size))

            # Barra inferior
            pygame.draw.rect(screen, WHITE, (0, screen_size, screen_size, 50))
            score_text = font.render(f"Score: {score}", True, BLACK)
            time_text = font.render(f"Time: {int(time.time() - start_time)}s", True, BLACK)
            screen.blit(score_text, (10, screen_size + 10))
            screen.blit(time_text, (200, screen_size + 10))

            # Dibujar comida
            pygame.draw.rect(screen, RED, (food[0], food[1], block_size, block_size))

            # Dibujar serpiente
            for i, segment in enumerate(snake):
                color = YELLOW if i == len(snake) - 1 else GREEN
                pygame.draw.rect(screen, color, (segment[0], segment[1], block_size, block_size))

                # Dibujar dirección en la cabeza
                if i == len(snake) - 1:
                    head_x, head_y = segment
                    dx, dy = direction
                    pygame.draw.line(screen, BLACK,
                                     (head_x + block_size // 2, head_y + block_size // 2),
                                     (head_x + block_size // 2 + dx * block_size // 2,
                                      head_y + block_size // 2 + dy * block_size // 2), 3)

            pygame.display.flip()

            # Estado actual
            state = get_state(snake, food, direction, board_size=screen_size // block_size)
            action = choose_action(state, model)

            # Movimiento
            direction = directions[action]
            head_x, head_y = snake[-1]
            new_head = [head_x + direction[0] * block_size, head_y + direction[1] * block_size]
            snake.append(new_head)

            if (new_head in snake[:-1]) or (new_head[0] < 0 or new_head[0] >= screen_size or
                                            new_head[1] < 0 or new_head[1] >= screen_size):
                game_over_screen(screen, font, score, time.time() - start_time)
                running = False

            elif new_head == food:
                score += 1
                food = [random.randint(0, (screen_size // block_size) - 1) * block_size,
                        random.randint(0, (screen_size // block_size) - 1) * block_size]
            else:
                snake.pop(0)

            # Eventos de Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            clock.tick(10)  # Ajusta los FPS a 20

        # Pantalla de "Game Over"
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        waiting = False
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        return



def main():
    model = DQNetwork(input_size=8, output_size=4).to(device)
    model.load_state_dict(torch.load("snake_dqn.pth", map_location=device))
    model.eval()
    play_game_with_ai(model)


if __name__ == "__main__":
    main()
