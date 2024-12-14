import pygame
import torch
import numpy as np
import random
import time
from collections import deque
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog


# Mida de la pantalla i el bloc
mida_pantalla = 400
mida_bloc = 20
# Direccions possibles de moviment
direccions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Colors definits
NEGRE = (0, 0, 0)
BLANC = (255, 255, 255)
VERD = (0, 255, 0)
VERMELL = (255, 0, 0)
GROC = (255, 255, 0)
GRIS = (200, 200, 200)

# Seleccionem el dispositiu per l'entrenament
dispositiu = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Xarxa neuronal per al model de Deep Q-Learning
class DQXarxa(nn.Module):
    def __init__(self, mida_dentrada, mida_sortida):
        super(DQXarxa, self).__init__()
        self.fc1 = nn.Linear(mida_dentrada, 128)  # Primera capa
        self.fc2 = nn.Linear(128, 128)  # Segona capa
        self.fc3 = nn.Linear(128, mida_sortida)  # Capa de sortida

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Funció d'activació ReLU per a la primera capa
        x = torch.relu(self.fc2(x))  # Funció d'activació ReLU per a la segona capa
        x = self.fc3(x)  # Capa de sortida
        return x


# Funció per seleccionar un arxiu de model
def seleccionar_arxiu_model():
    root = tk.Tk()
    root.withdraw()  # Amaga la finestra principal
    cami_model = filedialog.askopenfilename(
        title="Selecciona l'arxiu del model",
        filetypes=[("Arxius de Model", "*.pth"), ("Tots els Arxius", "*.*")]
    )
    if not cami_model:
        print("No s'ha seleccionat cap arxiu. Sortint...")
        exit()  # Sortir si no es selecciona cap arxiu
    return cami_model


# Funció per carregar el model seleccionat des d'un arxiu
def carregar_model_seleccionat(model, nom_arxiu):
    model.load_state_dict(torch.load(nom_arxiu))  # Carrega els pesos del model
    print(f"Model carregat des de: {nom_arxiu}")


# Funció per jugar al joc amb l'AI (qualsevol lògica d'AI pot ser afegida aquí)
def jugar_joc_amb_ai(model):
    pass


# Funció principal que inicialitza el joc i el model
def principal():
    pygame.init()  # Inicialitza pygame

    # Seleccionem el fitxer del model mitjançant una finestra emergent
    cami_model = seleccionar_arxiu_model()

    # Creem el model i carreguem-lo
    model = DQXarxa(mida_dentrada=8, mida_sortida=4).to(dispositiu)
    carregar_model_seleccionat(model, cami_model)

    model.eval()  # Posem el model en mode d'avaluació

    # Comencem a jugar amb el model carregat
    jugar_joc_amb_ai(model)


# Funció per obtenir l'estat actual del joc
def obtenir_estat(serp, menjar, direccio, mida_pantalla):
    cap_x, cap_y = serp[-1]  # Posició de la capçalera de la serp
    menjar_x, menjar_y = menjar  # Posició del menjar

    # Creem un vector d'estat amb la distància del menjar i la direcció actual
    estat = [
        (menjar_x - cap_x) / mida_pantalla,
        (menjar_y - cap_y) / mida_pantalla,
        direccio[0],
        direccio[1],
    ]

    # Afegim l'estat de les possibles direccions al voltant de la serp
    for dx, dy in direccions:
        prox_x = cap_x + dx * mida_bloc
        prox_y = cap_y + dy * mida_bloc
        if prox_x < 0 or prox_x >= mida_pantalla or prox_y < 0 or prox_y >= mida_pantalla:
            estat.append(1)  # L'estat és perillós (fora de la pantalla)
        elif [prox_x, prox_y] in serp[:-1]:
            estat.append(1)  # L'estat és perillós (ocupat per la serp)
        else:
            estat.append(0)  # L'estat és segur

    return np.array(estat, dtype=np.float32)


# Funció per triar l'acció en base a l'estat utilitzant el model
def triar_accio(estat, model):
    estat_tensor = torch.tensor(estat, dtype=torch.float32).unsqueeze(0).to(dispositiu)
    with torch.no_grad():
        valors_q = model(estat_tensor)  # Obtenim els valors Q per a l'estat actual
    return torch.argmax(valors_q).item()  # Retornem l'acció amb el valor més alt


# Funció per mostrar la pantalla de "Game Over"
def pantalla_game_over(pantalla, font, puntuacio, temps_transcorregut):
    pantalla.fill((50, 50, 50))  # Fons de pantalla gris fosc

    # Mostrem els missatges de "Game Over" i les estadístiques
    text_game_over = font.render("GAME OVER", True, VERMELL)
    text_puntuacio = font.render(f"Puntuació: {puntuacio}", True, BLANC)
    text_temps = font.render(f"Temps: {int(temps_transcorregut)} segons", True, BLANC)
    text_reintentar = font.render("Prem R per tornar a provar o Q per sortir", True, BLANC)

    amplada_pantalla, altura_pantalla = pantalla.get_size()
    centre_x = amplada_pantalla // 2
    centre_y = altura_pantalla // 2

    # Centrem els textos a la pantalla
    pantalla.blit(text_game_over, (centre_x - text_game_over.get_width() // 2, centre_y - 100))
    pantalla.blit(text_puntuacio, (centre_x - text_puntuacio.get_width() // 2, centre_y - 50))
    pantalla.blit(text_temps, (centre_x - text_temps.get_width() // 2, centre_y))
    pantalla.blit(text_reintentar, (centre_x - text_reintentar.get_width() // 2, centre_y + 50))

    pygame.display.flip()  # Actualitzem la pantalla

    pygame.time.wait(2000)  # Esperem 2 segons abans de continuar


# Funció per jugar el joc amb l'AI
def jugar_joc_amb_ai(model):
    pygame.init()  # Inicialitzem pygame
    pantalla = pygame.display.set_mode((mida_pantalla, mida_pantalla + 50))  # Creem la finestra
    pygame.display.set_caption("Snake AI")  # Títol de la finestra
    rellotge = pygame.time.Clock()  # Controlador de temps
    font = pygame.font.SysFont(None, 36)  # Font per al text

    while True:
        serp = [[mida_pantalla // 2, mida_pantalla // 2]]  # Inicialitzem la serp
        direccio = direccions[0]  # Direcció inicial
        menjar = [random.randint(0, (mida_pantalla // mida_bloc) - 1) * mida_bloc,
                  random.randint(0, (mida_pantalla // mida_bloc) - 1) * mida_bloc]  # Posició inicial del menjar
        puntuacio = 0
        temps_inici = time.time()  # Temps de començament del joc
        executant = True

        while executant:
            pantalla.fill(NEGRE)  # Fons de la pantalla

            # Dibuixem la graella de la pantalla
            for x in range(0, mida_pantalla, mida_bloc):
                for y in range(0, mida_pantalla, mida_bloc):
                    pygame.draw.rect(pantalla, GRIS if (x + y) // mida_bloc % 2 == 0 else BLANC,
                                     (x, y, mida_bloc, mida_bloc))

            pygame.draw.rect(pantalla, BLANC, (0, mida_pantalla, mida_pantalla, 50))  # Barra inferior
            text_puntuacio = font.render(f"Puntuació: {puntuacio}", True, NEGRE)
            text_temps = font.render(f"Temps: {int(time.time() - temps_inici)}s", True, NEGRE)
            pantalla.blit(text_puntuacio, (10, mida_pantalla + 10))  # Dibuixem la puntuació
            pantalla.blit(text_temps, (200, mida_pantalla + 10))  # Dibuixem el temps

            # Dibuixem el menjar
            pygame.draw.rect(pantalla, VERMELL, (menjar[0], menjar[1], mida_bloc, mida_bloc))

            # Dibuixem la serp
            for i, segment in enumerate(serp):
                color = GROC if i == len(serp) - 1 else VERD
                pygame.draw.rect(pantalla, color, (segment[0], segment[1], mida_bloc, mida_bloc))

                # Dibuixem la direcció a la capçalera de la serp
                if i == len(serp) - 1:
                    cap_x, cap_y = segment
                    dx, dy = direccio
                    pygame.draw.line(pantalla, NEGRE,
                                     (cap_x + mida_bloc // 2, cap_y + mida_bloc // 2),
                                     (cap_x + mida_bloc // 2 + dx * mida_bloc // 2,
                                      cap_y + mida_bloc // 2 + dy * mida_bloc // 2), 3)

            pygame.display.flip()  # Actualitzem la pantalla

            # Obtenim l'estat actual i triem l'acció
            estat = obtenir_estat(serp, menjar, direccio, mida_pantalla=mida_pantalla // mida_bloc)
            accio = triar_accio(estat, model)

            # Movem la serp
            direccio = direccions[accio]
            cap_x, cap_y = serp[-1]
            nou_cap = [cap_x + direccio[0] * mida_bloc, cap_y + direccio[1] * mida_bloc]
            serp.append(nou_cap)

            # Comprovem si el joc ha acabat
            if (nou_cap in serp[:-1]) or (nou_cap[0] < 0 or nou_cap[0] >= mida_pantalla or
                                          nou_cap[1] < 0 or nou_cap[1] >= mida_pantalla):
                pantalla_game_over(pantalla, font, puntuacio, time.time() - temps_inici)
                executant = False

            elif nou_cap == menjar:  # Si la serp menja el menjar
                puntuacio += 1
                menjar = [random.randint(0, (mida_pantalla // mida_bloc) - 1) * mida_bloc,
                          random.randint(0, (mida_pantalla // mida_bloc) - 1) * mida_bloc]
            else:
                serp.pop(0)  # Movem la serp

            # Processar esdeveniments de pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            rellotge.tick(10)  # Establim els FPS

        # Pantalla d'espera després del "Game Over"
        esperant = True
        while esperant:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        esperant = False  # Reiniciar el joc
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        return


# Funció per executar el joc
def principal():
    model = DQXarxa(mida_dentrada=8, mida_sortida=4).to(dispositiu)  # Creem el model
    model.load_state_dict(torch.load("snake_dqn.pth", map_location=dispositiu))  # Carreguem el model entrenat
    model.eval()  # Posem el model en mode d'avaluació
    jugar_joc_amb_ai(model)  # Iniciem el joc


if __name__ == "__main__":
    principal()  # Iniciem el programa
