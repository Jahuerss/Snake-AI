import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
import pandas as pd

# Paràmetres de l'entorn i el joc
tamany_pantalla = 400
tamany_bloc = 20
direccions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Paràmetres de l'aprenentatge per reforç
gamma = 0.95  # Augmentar la influència de recompenses futures
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
taxa_aprenentatge = 0.0005  # Reduir per a un aprenentatge més estable
tamany_lot = 64  # Augmentar la mida del lot per a una millor generalització
mida_buffer_reposicio = 5000  # Augmentar la mida del buffer per a més diversitat

# Dispositiu per entrenar
dispositiu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class XarxaDQ(nn.Module):
    def __init__(self, mida_entrada, mida_sortida):
        super(XarxaDQ, self).__init__()
        self.fc1 = nn.Linear(mida_entrada, 128)  # Augmentar el nombre de neurones
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, mida_sortida)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def obtenir_estat(cua, menjar, direccio, mida_pantalla):
    cap_x, cap_y = cua[-1]
    menjar_x, menjar_y = menjar

    estat = [
        (menjar_x - cap_x) / mida_pantalla,
        (menjar_y - cap_y) / mida_pantalla,
        direccio[0],
        direccio[1],
    ]

    for dx, dy in direccions:
        proper_x = cap_x + dx * tamany_bloc
        proper_y = cap_y + dy * tamany_bloc
        if proper_x < 0 or proper_x >= tamany_pantalla or proper_y < 0 or proper_y >= tamany_pantalla:
            estat.append(1)
        elif [proper_x, proper_y] in cua[:-1]:
            estat.append(1)
        else:
            estat.append(0)

    return np.array(estat, dtype=np.float32)

def triar_accio(estat, model, epsilon):
    if np.random.rand() < epsilon:
        return random.randint(0, 3)
    estat_tensor = torch.tensor(estat, dtype=torch.float32).unsqueeze(0).to(dispositiu)
    with torch.no_grad():
        valors_q = model(estat_tensor)
    return torch.argmax(valors_q).item()

def entrenar_model(model, model_objetiu, buffer_reposicio, optimitzador, tamany_lot, gamma):
    if len(buffer_reposicio) < tamany_lot:
        return

    minibatch = random.sample(buffer_reposicio, tamany_lot)
    estats, accions, recompenses, següents_estats, fets = zip(*minibatch)

    estats = torch.tensor(np.array(estats), dtype=torch.float32).to(dispositiu)
    accions = torch.tensor(accions, dtype=torch.int64).unsqueeze(1).to(dispositiu)
    recompenses = torch.tensor(recompenses, dtype=torch.float32).to(dispositiu)
    següents_estats = torch.tensor(np.array(següents_estats), dtype=torch.float32).to(dispositiu)
    fets = torch.tensor(fets, dtype=torch.float32).to(dispositiu)

    valors_q = model(estats).gather(1, accions)
    següents_valors_q = model_objetiu(següents_estats).max(1)[0].detach()
    valors_q_objectiu = recompenses + (1 - fets) * gamma * següents_valors_q

    pèrdua = nn.MSELoss()(valors_q.squeeze(), valors_q_objectiu)

    optimitzador.zero_grad()
    pèrdua.backward()
    optimitzador.step()

def desar_model(model, nom="snake_dqn.pth"):
    torch.save(model.state_dict(), nom)

def carregar_model(model, nom="snake_dqn.pth"):
    if os.path.exists(nom):
        model.load_state_dict(torch.load(nom))

def desar_a_excel(dades, nom="training_data.xlsx"):
    df = pd.DataFrame(dades, columns=["Episodi", "Recompensa", "Longitud"])
    if os.path.exists(nom):
        df_existent = pd.read_excel(nom)
        df = pd.concat([df_existent, df], ignore_index=True).drop_duplicates()
    df.to_excel(nom, index=False)

def generar_menjar(cua, mida_pantalla, tamany_bloc):
    while True:
        menjar_x = random.randint(0, mida_pantalla - 1) * tamany_bloc
        menjar_y = random.randint(0, mida_pantalla - 1) * tamany_bloc
        menjar = [menjar_x, menjar_y]
        # Verifica que el menjar no estigui dins de la cua
        if menjar not in cua:
            # Verifica que el menjar no estigui massa a prop del cap
            cap_x, cap_y = cua[-1]
            distancia_cap = abs(menjar_x - cap_x) + abs(menjar_y - cap_y)
            if distancia_cap >= tamany_bloc:  # Canvia segons la distància mínima desitjada
                return menjar

def bucle_joc(model, model_objetiu, optimitzador, buffer_reposicio, epsilon, episodi, dades):
    cua = [[tamany_pantalla // 2, tamany_pantalla // 2]]
    direccio = direccions[0]
    menjar = generar_menjar(cua, tamany_pantalla // tamany_bloc, tamany_bloc)

    recompensa_total = 0
    fet = False

    while not fet:
        estat = obtenir_estat(cua, menjar, direccio, mida_pantalla=tamany_pantalla // tamany_bloc)
        accio = triar_accio(estat, model, epsilon)

        direccio = direccions[accio]
        cap_x, cap_y = cua[-1]
        nou_cap = [cap_x + direccio[0] * tamany_bloc, cap_y + direccio[1] * tamany_bloc]
        cua.append(nou_cap)

        if (nou_cap in cua[:-1]) or (nou_cap[0] < 0 or nou_cap[0] >= tamany_pantalla or
                                      nou_cap[1] < 0 or nou_cap[1] >= tamany_pantalla):
            fet = True
            recompensa = -50  # Reduir la penalització per col·lisió per evitar un aprenentatge extrem
        elif nou_cap == menjar:
            recompensa = 100
            menjar = [random.randint(0, (tamany_pantalla // tamany_bloc) - 1) * tamany_bloc,
                      random.randint(0, (tamany_pantalla // tamany_bloc) - 1) * tamany_bloc]
        else:
            recompensa = -1
            cua.pop(0)

        recompensa_total += recompensa
        següent_estat = obtenir_estat(cua, menjar, direccio, mida_pantalla=tamany_pantalla // tamany_bloc)

        buffer_reposicio.append((estat, accio, recompensa, següent_estat, fet))
        entrenar_model(model, model_objetiu, buffer_reposicio, optimitzador, tamany_lot, gamma)

    dades.append([episodi, recompensa_total, len(cua)])

    if episodi % 10 == 0:
        desar_a_excel(dades)

    return recompensa_total, len(cua)


def entrenar_ai():
    model = XarxaDQ(mida_entrada=8, mida_sortida=4).to(dispositiu)
    model_objetiu = XarxaDQ(mida_entrada=8, mida_sortida=4).to(dispositiu)
    model_objetiu.load_state_dict(model.state_dict())
    optimitzador = optim.Adam(model.parameters(), lr=taxa_aprenentatge)

    buffer_reposicio = deque(maxlen=mida_buffer_reposicio)
    epsilon = 1.0
    epsilon_min = 10 ** -6  # Valor final molt proper a 0
    epsilon_decay = 0.9995395890030878  # Ajustat per a 10.000 episodis
    episodis = 300000  # Assegura't que el total d'episodis sigui com a mínim aquest valor
    dades = []

    for episodi in range(episodis):
        recompensa_total, longitud = bucle_joc(model, model_objetiu, optimitzador, buffer_reposicio, epsilon, episodi, dades)

        # Actualitzar epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episodi % 10 == 0:
            model_objetiu.load_state_dict(model.state_dict())

        print(f"Episodi: {episodi}, Recompensa: {recompensa_total}, Longitud: {longitud}")

        if episodi % 100 == 0:
            desar_model(model)

    desar_model(model)

entrenar_ai()
