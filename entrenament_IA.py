import tkinter as tk
from tkinter import scrolledtext
import threading
import sys
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
import pandas as pd

# Paràmetres de l'entorn i el joc
mida_pantalla = 400  # Mida de la pantalla del joc
mida_bloc = 20  # Mida de cada bloc
direccions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Direccions possibles del moviment

# Paràmetres d'aprenentatge per reforç
gamma = 0.95  # Factor de descompte
epsilon = 1.0  # Taxa d'exploració inicial
epsilon_min = 0.01  # Valor mínim de epsilon
epsilon_decay = 0.995  # Decadència d'epsilon
taxa_aprenentatge = 0.0005  # Taxa d'aprenentatge
mida_lot = 64  # Mida del lot per a l'entrenament
mida_buffer_reposicio = 5000  # Mida màxima del buffer de reposició

# Configuració del dispositiu (GPU si està disponible, sinó CPU)
dispositiu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classe per a la xarxa neuronal Deep Q
class XarxaDQ(nn.Module):
    def __init__(self, mida_entrada, mida_sortida):
        super(XarxaDQ, self).__init__()
        self.fc1 = nn.Linear(mida_entrada, 128)  # Primera capa completament connectada
        self.fc2 = nn.Linear(128, 128)  # Segona capa completament connectada
        self.fc3 = nn.Linear(128, mida_sortida)  # Capa final que dóna la sortida

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Funció d'activació ReLU per a la primera capa
        x = torch.relu(self.fc2(x))  # Funció d'activació ReLU per a la segona capa
        x = self.fc3(x)  # Passar a la capa final
        return x

# Funció per obtenir l'estat del joc (posició de la serp, menjar i obstacles)
def obtenir_estat(cua, menjar, direccio, mida_pantalla):
    cap_x, cap_y = cua[-1]
    menjar_x, menjar_y = menjar

    estat = [
        (menjar_x - cap_x) / mida_pantalla,  # Diferència de posició entre la serp i el menjar en l'eix X
        (menjar_y - cap_y) / mida_pantalla,  # Diferència de posició entre la serp i el menjar en l'eix Y
        direccio[0],  # Direcció actual de la serp en l'eix X
        direccio[1],  # Direcció actual de la serp en l'eix Y
    ]

    # Comprovem si hi ha obstacles (mur o cos de la serp)
    for dx, dy in direccions:
        proper_x = cap_x + dx * mida_bloc
        proper_y = cap_y + dy * mida_bloc
        if proper_x < 0 or proper_x >= mida_pantalla or proper_y < 0 or proper_y >= mida_pantalla:
            estat.append(1)  # Mur
        elif [proper_x, proper_y] in cua[:-1]:
            estat.append(1)  # Cos de la serp
        else:
            estat.append(0)  # Espai buit

    return np.array(estat, dtype=np.float32)  # Retornem l'estat com un array NumPy

# Funció per triar una acció a partir de l'estat
def triar_accio(estat, model, epsilon):
    if np.random.rand() < epsilon:
        return random.randint(0, 3)  # Exploració: triar una acció aleatòria
    estat_tensor = torch.tensor(estat, dtype=torch.float32).unsqueeze(0).to(dispositiu)
    with torch.no_grad():
        valors_q = model(estat_tensor)  # Predicció de la xarxa neuronal
    return torch.argmax(valors_q).item()  # Escollir l'acció amb el valor Q més alt

# Funció per entrenar el model
def entrenar_model(model, model_objetiu, buffer_reposicio, optimitzador, mida_lot, gamma):
    if len(buffer_reposicio) < mida_lot:  # Si no tenim prou experiència per entrenar, sortim
        return

    minibatch = random.sample(buffer_reposicio, mida_lot)  # Agafem una mostra aleatòria de les experiències
    estats, accions, recompenses, següents_estats, fets = zip(*minibatch)

    # Convertim les experiències en tensors per a l'entrenament
    estats = torch.tensor(np.array(estats), dtype=torch.float32).to(dispositiu)
    accions = torch.tensor(accions, dtype=torch.int64).unsqueeze(1).to(dispositiu)
    recompenses = torch.tensor(recompenses, dtype=torch.float32).to(dispositiu)
    següents_estats = torch.tensor(np.array(següents_estats), dtype=torch.float32).to(dispositiu)
    fets = torch.tensor(fets, dtype=torch.float32).to(dispositiu)

    # Calculem el valor Q actual i l'objectiu Q
    valors_q = model(estats).gather(1, accions)
    següents_valors_q = model_objetiu(següents_estats).max(1)[0].detach()  # No es modifica el model objectiu
    valors_q_objectiu = recompenses + (1 - fets) * gamma * següents_valors_q

    # Funció de pèrdua (error quadràtic mitjà)
    pèrdua = nn.MSELoss()(valors_q.squeeze(), valors_q_objectiu)

    # Actualitzem els pesos del model
    optimitzador.zero_grad()
    pèrdua.backward()
    optimitzador.step()

# Funció per desar el model entrenat
def desar_model(model, nom="snake_dqn.pth"):
    torch.save(model.state_dict(), nom)

# Funció per carregar un model preentrenat
def carregar_model(model, nom="snake_dqn.pth"):
    if os.path.exists(nom):
        model.load_state_dict(torch.load(nom))

# Funció per desar les dades d'entrenament en un fitxer Excel
def desar_a_excel(dades, nom="training_data.xlsx"):
    df = pd.DataFrame(dades, columns=["Episodi", "Recompensa", "Longitud"])
    if os.path.exists(nom):
        df_existent = pd.read_excel(nom)
        df = pd.concat([df_existent, df], ignore_index=True).drop_duplicates()  # Evitem duplicats
    df.to_excel(nom, index=False)

# Funció per generar una posició aleatòria del menjar
def generar_menjar(cua, mida_pantalla, mida_bloc):
    while True:
        menjar_x = random.randint(0, mida_pantalla - 1) * mida_bloc
        menjar_y = random.randint(0, mida_pantalla - 1) * mida_bloc
        menjar = [menjar_x, menjar_y]
        if menjar not in cua:  # Assegurem-nos que el menjar no aparegui dins la cua de la serp
            cap_x, cap_y = cua[-1]
            distancia_cap = abs(menjar_x - cap_x) + abs(menjar_y - cap_y)
            if distancia_cap >= mida_bloc:  # Comprovem que el menjar no estigui massa a prop de la serp
                return menjar

# Funció per gestionar el bucle principal del joc
def bucle_joc(model, model_objetiu, optimitzador, buffer_reposicio, epsilon, episodi, dades):
    cua = [[mida_pantalla // 2, mida_pantalla // 2]]  # Iniciem la serp al mig de la pantalla
    direccio = direccions[0]  # Direcció inicial (cap a la dreta)
    menjar = generar_menjar(cua, mida_pantalla // mida_bloc, mida_bloc)  # Generem el menjar inicial

    recompensa_total = 0
    fet = False  # Variable per controlar si la partida ha acabat

    while not fet:
        estat = obtenir_estat(cua, menjar, direccio, mida_pantalla=mida_pantalla // mida_bloc)  # Obtenim l'estat
        accio = triar_accio(estat, model, epsilon)  # Triem l'acció amb base en l'estat

        direccio = direccions[accio]  # Actualitzem la direcció segons l'acció
        cap_x, cap_y = cua[-1]
        nou_cap = [cap_x + direccio[0] * mida_bloc, cap_y + direccio[1] * mida_bloc]  # Movem la serp
        cua.append(nou_cap)

        # Comprovem si la serp ha col·lidit amb alguna cosa (mur o cos propi)
        if (nou_cap in cua[:-1]) or (nou_cap[0] < 0 or nou_cap[0] >= mida_pantalla or
                                      nou_cap[1] < 0 or nou_cap[1] >= mida_pantalla):
            fet = True
            recompensa = -50  # Penalització per perdre
        elif nou_cap == menjar:  # Si la serp menja el menjar
            recompensa = 100  # Recompensa per menjar
            menjar = generar_menjar(cua, mida_pantalla // mida_bloc, mida_bloc)  # Generem un nou menjar
        else:  # Si la serp es mou sense menjar
            recompensa = -1  # Petita penalització per cada moviment
            cua.pop(0)  # Eliminar el darrer segment de la serp

        recompensa_total += recompensa
        següent_estat = obtenir_estat(cua, menjar, direccio, mida_pantalla=mida_pantalla // mida_bloc)

        # Afegim l'experiència al buffer de reposició
        buffer_reposicio.append((estat, accio, recompensa, següent_estat, fet))
        entrenar_model(model, model_objetiu, buffer_reposicio, optimitzador, mida_lot, gamma)

    dades.append([episodi, recompensa_total, len(cua)])  # Desem les dades de l'episodi

    # Desem les dades cada cert nombre d'episodis
    if episodi % 10 == 0:
        desar_a_excel(dades)

    return recompensa_total, len(cua)

# Funció per entrenar la IA
def entrenar_ai():
    model = XarxaDQ(mida_entrada=8, mida_sortida=4).to(dispositiu)  # Creem el model
    model_objetiu = XarxaDQ(mida_entrada=8, mida_sortida=4).to(dispositiu)  # Creem el model objectiu
    model_objetiu.load_state_dict(model.state_dict())  # Inicialitzem el model objectiu amb el mateix pes
    optimitzador = optim.Adam(model.parameters(), lr=taxa_aprenentatge)  # Optimitzador Adam

    buffer_reposicio = deque(maxlen=mida_buffer_reposicio)  # Buffer de reposició per guardar les experiències
    epsilon = 1.0  # Valor inicial de epsilon
    epsilon_min = 10 ** -6  # Valor mínim de epsilon
    epsilon_decay = 0.9995395890030878  # Decadència d'epsilon
    episodis = 300000  # Nombre total d'episodis per entrenar
    dades = []  # Llista per guardar les dades d'entrenament

    for episodi in range(episodis):
        recompensa_total, longitud = bucle_joc(model, model_objetiu, optimitzador, buffer_reposicio, epsilon, episodi, dades)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Actualitzem epsilon

        if episodi % 10 == 0:
            model_objetiu.load_state_dict(model.state_dict())  # Actualitzem el model objectiu

        print(f"Episodi: {episodi}, Recompensa: {recompensa_total}, Longitud: {longitud}")

        if episodi % 100 == 0:
            desar_model(model)  # Desem el model cada 100 episodis

    desar_model(model)  # Desem el model final

# Classe per redirigir la sortida per la consola
class CustomConsole:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)  # Inserim el missatge a la consola
        self.text_widget.see(tk.END)  # Ens assegurem que es vegi el missatge

    def flush(self):
        pass  # No fem res quan es buida el buffer de sortida

# Funció per iniciar l'entrenament en un fil separat per la interfície gràfica
def entrenar_ai_interficie():
    def entrenar_thread():
        entrenar_ai()

    threading.Thread(target=entrenar_thread, daemon=True).start()  # Iniciem el fil

# Funció per crear la interfície gràfica amb tkinter
def crear_interficie():
    ventana = tk.Tk()  # Crear la ventana principal
    ventana.title("Entrenament de IA")  # Título de la ventana
    ventana.geometry("800x600")  # Tamaño de la ventana

    # Cargar la imagen como logo
    logo = tk.PhotoImage(file="assets/images/logo_1.png")
    ventana.iconphoto(False, logo)  # Establecer la imagen como ícono de la ventana

    # Consola deslizante
    consola = scrolledtext.ScrolledText(ventana, wrap=tk.WORD, height=30, width=100)
    consola.pack(pady=10)

    sys.stdout = CustomConsole(consola)  # Redirigir la salida del sistema a la consola

    # Botón para terminar el entrenamiento
    boton_terminar = tk.Button(ventana, text="Terminar Entrenament", command=ventana.destroy)
    boton_terminar.pack(pady=10)

    entrenar_ai_interficie()  # Iniciar el entrenamiento en un hilo separado

    ventana.mainloop()  # Iniciar el bucle de la interfaz gráfica

crear_interficie()  # Creem i mostrem la interfície gràfica
