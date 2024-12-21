epsilon_inicial = 1.0
epsilon_min = 0.01
episodios_hasta_min = 100000

epsilon_decay = (epsilon_min / epsilon_inicial) ** (1 / episodios_hasta_min)
print(epsilon_decay)