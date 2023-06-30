from datetime import datetime

# Obter o horário de início
inicio = datetime.now()

# Simular uma operação de longa duração
for _ in range(10000000):
    pass

# Obter o horário de término
fim = datetime.now()

# Calcular o tempo de execução total
tempo_execucao = fim - inicio

# Exibir o tempo de execução total
print("Tempo 1:", inicio)
print("Tempo 2:", fim)
print("Tempo de execução:", tempo_execucao)