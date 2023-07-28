#!/bin/bash

# Pasta de destino dos arquivos de saída
pasta_saida="txt_outputs"

# Lista de arquivos a serem executados
arquivos=(
    "src_apolar/get_opt_models_apolar.py"
    "src_polar/get_opt_models_polar.py"
    "src_polar_apolar/get_opt_models_polar_apolar.py"
    "src_apolar/eval_apolar_models.py"
    "src_polar/eval_polar_models.py"
    "src_polar_apolar/eval_polar_apolar_models.py"
)

# Verifica se a pasta de saída existe, caso contrário, cria-a
nome_arquivos=(
    "get_opt_models_apolar.py"
    "get_opt_models_polar.py"
    "get_opt_models_polar_apolar.py"
    "eval_apolar_models.py"
    "eval_polar_models.py"
    "eval_polar_apolar_models.py"
)

if [ ! -d "$pasta_saida" ]; then
    mkdir "$pasta_saida"
fi
# Verifica se a pasta de saída existe, caso contrário, cria-a

# Loop para processar cada arquivo da fila
for ((i = 0; i < ${#arquivos[@]}; i++)); do
    nome_arquivo_saida="${nome_arquivos[i]%.*}_output.txt"  # Gera o nome do arquivo de saída
    caminho_arquivo_saida="$pasta_saida/$nome_arquivo_saida"  # Caminho completo do arquivo de saída
    echo "Arquivo ${nome_arquivos[i]} adicionado à fila. Saída em $caminho_arquivo_saida"
    nohup poetry run python3 "${arquivos[i]}" > "$caminho_arquivo_saida"
    wait  # Aguarda a conclusão da execução atual antes de prosseguir

done

echo "Todos os arquivos foram executados."
