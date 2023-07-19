#!/bin/bash

# Pasta de destino dos arquivos de saída
pasta_saida="txt_outputs"

# Lista de arquivos a serem executados
arquivos=(
    "src_polar/get_opt_models_polar.py"
    "src_polar_apolar/get_opt_models_polar_apolar.py"
    "src_apolar/eval_apolar_models.py"
    "src_polar/eval_polar_models.py"
    "src_polar_apolar/eval_polar_apolar_models.py"
)

# Verifica se a pasta de saída existe, caso contrário, cria-a
if [ ! -d "$pasta_saida" ]; then
    mkdir "$pasta_saida"
fi

# Loop para processar cada arquivo da fila
for arquivo in "${arquivos[@]}"; do
    nome_arquivo_saida="${arquivo%.*}_output.txt"  # Gera o nome do arquivo de saída
    caminho_arquivo_saida="$pasta_saida/$nome_arquivo_saida"  # Caminho completo do arquivo de saída
    echo "Arquivo $arquivo adicionado à fila. Saída em $caminho_arquivo_saida"
    nohup poetry run python3 "$arquivo" > "$caminho_arquivo_saida"
    wait  # Aguarda a conclusão da execução atual antes de prosseguir
done

echo "Todos os arquivos foram executados."