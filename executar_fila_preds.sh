#!/bin/bash

# Pasta de destino dos arquivos de saída
pasta_saida="txt_outputs"

# Verifica se a pasta de saída existe, caso contrário, cria-a
if [ ! -d "$pasta_saida" ]; then
    mkdir "$pasta_saida"
fi

# Lista de arquivos a serem executados
arquivos=(
    "src_apolar/get_test_preds_apolar.py"
    "src_polar/get_test_preds_polar.py"
    "src_polar_apolar/get_test_preds_polar_apolar.py"
)

# Salva o diretório atual
diretorio_atual=$(pwd)

# Loop para processar cada arquivo da fila
for arquivo in "${arquivos[@]}"; do
    nome_arquivo_saida="${arquivo%.*}_output.txt"  # Gera o nome do arquivo de saída
    caminho_arquivo_saida="$diretorio_atual/$pasta_saida/$nome_arquivo_saida"  # Caminho completo do arquivo de saída
    echo "Arquivo $arquivo adicionado à fila. Saída em $caminho_arquivo_saida"
    
    # Altera o diretório atual para o diretório do arquivo
    cd "$(dirname "$arquivo")"
    
    # Executa o script Python
    nohup poetry run python3 "$(basename "$arquivo")" > "$caminho_arquivo_saida"
    
    # Volta ao diretório original
    cd "$diretorio_atual"
    
    wait  # Aguarda a conclusão da execução atual antes de prosseguir
done

echo "Todos os arquivos foram executados."

