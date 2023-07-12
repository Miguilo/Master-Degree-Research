sta de arquivos a serem executados
arquivos=(
    "eval_apolar_models.py"
    "eval_polar_models.py"
    "eval_polar_apolar_models.py"
    "get_opt_models_apolar.py"
    "get_opt_models_polar.py"
    "get_opt_models_polar_apolar.py"
)

# Loop para processar cada arquivo da fila
for arquivo in "${arquivos[@]}"; do
    nome_arquivo_saida="$txt_outputs/{arquivo%.*}_output.txt"  # Gera o nome do arquivo de saída
    nohup poetry run python3 "$arquivo" > "$nome_arquivo_saida" &
    echo "Arquivo $arquivo adicionado à fila. Saída em $nome_arquivo_saida"
    wait  # Pausa de 1 segundo entre as execuções
done

echo "Todos os arquivos foram adicionados à fila."

