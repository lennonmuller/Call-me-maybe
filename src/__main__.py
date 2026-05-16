"""
Ponto de entrada principal para o projeto de Function Calling.
"""

import argparse
import sys
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """Configura e processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="LLM Function Calling com Constrained Decoding."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Caminho para o arquivo JSON contendo as definições das funções."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Caminho para o arquivo JSON contendo os prompts de teste."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Caminho onde o JSON de saída será salvo."
    )
    return parser.parse_args()


def main() -> None:
    """Função principal da aplicação."""
    args = parse_arguments()

    # Criar diretório de output, caso não exista, para evitar erros
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Iniciando pipeline de Function Calling...")
    print(f"Definições: {args.functions_definition}")
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")

    # TODO: Inicializar o modelo e iniciar o processamento
    # Isso será feito na Fase 2, após desenharmos o decodificador.
    print("Sucesso! Setup inicial validado.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erro inesperado: {e}", file=sys.stderr)
        sys.exit(1)
