"""Avalia as hiperparametrizações U, E e O do CART e salva os números das Tabelas 2 e 3.

Cada hiperparametrização do DecisionTreeRegressor passa pela validação cruzada do módulo comum
(k=5, semente 42); o melhor CART sai do critério do item 9 de observacoes.md (escolher_melhor).
Nunca lê o teste cego.

Uso: uv run python src/treinar_cart.py
"""

import json
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor

from validacao_cruzada import (
    K_FOLDS,
    avaliar,
    carregar_dataset,
    conferir_ueo,
    descrever,
    escolher_melhor,
    formatar,
    montar,
)

RAIZ = Path(__file__).resolve().parent.parent
TABELAS_2_3 = RAIZ / "resultados" / "tabelas2_3_cart.json"

CART = DecisionTreeRegressor()

# Hiperparametrizações U, E e O (Tabela 2), item 11 de observacoes.md. As linhas obrigatórias da
# Tabela 2 ficam explícitas mesmo quando o valor é o padrão do scikit-learn; random_state é a
# linha <outras>. max_depth=None (sem limite) vai como null no JSON; o relatório o exibe "None".
HIPERPARAMETRIZACOES = {
    "U": {"min_samples_leaf": 0.25, "max_depth": 2, "random_state": 42},
    "E": {"min_samples_leaf": 20, "max_depth": 10, "random_state": 42},
    "O": {"min_samples_leaf": 1, "max_depth": None, "random_state": 42},
}

LINHAS_TABELA_2 = {
    "min_samples_leaf": "Min_samples_leaf",
    "max_depth": "Max_depth",
    "random_state": "Random_state",
}
LINHAS_TABELA_3 = {
    "treino": "TREINO MSE (dpa)",
    "validacao": "VALIDAÇÃO MSE (dpa)",
    "diferenca": "MÉDIA DAS DIFS. (dpa)",
}


def imprimir_tabela(titulo, linhas):
    """Tabela em texto: `linhas` mapeia o nome da linha -> células de U, E e O."""
    print(f"\n{titulo}")
    print(f"{'MODELO':<24}" + "".join(f"{rotulo:<24}" for rotulo in HIPERPARAMETRIZACOES))
    for nome, celulas in linhas.items():
        print(f"{nome:<24}" + "".join(f"{str(c):<24}" for c in celulas))


def main():
    X, y = carregar_dataset()
    resultados = {}
    for rotulo, hiperparametrizacao in HIPERPARAMETRIZACOES.items():
        resultados[rotulo], _ = avaliar(CART, hiperparametrizacao, X, y)
        print(f"\n{rotulo}: {montar(CART, hiperparametrizacao)}")
        print(descrever(resultados[rotulo]))
    melhor, criterio = escolher_melhor(resultados)
    conferir_ueo(resultados, melhor)

    saida = {
        "tabela_2": HIPERPARAMETRIZACOES,
        "tabela_3": resultados,
        "melhor_cart": melhor,
        "criterio_melhor": criterio,
        "k_folds": K_FOLDS,
    }
    TABELAS_2_3.parent.mkdir(exist_ok=True)
    TABELAS_2_3.write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    imprimir_tabela(
        "Tabela 2: hiperparametrizações CART",
        {nome: [h[chave] for h in HIPERPARAMETRIZACOES.values()] for chave, nome in LINHAS_TABELA_2.items()},
    )
    imprimir_tabela(
        "Tabela 3: resultados dos modelos CART",
        {nome: [formatar(r[chave]) for r in resultados.values()] for chave, nome in LINHAS_TABELA_3.items()},
    )
    print(f"\nEmpate técnico (até {criterio['limite_empate']:.6f}): {criterio['empatados']}")
    print(f"Melhor CART: {melhor}")
    print(f"Tabelas 2 e 3 salvas em: {TABELAS_2_3.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
