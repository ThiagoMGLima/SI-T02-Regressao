"""Avalia as hiperparametrizações U, E e O da RN e salva os números das Tabelas 4 e 5.

A RN é um Pipeline com StandardScaler e MLPRegressor (item 12 de observacoes.md), para que a
padronização seja ajustada só no fold de treino. Cada hiperparametrização passa pela validação
cruzada do módulo comum (k=5, semente 42), com um processo por fold; a melhor RN sai do critério
do item 9 de observacoes.md (escolher_melhor). Os folds que param por max_iter, e não pelo
critério de tol (os do ConvergenceWarning), são contados a partir das épocas de cada fold e vão
para o JSON junto com o tempo de cada avaliação.
Nunca lê o teste cego.

Uso: uv run python src/treinar_rn.py
"""

import json
import time
from pathlib import Path

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from validacao_cruzada import (
    K_FOLDS,
    avaliar,
    carregar_dataset,
    conferir_ueo,
    descrever,
    escolher_melhor,
    formatar,
)

RAIZ = Path(__file__).resolve().parent.parent
TABELAS_4_5 = RAIZ / "resultados" / "tabelas4_5_rn.json"

RN = Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor())])

# Fixos nas três hiperparametrizações (base do notebook de referência do professor, item 13).
BASE = {
    "solver": "sgd",
    "activation": "tanh",
    "momentum": 0.95,
    "learning_rate": "constant",
    "shuffle": True,
    "random_state": 42,
}

# O que varia entre U, E e O (Tabela 4), item 13 de observacoes.md.
HIPERPARAMETRIZACOES = {
    "U": {"hidden_layer_sizes": (2,), "learning_rate_init": 0.001, "alpha": 1.0, "batch_size": "auto",
          "max_iter": 200, "tol": 1e-4, "n_iter_no_change": 10},
    "E": {"hidden_layer_sizes": (64, 32), "learning_rate_init": 0.02, "alpha": 0.01, "batch_size": "auto",
          "max_iter": 2000, "tol": 1e-6, "n_iter_no_change": 20},
    "O": {"hidden_layer_sizes": (64,) * 10, "learning_rate_init": 0.05, "alpha": 0.0, "batch_size": 16,
          "max_iter": 400, "tol": 1e-4, "n_iter_no_change": 400},
}  # fmt: skip

# As quatro primeiras são as linhas do enunciado; as outras completam a hiperparametrização.
LINHAS_TABELA_4 = {
    "hidden_layer_sizes": "Topologia",
    "activation": "Função de ativação",
    "learning_rate_init": "Learning rate",
    "solver": "solver",
    "momentum": "Momentum",
    "alpha": "Alpha (L2)",
    "batch_size": "Batch_size",
    "max_iter": "Max_iter",
    "tol": "Tol",
    "n_iter_no_change": "N_iter_no_change",
    "random_state": "Random_state",
}
LINHAS_TABELA_5 = {
    "treino": "TREINO MSE (dpa)",
    "validacao": "VALIDAÇÃO MSE (dpa)",
    "diferenca": "MÉDIA DAS DIFS. (dpa)",
}


def parametros_mlp(rotulo):
    """Todos os parâmetros do MLPRegressor de uma hiperparametrização (BASE + o que varia)."""
    return {**BASE, **HIPERPARAMETRIZACOES[rotulo]}


def no_pipeline(parametros):
    """Parâmetros do MLPRegressor com o prefixo do passo do Pipeline (mlp__)."""
    return {f"mlp__{nome}": valor for nome, valor in parametros.items()}


def tabela_4():
    """Linhas da Tabela 4 por hiperparametrização; a topologia vira lista (neurônios por camada)."""
    tabela = {}
    for rotulo in HIPERPARAMETRIZACOES:
        parametros = parametros_mlp(rotulo)
        linhas = {chave: parametros[chave] for chave in LINHAS_TABELA_4}
        linhas["hidden_layer_sizes"] = list(linhas["hidden_layer_sizes"])
        tabela[rotulo] = linhas
    return tabela


def avaliar_rn(rotulo, X, y):
    """Resultado de avaliar, mais o tempo, as épocas por fold e os folds que pararam por max_iter.

    Os 5 folds rodam em paralelo (n_jobs=5); com random_state fixo, o resultado não depende do
    paralelismo. Como os folds rodam em outros processos, o ConvergenceWarning não chega aqui: os
    folds sem convergir são os que usaram max_iter épocas.
    """
    inicio = time.perf_counter()
    resultado, modelos = avaliar(RN, no_pipeline(parametros_mlp(rotulo)), X, y, n_jobs=K_FOLDS)
    tempo = time.perf_counter() - inicio
    epocas = [int(m.named_steps["mlp"].n_iter_) for m in modelos]
    max_iter = HIPERPARAMETRIZACOES[rotulo]["max_iter"]
    return resultado, {
        "tempo_s": tempo,
        "epocas_por_fold": epocas,
        "folds_em_max_iter": sum(e >= max_iter for e in epocas),
    }


def celula_tabela_4(chave, valor):
    if chave == "hidden_layer_sizes":
        return "[" + " ".join(str(n) for n in valor) + "]"
    return valor


def imprimir_tabela(titulo, linhas):
    """Tabela em texto: `linhas` mapeia o nome da linha -> células de U, E e O."""
    print(f"\n{titulo}")
    print(f"{'MODELO':<24}" + "".join(f"{rotulo:<34}" for rotulo in HIPERPARAMETRIZACOES))
    for nome, celulas in linhas.items():
        print(f"{nome:<24}" + "".join(f"{str(c):<34}" for c in celulas))


def main():
    X, y = carregar_dataset()
    resultados, execucao = {}, {}
    for rotulo in HIPERPARAMETRIZACOES:
        resultados[rotulo], execucao[rotulo] = avaliar_rn(rotulo, X, y)
        e = execucao[rotulo]
        print(f"\n{rotulo}: {parametros_mlp(rotulo)}")
        print(f"{e['tempo_s']:.1f} s; épocas por fold {e['epocas_por_fold']}; "
              f"{e['folds_em_max_iter']}/{K_FOLDS} folds em max_iter")
        print(descrever(resultados[rotulo]))
    melhor, criterio = escolher_melhor(resultados)
    conferir_ueo(resultados, melhor)
    t4 = tabela_4()

    saida = {
        "tabela_4": t4,
        "tabela_5": resultados,
        "melhor_rn": melhor,
        "criterio_melhor": criterio,
        "k_folds": K_FOLDS,
        "execucao": execucao,
    }
    TABELAS_4_5.parent.mkdir(exist_ok=True)
    TABELAS_4_5.write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    imprimir_tabela(
        "Tabela 4: hiperparametrizações RN",
        {nome: [celula_tabela_4(chave, h[chave]) for h in t4.values()] for chave, nome in LINHAS_TABELA_4.items()},
    )
    imprimir_tabela(
        "Tabela 5: resultados dos modelos RN",
        {nome: [formatar(r[chave]) for r in resultados.values()] for chave, nome in LINHAS_TABELA_5.items()},
    )
    print(f"\nEmpate técnico (até {criterio['limite_empate']:.6f}): {criterio['empatados']}")
    print(f"Melhor RN: {melhor}")
    print(f"Tabelas 4 e 5 salvas em: {TABELAS_4_5.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
