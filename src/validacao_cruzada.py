"""Validação cruzada e seleção do melhor CART / melhor RN.

Módulo comum aos scripts de treino. Avalia um regressor com KFold (k=5, embaralhado, semente 42)
no dataset de treino/validação, usando só as 10 features de entrada para estimar a sobr, e
escolhe entre U, E e O pelo critério do item 9 de observacoes.md.
Nunca lê o teste cego.

Uso nos scripts de src/ (rodados com `uv run python src/<script>.py`):
    from validacao_cruzada import avaliar, carregar_dataset, escolher_melhor, formatar

Verificação com o exemplo do enunciado: uv run python src/validacao_cruzada.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / "dados" / "treino_validacao_10000v.csv"

FEATURES_ENTRADA = ["idade", "fc", "fr", "pas", "spo2", "temp", "pr", "sg", "fx", "queim"]
FEATURES_PROIBIDAS = ["gcs", "avpu", "tri"]
ALVO = "sobr"
N_VITIMAS = 10000

K_FOLDS = 5
SEMENTE = 42
TOLERANCIA_EMPATE = 0.05  # empate técnico: até 5% acima do menor MSE médio de validação (item 9)

# Exemplo do enunciado (Tabela 3): MSE por fold de um CART U com k=3, e a média (dpa) esperada,
# como texto, com as casas que o enunciado mostra.
EXEMPLO_TREINO = [0.0232, 0.0011, 0.0198]
EXEMPLO_VALIDACAO = [0.0721, 0.0703, 0.0699]
EXEMPLO_ESPERADO = {
    "treino": ("0.0147", "0.0119"),
    "validacao": ("0.0707", "0.0011"),
    "diferenca": ("0.0560", "0.01139"),
}


def carregar_dataset():
    """Dataset de treino/validação: X com as 10 features de entrada e y = sobr."""
    df = pd.read_csv(DATASET)
    if len(df) != N_VITIMAS:
        raise SystemExit(f"{DATASET.name}: esperado {N_VITIMAS} vítimas, obtido {len(df)}")
    X = df[FEATURES_ENTRADA]
    if set(X.columns) & set(FEATURES_PROIBIDAS + [ALVO]):
        raise SystemExit(f"feature proibida em X: {set(X.columns) & set(FEATURES_PROIBIDAS + [ALVO])}")
    return X, df[ALVO]


def montar(estimador, hiperparametrizacao):
    """Cópia não ajustada do estimador com a hiperparametrização aplicada.

    Todo `random_state` deixado em None, inclusive dentro de um Pipeline (ex.: `mlp__random_state`),
    recebe a semente 42. O retreino deve montar o modelo por aqui para reproduzir o da validação.
    """
    modelo = clone(estimador).set_params(**hiperparametrizacao)
    sem_semente = {
        nome: SEMENTE
        for nome, valor in modelo.get_params().items()
        if nome.split("__")[-1] == "random_state" and valor is None
    }
    return modelo.set_params(**sem_semente)


def dpa(valores):
    """Desvio padrão amostral (divisor k−1, ADR 0001)."""
    return float(np.std(valores, ddof=1))


def _estatisticas(valores):
    return {
        "por_fold": [float(v) for v in valores],
        "media": float(np.mean(valores)),
        "dpa": dpa(valores),
    }


def resumir_folds(mse_treino, mse_validacao):
    """Por fold, média e dpa (ddof=1) do MSE de treino, de validação e da diferença treino-validação.

    Devolve {"treino", "validacao", "diferenca"}, cada um com "por_fold", "media" e "dpa".
    A diferença treino-validação é o valor absoluto, fold a fold (as "difs" do enunciado).
    """
    treino = np.asarray(mse_treino, dtype=float)
    validacao = np.asarray(mse_validacao, dtype=float)
    return {
        "treino": _estatisticas(treino),
        "validacao": _estatisticas(validacao),
        "diferenca": _estatisticas(np.abs(treino - validacao)),
    }


def avaliar(estimador, hiperparametrizacao, X, y, n_jobs=None):
    """Avalia uma hiperparametrização com validação cruzada (k=5, KFold embaralhado, semente 42).

    Os folds são sempre os mesmos, para qualquer estimador e hiperparametrização. O MSE é a média
    dos erros quadráticos, sem raiz (ADR 0001); o cross_validate devolve o MSE negativo, por isso
    o sinal é trocado. Devolve (resultado, modelos): o resultado é um dicionário serializável em
    JSON, como em resumir_folds, e modelos são os k estimadores ajustados, um por fold.
    """
    folds = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMENTE)
    cv = cross_validate(
        montar(estimador, hiperparametrizacao),
        X,
        y,
        cv=folds,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        return_estimator=True,
        n_jobs=n_jobs,
    )
    return resumir_folds(-cv["train_score"], -cv["test_score"]), cv["estimator"]


def escolher_melhor(resultados):
    """Melhor CART / melhor RN entre U, E e O, pelo critério do item 9 de observacoes.md.

    `resultados` mapeia rótulo -> resultado de avaliar, ex.: {"U": ..., "E": ..., "O": ...}.
    Empate técnico entre quem tiver MSE médio de validação até 5% acima do menor; desempate por
    menor dpa de validação e, depois, menor média das difs. Devolve o rótulo e o critério aplicado
    ({"tolerancia_empate", "limite_empate", "empatados"}).
    """
    menor = min(r["validacao"]["media"] for r in resultados.values())
    limite = menor * (1 + TOLERANCIA_EMPATE)
    empatados = [rotulo for rotulo, r in resultados.items() if r["validacao"]["media"] <= limite]
    melhor = min(
        empatados,
        key=lambda rotulo: (resultados[rotulo]["validacao"]["dpa"], resultados[rotulo]["diferenca"]["media"]),
    )
    return melhor, {"tolerancia_empate": TOLERANCIA_EMPATE, "limite_empate": limite, "empatados": empatados}


def conferir_ueo(resultados, melhor):
    """Para se U, E e O não cumprirem o critério do item 10 de observacoes.md.

    U tem o maior MSE médio de validação, O a maior média das difs (com o treino abaixo da
    validação) e o critério de melhor modelo escolhe E.
    """
    maior_validacao = max(resultados, key=lambda r: resultados[r]["validacao"]["media"])
    maior_diferenca = max(resultados, key=lambda r: resultados[r]["diferenca"]["media"])
    o = resultados["O"]
    if maior_validacao != "U":
        raise SystemExit(f"U deveria ter o maior MSE médio de validação, mas foi {maior_validacao}")
    if maior_diferenca != "O" or o["treino"]["media"] >= o["validacao"]["media"]:
        raise SystemExit(f"O deveria ter a maior média das difs, com treino < validação (maior: {maior_diferenca})")
    if melhor != "E":
        raise SystemExit(f"o critério de melhor modelo deveria escolher E, mas escolheu {melhor}")


def numero(valor, casas=6):
    """Número com vírgula decimal, como no enunciado."""
    return f"{valor:.{casas}f}".replace(".", ",")


def formatar(estatistica, casas=6):
    """Célula das Tabelas 3, 5 e 6: média (dpa), ex.: "0,001968 (0,000013)"."""
    return f"{numero(estatistica['media'], casas)} ({numero(estatistica['dpa'], casas)})"


def descrever(resultado):
    """Texto no formato do exemplo do enunciado: MSE por fold, média e dpa."""
    linhas = []
    for rotulo, nome in [
        ("Treino    MSE por fold:", "treino"),
        ("Validação MSE por fold:", "validacao"),
        ("Diferença absoluta MSE:", "diferenca"),
    ]:
        e = resultado[nome]
        por_fold = ", ".join(f"{v:.6f}" for v in e["por_fold"])
        linhas.append(f"{rotulo} [{por_fold}] média={e['media']:.6f} dpa={e['dpa']:.6f}")
    return "\n".join(linhas)


def main():
    """Confere o cálculo de média, dpa e diferença treino-validação com o exemplo do enunciado."""
    resumo = resumir_folds(EXEMPLO_TREINO, EXEMPLO_VALIDACAO)
    print(descrever(resumo))
    for nome, esperado in EXEMPLO_ESPERADO.items():
        obtido = (resumo[nome]["media"], resumo[nome]["dpa"])
        # O enunciado trunca (0,070767 aparece como 0,0707): aceita até 1 na última casa mostrada.
        for o, e in zip(obtido, esperado):
            if abs(o - float(e)) > 10 ** -len(e.split(".")[1]):
                raise SystemExit(f"{nome}: obtido {obtido}, esperado {esperado}")
        print(f"{nome:<10} média={obtido[0]:.5f} dpa={obtido[1]:.5f}  (enunciado: {esperado[0]} ({esperado[1]}))")
    print("Exemplo do enunciado reproduzido.")


if __name__ == "__main__":
    main()
