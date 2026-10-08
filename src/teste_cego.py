"""Retreina o melhor CART e a melhor RN, salva os modelos e roda o teste cego (Tabelas 6, 7 e 8).

Etapas 5 e 6 do enunciado. Os rótulos do melhor CART e da melhor RN saem dos JSON das Tabelas 2
e 3 e 4 e 5 (`melhor_cart`, `melhor_rn`). Cada um é retreinado com as 10.000 vítimas do dataset
de treino/validação, sem validação cruzada, e salvo em modelos/ com os nomes da seção ENTREGA do
enunciado: melhor_cart.joblib, e, para a RN, scaler.joblib (o StandardScaler ajustado nas 10.000
vítimas) e melhor_rn.joblib (o MLPRegressor treinado com os dados padronizados). Só depois disso
o teste cego é lido: este é o único script do projeto que o lê (item 5 de observacoes.md), e as
métricas saem dos modelos recarregados dos .joblib, os mesmos da entrega.

Uso: uv run python src/teste_cego.py
"""

import io
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from treinar_cart import CART, TABELAS_2_3
from treinar_cart import HIPERPARAMETRIZACOES as HIPERPARAMETRIZACOES_CART
from treinar_rn import TABELAS_4_5, parametros_mlp, tabela_4
from validacao_cruzada import ALVO, FEATURES_ENTRADA, carregar_dataset, formatar, montar, numero

RAIZ = Path(__file__).resolve().parent.parent
TESTE_CEGO = RAIZ / "dados" / "teste_cego_1300v.csv"
N_VITIMAS_TESTE_CEGO = 1300
MODELOS = RAIZ / "modelos"
MELHOR_CART = MODELOS / "melhor_cart.joblib"
SCALER = MODELOS / "scaler.joblib"
MELHOR_RN = MODELOS / "melhor_rn.joblib"
RESULTADOS = RAIZ / "resultados"
TABELAS_6_7_8 = RESULTADOS / "tabelas6_7_8_teste_cego.json"
DISPERSAO = {"cart": RESULTADOS / "dispersao_cart.png", "rn": RESULTADOS / "dispersao_rn.png"}

NOMES = {"cart": "CART", "rn": "RN"}
CORES = {"cart": "tab:blue", "rn": "tab:orange"}
LINHAS_TABELA_6 = {
    "treino": "TREINO MSE (dpa)",
    "validacao": "VALIDAÇÃO MSE (dpa)",
    "diferenca": "MÉDIA DAS DIFS. (dpa)",
}


def ler_json(caminho):
    return json.loads(caminho.read_text(encoding="utf-8"))


def melhores():
    """Rótulos do melhor CART e da melhor RN e a Tabela 6, a partir dos JSON das Tabelas 2 a 5.

    A Tabela 6 copia `tabela_3[melhor_cart]` e `tabela_5[melhor_rn]`. Confere que as
    hiperparametrizações gravadas nesses JSON são as dos scripts de treino, para que o retreino
    use a mesma hiperparametrização que foi validada.
    """
    cart, rn = ler_json(TABELAS_2_3), ler_json(TABELAS_4_5)
    melhor_cart, melhor_rn = cart["melhor_cart"], rn["melhor_rn"]
    if cart["tabela_2"][melhor_cart] != HIPERPARAMETRIZACOES_CART[melhor_cart]:
        raise SystemExit(f"CART {melhor_cart}: a Tabela 2 do JSON difere de treinar_cart.py")
    if rn["tabela_4"][melhor_rn] != tabela_4()[melhor_rn]:
        raise SystemExit(f"RN {melhor_rn}: a Tabela 4 do JSON difere de treinar_rn.py")
    tabela_6 = {"cart": cart["tabela_3"][melhor_cart], "rn": rn["tabela_5"][melhor_rn]}
    return melhor_cart, melhor_rn, tabela_6


def retreinar_cart(rotulo, X, y):
    """Melhor CART montado como na validação e ajustado com as 10.000 vítimas, sem validação cruzada."""
    modelo = montar(CART, HIPERPARAMETRIZACOES_CART[rotulo])
    inicio = time.perf_counter()
    modelo.fit(X, y)
    tempo = time.perf_counter() - inicio
    return modelo, {
        "tempo_s": tempo,
        "profundidade": int(modelo.get_depth()),
        "folhas": int(modelo.get_n_leaves()),
        "mse_treino": float(mean_squared_error(y, modelo.predict(X))),
    }


def retreinar_rn(rotulo, X, y):
    """Melhor RN ajustada com as 10.000 vítimas, sem validação cruzada.

    Na validação, o StandardScaler ficava no Pipeline, ajustado só no treino de cada fold. Aqui
    não há folds: o scaler é ajustado com as 10.000 vítimas e devolvido à parte (scaler.joblib,
    como pede a seção ENTREGA), e a RN é treinada com os dados padronizados.
    """
    parametros = parametros_mlp(rotulo)
    scaler = StandardScaler()
    X_padronizado = scaler.fit_transform(X)
    modelo = MLPRegressor(**parametros)
    inicio = time.perf_counter()
    modelo.fit(X_padronizado, y)
    tempo = time.perf_counter() - inicio
    return scaler, modelo, {
        "tempo_s": tempo,
        "epocas": int(modelo.n_iter_),
        "max_iter": parametros["max_iter"],
        "mse_treino": float(mean_squared_error(y, modelo.predict(X_padronizado))),
    }


def conferir_sem_vazamento():
    """Para se algum outro script de src/ mencionar o arquivo do teste cego."""
    for script in sorted((RAIZ / "src").glob("*.py")):
        if script.name != Path(__file__).name and TESTE_CEGO.name in script.read_text(encoding="utf-8"):
            raise SystemExit(f"{script.name} menciona o teste cego ({TESTE_CEGO.name})")


def carregar_teste_cego():
    """Teste cego: X com as 10 features de entrada e y = sobr. Lido só aqui."""
    df = pd.read_csv(TESTE_CEGO)
    if len(df) != N_VITIMAS_TESTE_CEGO:
        raise SystemExit(f"{TESTE_CEGO.name}: esperado {N_VITIMAS_TESTE_CEGO} vítimas, obtido {len(df)}")
    return df[FEATURES_ENTRADA], df[ALVO]


def ganho(metrica):
    """Melhor modelo e ganho % da Tabela 7: (1 − melhor / pior) × 100, só para o melhor."""
    if metrica["cart"] < metrica["rn"]:
        return "cart", (1 - metrica["cart"] / metrica["rn"]) * 100
    if metrica["rn"] < metrica["cart"]:
        return "rn", (1 - metrica["rn"] / metrica["cart"]) * 100
    return None, 0.0  # empate exato: nenhum ganho


def salvar_dispersao(modelo, rotulo, y, predito, mse, destino):
    """Gráfico de dispersão da Tabela 8: sobr real × predita no teste cego, com a diagonal y = x.

    O PNG não leva metadados de software, para sair idêntico a cada execução.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, predito, s=10, alpha=0.5, color=CORES[modelo], edgecolors="none")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="y = x")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("sobr real")
    ax.set_ylabel("sobr predita")
    ax.set_title(f"Teste cego: melhor {NOMES[modelo]} ({rotulo}), MSE = {numero(mse)}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.4)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, metadata={"Software": None})
    plt.close(fig)
    destino.write_bytes(buffer.getvalue())


def main():
    conferir_sem_vazamento()
    melhor_cart, melhor_rn, t6 = melhores()
    rotulos = {"cart": melhor_cart, "rn": melhor_rn}
    X, y = carregar_dataset()

    # Retreino com as 10.000 vítimas; os modelos são salvos antes de qualquer leitura do teste cego.
    cart, retreino_cart = retreinar_cart(melhor_cart, X, y)
    scaler, rn, retreino_rn = retreinar_rn(melhor_rn, X, y)
    MODELOS.mkdir(exist_ok=True)
    joblib.dump(cart, MELHOR_CART)
    joblib.dump(scaler, SCALER)
    joblib.dump(rn, MELHOR_RN)
    print(f"Melhor CART ({melhor_cart}) retreinado: {retreino_cart}")
    print(f"Melhor RN ({melhor_rn}) retreinada: {retreino_rn}")
    print(f"Modelos salvos em: {', '.join(str(p.relative_to(RAIZ)) for p in (MELHOR_CART, SCALER, MELHOR_RN))}")

    # Teste cego: primeira e única leitura, com os modelos recarregados dos .joblib da entrega.
    X_cego, y_cego = carregar_teste_cego()
    cart, scaler, rn = joblib.load(MELHOR_CART), joblib.load(SCALER), joblib.load(MELHOR_RN)
    predito = {"cart": cart.predict(X_cego), "rn": rn.predict(scaler.transform(X_cego))}
    mse = {m: float(mean_squared_error(y_cego, p)) for m, p in predito.items()}
    rmse = {m: float(np.sqrt(v)) for m, v in mse.items()}
    melhor_mse, ganho_mse = ganho(mse)
    melhor_rmse, ganho_rmse = ganho(rmse)
    if melhor_mse != melhor_rmse:  # RMSE = √MSE preserva a ordem
        raise SystemExit(f"melhor pelo MSE ({melhor_mse}) difere do melhor pelo RMSE ({melhor_rmse})")

    RESULTADOS.mkdir(exist_ok=True)
    for m in predito:
        salvar_dispersao(m, rotulos[m], y_cego, predito[m], mse[m], DISPERSAO[m])

    saida = {
        "melhor_cart": melhor_cart,
        "melhor_rn": melhor_rn,
        "tabela_6": t6,
        "tabela_7": {
            "mse": mse,
            "rmse": rmse,
            "melhor": melhor_mse,
            "ganho_mse_pct": ganho_mse,
            "ganho_rmse_pct": ganho_rmse,
        },
        "tabela_8": {m: str(DISPERSAO[m].relative_to(RAIZ)) for m in DISPERSAO},
        "teste_cego": {"vitimas": int(len(y_cego))},
        "modelos": {
            "cart": str(MELHOR_CART.relative_to(RAIZ)),
            "scaler": str(SCALER.relative_to(RAIZ)),
            "rn": str(MELHOR_RN.relative_to(RAIZ)),
        },
        "retreino": {"cart": retreino_cart, "rn": retreino_rn},
    }
    TABELAS_6_7_8.write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    colunas = f"{'':<24}{'Melhor CART (' + melhor_cart + ')':<28}{'Melhor RN (' + melhor_rn + ')':<28}"
    print(f"\nTabela 6: melhor CART x melhor RN\n{colunas}")
    for chave, nome in LINHAS_TABELA_6.items():
        print(f"{nome:<24}{formatar(t6['cart'][chave]):<28}{formatar(t6['rn'][chave]):<28}")
    print(f"\nTabela 7: teste cego, melhor CART x melhor RN\n{colunas}")
    print(f"{'MSE':<24}{numero(mse['cart']):<28}{numero(mse['rn']):<28}")
    print(f"{'RMSE':<24}{numero(rmse['cart']):<28}{numero(rmse['rn']):<28}")
    print(f"Melhor no teste cego: {NOMES[melhor_mse]}; ganho % de MSE {numero(ganho_mse, 2)}, "
          f"de RMSE {numero(ganho_rmse, 2)}")
    print(f"\nGráficos de dispersão (Tabela 8): {', '.join(str(p.relative_to(RAIZ)) for p in DISPERSAO.values())}")
    print(f"Tabelas 6, 7 e 8 salvas em: {TABELAS_6_7_8.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
