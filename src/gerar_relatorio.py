"""Gera o relatório PDF com as tabelas do enunciado e monta o pacote de entrega em entrega/.

Última etapa. Lê só os números e os gráficos que os scripts anteriores salvaram em resultados/
(Tabela 1 de gerar_dataset.py, 2 e 3 de treinar_cart.py, 4 e 5 de treinar_rn.py, 6 a 8 de
teste_cego.py), formata-os como texto e os passa em JSON (sys.inputs) ao template Typst
relatorio/relatorio.typ, que faz toda a diagramação. O PDF tem somente as 7 tabelas e os
gráficos de dispersão da seção RELATÓRIO do enunciado, sem nenhum texto adicional (item 15 de
observacoes.md). Nunca lê os datasets.

Depois monta em entrega/ o que a seção ENTREGA do enunciado pede:
- relatorio_T02.pdf: o relatório;
- melhor_cart.joblib, melhor_rn.joblib e scaler.joblib: cópias dos modelos de modelos/;
- codigo_fonte_T02.zip: src/ (scripts e template), vendor/victsim3/, pyproject.toml, uv.lock e
  .python-version, sem os datasets.
O PDF e o zip saem idênticos byte a byte a cada execução: o PDF não leva data de criação, e o zip
usa ordem de arquivos, datas e permissões fixas.

Uso: uv run python src/gerar_relatorio.py
"""

import json
import shutil
import zipfile
from pathlib import Path

import typst

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "resultados"
TABELA_1 = RESULTADOS / "tabela1_dataset.json"
TABELAS_2_3 = RESULTADOS / "tabelas2_3_cart.json"
TABELAS_4_5 = RESULTADOS / "tabelas4_5_rn.json"
TABELAS_6_7_8 = RESULTADOS / "tabelas6_7_8_teste_cego.json"
TEMPLATE = RAIZ / "src" / "relatorio" / "relatorio.typ"

ENTREGA = RAIZ / "entrega"
RELATORIO = ENTREGA / "relatorio_T02.pdf"
CODIGO_FONTE = ENTREGA / "codigo_fonte_T02.zip"

# Modelos salvos por teste_cego.py, já com os nomes da seção ENTREGA do enunciado.
MODELOS = [RAIZ / "modelos" / nome for nome in ("melhor_cart.joblib", "melhor_rn.joblib", "scaler.joblib")]

# Código-fonte do zip, relativo à raiz do repositório. Os datasets ficam de fora: o de
# treino/validação sai de novo, idêntico, de src/gerar_dataset.py (semente 42), e o teste cego é o
# 1300v do VictSim3, que o professor já tem (a origem está em vendor/victsim3/README.md).
CODIGO = ["src", "vendor/victsim3", "pyproject.toml", "uv.lock", ".python-version"]
PASTA_NO_ZIP = "codigo_fonte_T02"
DATA_FIXA = (1980, 1, 1, 0, 0, 0)  # a menor data do formato zip

UEO = ["U", "E", "O"]
CORES = ["verde", "amarelo", "vermelho", "preto"]

# Títulos e rótulos de linha, copiados da seção RELATÓRIO do enunciado. "{MSE}" é o MSE com
# barra (média dos k folds); o template o troca pelo símbolo.
TITULOS = {
    1: "1) DATASET DE TREINAMENTO/VALIDAÇÃO",
    2: "2) HIPERPARAMETRIZAÇÕES CART",
    3: "3) RESULTADOS DOS MODELOS CART",
    4: "4) HIPERPARAMETRIZAÇÕES REDE NEURAL",
    5: "5) RESULTADOS DOS MODELOS RN",
    6: "6) MELHOR MODELO CART X MELHOR MODELO RN",
    7: "7) RESULTADOS DO TESTE CEGO: MELHOR CART X MELHOR RN",
    8: "8) GRÁFICO DE DISPERSÃO",
}
SUBTITULO_3 = "Médias dos MSEs com desvio padrão amostral por fold por hiperparametrização (U, E, O)"

LINHAS_TABELA_2 = {
    "min_samples_leaf": "Min_samples_leaf",
    "max_depth": "Max_depth",
    "random_state": "Random_state",
}
LINHAS_MSE = {  # Tabelas 3, 5 e 6
    "treino": "TREINO {MSE} (dpa)",
    "validacao": "VALIDAÇÃO {MSE} (dpa)",
    "diferenca": "MÉDIA DAS DIFS. (dpa)",
}
LINHAS_TABELA_4 = {
    "hidden_layer_sizes": "Topologia",
    "activation": "Função de ativação",
    "learning_rate_init": "Learning rate",
    "solver": "solver",
    # Linhas extras, que o enunciado permite: completam a hiperparametrização.
    "momentum": "Momentum",
    "alpha": "Alpha (L2)",
    "batch_size": "Batch_size",
    "max_iter": "Max_iter",
    "tol": "Tol",
    "n_iter_no_change": "N_iter_no_change",
    "random_state": "Random_state",
}
NOTA_GANHO = "(somente para o melhor entre CART e RN)"


def ler_json(caminho):
    return json.loads(caminho.read_text(encoding="utf-8"))


def numero(valor, casas=6):
    """Número com vírgula decimal, como no enunciado."""
    return f"{valor:.{casas}f}".replace(".", ",")


def hiperparametro(valor):
    """Célula das Tabelas 2 e 4: topologia como [64 32], float com vírgula e sem notação
    científica (0,000001), e null do JSON (max_depth sem limite) como "None"."""
    if valor is None:
        return "None"
    if isinstance(valor, list):  # neurônios por camada oculta
        return "[" + " ".join(str(n) for n in valor) + "]"
    if isinstance(valor, float):
        return f"{valor:.10f}".rstrip("0").rstrip(".").replace(".", ",")
    return str(valor)


def media_dpa(estatistica):
    """Célula das Tabelas 3, 5 e 6 como [média, dpa], com 6 casas; o template a mostra como
    "0,001968 (0,000013)", o formato de formatar em validacao_cruzada.py."""
    return [numero(estatistica["media"]), numero(estatistica["dpa"])]


def linhas_mse(resultados, rotulos):
    """Linhas das Tabelas 3, 5 e 6: treino, validação e diferença, uma coluna por rótulo."""
    return [[nome, *[media_dpa(resultados[r][chave]) for r in rotulos]] for chave, nome in LINHAS_MSE.items()]


def conferir(cart, rn, teste):
    """Confere que os JSON das Tabelas 2 a 8 vêm da mesma sequência de execuções: os rótulos do
    melhor CART e da melhor RN e a Tabela 6 do teste cego têm de ser os das Tabelas 3 e 5."""
    for nome, resultados, tabela in [("cart", cart, "tabela_3"), ("rn", rn, "tabela_5")]:
        melhor = resultados[f"melhor_{nome}"]
        if teste[f"melhor_{nome}"] != melhor:
            raise SystemExit(f"{nome}: o melhor do teste cego ({teste[f'melhor_{nome}']}) difere de {melhor}")
        if teste["tabela_6"][nome] != resultados[tabela][melhor]:
            raise SystemExit(f"{nome}: a Tabela 6 difere da {tabela}[{melhor}]")


def linhas_tabela_7(t7):
    """MSE e RMSE dos dois modelos; o ganho % só na coluna do melhor, "—" na outra."""
    linhas = []
    for metrica, nome in [("mse", "MSE"), ("rmse", "RMSE")]:
        ganho = t7[f"ganho_{metrica}_pct"]
        linhas.append([nome, *[numero(t7[metrica][m]) for m in ("cart", "rn")]])
        linhas.append([
            f"Ganho % de {nome}\n{NOTA_GANHO}",
            *[f"{numero(ganho, 2)}%" if m == t7["melhor"] else "—" for m in ("cart", "rn")],
        ])
    return linhas


def montar_tabelas():
    """Textos das tabelas e caminhos dos gráficos, prontos para o template: {"t1": ..., "t8": ...}."""
    t1 = ler_json(TABELA_1)["tabela_1"]
    cart, rn, teste = ler_json(TABELAS_2_3), ler_json(TABELAS_4_5), ler_json(TABELAS_6_7_8)
    conferir(cart, rn, teste)
    melhores = [f"Melhor CART ({teste['melhor_cart']})", f"Melhor RN ({teste['melhor_rn']})"]
    return {
        "t1": {
            "titulo": TITULOS[1],
            "colunas": ["MODELO", "VALOR"],
            "linhas": [
                *[[f"Vítimas tri={cor}", str(t1["vitimas_por_classe"][cor])] for cor in CORES],
                ["Idade das vítimas", numero(t1["idade_media"], 2)],
                ["Desvio padrão amostral da idade", numero(t1["dpa_idade"], 2)],
                ["ruído", hiperparametro(t1["ruido"])],
            ],
        },
        "t2": {
            "titulo": TITULOS[2],
            "colunas": ["MODELO", *UEO],
            "linhas": [
                [nome, *[hiperparametro(cart["tabela_2"][r][chave]) for r in UEO]]
                for chave, nome in LINHAS_TABELA_2.items()
            ],
        },
        "t3": {
            "titulo": TITULOS[3],
            "subtitulo": SUBTITULO_3,
            "colunas": ["MODELO", *[f"CART {r}" for r in UEO]],
            "linhas": linhas_mse(cart["tabela_3"], UEO),
        },
        "t4": {
            "titulo": TITULOS[4],
            "colunas": ["MODELO", *UEO],
            "linhas": [
                [nome, *[hiperparametro(rn["tabela_4"][r][chave]) for r in UEO]]
                for chave, nome in LINHAS_TABELA_4.items()
            ],
        },
        "t5": {
            "titulo": TITULOS[5],
            "colunas": ["MODELO", *[f"RN {r}" for r in UEO]],
            "linhas": linhas_mse(rn["tabela_5"], UEO),
        },
        "t6": {
            "titulo": TITULOS[6],
            "colunas": ["MODELO", *melhores],
            "linhas": linhas_mse(teste["tabela_6"], ["cart", "rn"]),
        },
        "t7": {"titulo": TITULOS[7], "colunas": ["MODELO", *melhores], "linhas": linhas_tabela_7(teste["tabela_7"])},
        "t8": {
            "titulo": TITULOS[8],
            # Caminhos relativos à raiz do projeto, que é a raiz do Typst na compilação.
            "graficos": ["/" + teste["tabela_8"][m] for m in ("cart", "rn")],
        },
    }


def gerar_pdf(tabelas, destino):
    """Compila o template com as tabelas e grava o PDF.

    Para em qualquer aviso do Typst: sem a fonte Carlito, por exemplo, ele só avisa e troca pela
    fonte padrão.
    """
    pdf, avisos = typst.compile_with_warnings(
        str(TEMPLATE), root=str(RAIZ), sys_inputs={"dados": json.dumps(tabelas, ensure_ascii=False)}
    )
    if avisos:
        raise SystemExit("O Typst emitiu avisos:\n" + "\n".join(f"- {a.message}" for a in avisos))
    destino.write_bytes(pdf)


def arquivos_do_codigo():
    """Arquivos do código-fonte do zip, em ordem fixa e sem os caches do Python."""
    for item in CODIGO:
        caminho = RAIZ / item
        if caminho.is_dir():
            yield from sorted(p for p in caminho.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        else:
            yield caminho


def montar_zip(destino):
    """Zip do código-fonte, dentro da pasta PASTA_NO_ZIP. Data e permissões fixas em todos os
    arquivos, para que duas execuções deem o mesmo zip byte a byte. Devolve os caminhos no zip."""
    nomes = []
    with zipfile.ZipFile(destino, "w") as zf:
        for arquivo in arquivos_do_codigo():
            info = zipfile.ZipInfo(f"{PASTA_NO_ZIP}/{arquivo.relative_to(RAIZ).as_posix()}", date_time=DATA_FIXA)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16  # rw-r--r--
            zf.writestr(info, arquivo.read_bytes(), compresslevel=9)
            nomes.append(info.filename)
    return nomes


def main():
    ENTREGA.mkdir(exist_ok=True)
    gerar_pdf(montar_tabelas(), RELATORIO)
    for modelo in MODELOS:
        shutil.copyfile(modelo, ENTREGA / modelo.name)
    nomes = montar_zip(CODIGO_FONTE)

    print(f"Pacote de entrega em {ENTREGA.relative_to(RAIZ)}/:")
    for arquivo in sorted(ENTREGA.iterdir()):
        print(f"  {arquivo.name:<24} {arquivo.stat().st_size:>8} bytes")
    print(f"\n{CODIGO_FONTE.name} ({len(nomes)} arquivos):")
    for nome in nomes:
        print(f"  {nome}")


if __name__ == "__main__":
    main()
