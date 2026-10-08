# SI-T02-Regressao

Tarefa 2 de Sistemas Inteligentes 1: regressão da probabilidade de sobrevivência (`sobr`) com CART e RN MLP.

O enunciado (`T02_enunciado_regres.pdf`) é material da disciplina e não é versionado aqui.

## Estrutura

```
src/                    um script por etapa + o módulo comum
  validacao_cruzada.py  validação cruzada, MSE/dpa, critério do melhor modelo
  gerar_dataset.py      etapa 1: 10.000 vítimas e Tabela 1
  treinar_cart.py       etapa 2: CART U/E/O, Tabelas 2 e 3
  treinar_rn.py         etapa 3: RN U/E/O, Tabelas 4 e 5
  teste_cego.py         etapas 5 e 6: retreino, .joblib, Tabelas 6, 7 e 8
  gerar_relatorio.py    PDF (Typst, relatorio/relatorio.typ) e pacote entrega/
T02_regressao.ipynb     notebook único que chama os scripts na ordem
vendor/victsim3/        gerador do professor, sem modificação
dados/                  dataset de treino/validação e teste cego
resultados/             JSON de cada tabela e gráficos de dispersão
modelos/                melhor_cart.joblib, scaler.joblib, melhor_rn.joblib
entrega/                o que vai para o Moodle
observacoes.md          decisões de cada etapa
```

## Como rodar

```
uv sync
uv run python src/validacao_cruzada.py   # "Exemplo do enunciado reproduzido."
uv run python src/gerar_dataset.py
uv run python src/treinar_cart.py
uv run python src/treinar_rn.py          # ~3,5 min
uv run python src/teste_cego.py
uv run python src/gerar_relatorio.py
```

Ou abra `T02_regressao.ipynb` (kernel do `.venv`) e rode todas as células.
