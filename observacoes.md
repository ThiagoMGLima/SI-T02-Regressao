# Observações

Decisões da Tarefa 2 (Regressão), tiradas dos notebooks 01 a 04 quando foram convertidos em scripts de `src/` (2026-10-08). Os termos seguem o `GLOSSARY.md`. Os números citados são os de `resultados/`.

## Código e ambiente

1. **Formato**: igual ao da Tarefa 1. Projeto `uv` e scripts `.py` em `src/`, um por etapa. Cada script salva seus números em `resultados/`, e o relatório é montado a partir deles. O notebook `T02_regressao.ipynb` só chama os `main()` dos scripts, na ordem.
2. **Módulo comum**: `src/validacao_cruzada.py`. Tem `carregar_dataset()`, `avaliar()`, `escolher_melhor()`, `conferir_ueo()`, `formatar()` e `montar()`. `uv run python src/validacao_cruzada.py` confere o exemplo do enunciado.
3. **Reprodução dos notebooks**: os scripts reproduzem os números dos notebooks. O CSV de treino/validação sai idêntico byte a byte, os MSE por fold do CART e da RN são iguais em todos os folds e o teste cego dá os mesmos MSE.
4. **Dependências**: `typst` gera o PDF. `ipykernel` e `nbconvert` ficam no grupo `dev`, só para o notebook.

## Dados

5. **Fontes externas**: `gerar_dados_vitimas.py` fica em `vendor/victsim3/`, sem modificação. O teste cego fica em `dados/teste_cego_1300v.csv` e é lido **somente** por `src/teste_cego.py`, que para se outro script de `src/` mencionar esse arquivo. Os dois são idênticos (SHA-256) aos arquivos dos links do enunciado (VictSim3, commit `99a3821`). A origem está em `vendor/victsim3/README.md`.
6. **Dataset de treino/validação**: 2.500 vítimas por classe de tri antes do ruído, idade média 40 e desvio 25, ruído 0,05 e semente 42. Depois do ruído, ficam 2448/2567/2561/2424. A idade medida tem média 39,92 e dpa 23,20, abaixo dos parâmetros, porque o gerador limita a idade a [1, 90].
   _Atenção_: o teste cego foi gerado com outra proporção de classes (106/379/419/396 vítimas, ~8% verdes). Na Tarefa 1, o dataset de treino replicou essa proporção.
7. **Entradas**: só as 10 features de entrada. `carregar_dataset` para se gcs, avpu, tri ou sobr entrarem em X.

## Validação cruzada e seleção

8. **Métricas** (ADR 0001): o MSE não tem raiz. O dpa usa divisor k−1 (`ddof=1`), o único que reproduz o exemplo do enunciado. A média das difs é a média de |MSE de treino − MSE de validação| por fold. A validação cruzada usa k = 5, com `KFold` embaralhado e semente 42, e os mesmos folds para o CART e a RN.
9. **Melhor modelo**: o de menor MSE médio de validação. Há empate técnico até 5% acima do menor. O desempate é pelo menor dpa de validação e, depois, pela menor média das difs.
10. **Critério de U/E/O** (`conferir_ueo`): a U tem o maior MSE médio de validação, a O tem a maior média das difs, com o treino abaixo da validação, e o critério do item 9 escolhe a E.

## Modelos

11. **CART** (`DecisionTreeRegressor`, `random_state=42`):
    - U: `max_depth=2` e `min_samples_leaf=0.25`, no máximo 4 folhas.
    - E: `max_depth=10` e `min_samples_leaf=20`. O MSE de validação se estabiliza em ~0,0026 a partir da profundidade 10.
    - O: `max_depth=None` e `min_samples_leaf=1`. A árvore decora o treino (MSE ≈ 0).

    O melhor é **E**: validação 0,002614 (0,000058).
12. **RN**: `StandardScaler` e `MLPRegressor` num Pipeline durante a validação cruzada. No retreino, o scaler é ajustado com as 10.000 vítimas e salvo à parte (`scaler.joblib`), como pede a seção ENTREGA.
13. **U/E/O da RN**: todas usam sgd, tanh, momentum 0,95, learning rate constante e `random_state=42`.
    - U: topologia [2], lr 0,001 e alpha 1.
    - E: [64 32], lr 0,02, alpha 0,01, `tol=1e-6` e até 2000 épocas.
    - O: dez camadas de 64, lr 0,05, alpha 0, batch 16 e 400 épocas sem parada antecipada. Os 5 folds param por `max_iter`, como esperado.

    A melhor é **E**: validação 0,001680 (0,000026).

## Teste cego e relatório

14. **Teste cego**: MSE de 0,002677 no CART e de 0,001694 na RN. RMSE de 0,051744 no CART e de 0,041161 na RN. A RN é melhor, com ganho de 36,72% no MSE e de 20,45% no RMSE.
15. **Relatório**: `src/gerar_relatorio.py` gera `entrega/relatorio_T02.pdf` com Typst e a fonte Carlito. O PDF tem só as tabelas 1 a 7 e os gráficos de dispersão, sem nenhum texto adicional. Os números usam vírgula decimal e 6 casas, e as cores dos cabeçalhos foram tiradas do enunciado. O mesmo script monta `entrega/` com o PDF, os três `.joblib` e `codigo_fonte_T02.zip` (`src/`, `vendor/`, `pyproject.toml`, `uv.lock` e `.python-version`). O PDF e o zip saem idênticos byte a byte a cada execução.
