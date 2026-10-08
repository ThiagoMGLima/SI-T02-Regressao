# VictSim3: gerador de vítimas

`gerar_dados_vitimas.py` é uma cópia **sem modificação** do gerador de vítimas do projeto [VictSim3](https://github.com/tacla/VictSim3), do Prof. Tacla (UTFPR), o mesmo arquivo do link do enunciado.

- Origem: [`data_creation/gerar_dados_vitimas.py`](https://github.com/tacla/VictSim3/blob/99a38217e0fcb8e173d4c2b96b10d53ef28faac3/data_creation/gerar_dados_vitimas.py), commit `99a3821` (o `main` em 2026-10-08).
- SHA-256: `3bd9cc4eee2d7f71105eca203656fadbc52ecca704a0fc422a531de037c457ec`.

O teste cego `dados/teste_cego_1300v.csv` vem do mesmo commit: [`datasets/vict/1300v/data.csv`](https://github.com/tacla/VictSim3/blob/99a38217e0fcb8e173d4c2b96b10d53ef28faac3/datasets/vict/1300v/data.csv) (SHA-256 `21a443aa8185fa9f9bf003dde42e91fc876fe87f2452daedb7cbbfdaceb8aaa6`), o link "1300 vítimas" do enunciado.

`src/gerar_dataset.py` importa este arquivo e contorna dois efeitos colaterais dele sem alterá-lo: a criação de `./datasets/vict/1300v/` no import e o caminho de saída fixo (`OUTPUT_CSV`).
