// Relatório da Tarefa 2: somente as 7 tabelas e os gráficos de dispersão da seção RELATÓRIO do
// enunciado, sem nenhum texto adicional (item 15 de observacoes.md). Compilado por
// src/gerar_relatorio.py, que lê resultados/ e passa todos os textos já formatados em
// sys.inputs.dados (JSON): títulos, cabeçalhos, rótulos de linha, números e os caminhos dos
// gráficos. Este template só faz a diagramação: cores, fontes, larguras e alinhamentos.
#let d = json(bytes(sys.inputs.dados))

// Sem data de criação nos metadados: o PDF sai idêntico byte a byte a cada execução.
#set document(date: none)
#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm))
// Carlito tem as métricas da Calibri, a fonte do enunciado.
#set text(font: "Carlito", size: 11pt, lang: "pt")
#set par(spacing: 0.5em)

#let pequeno = 8.5pt

// Por tabela: cor do cabeçalho e do texto dele (tiradas do enunciado), larguras das colunas e se
// os rótulos de linha saem pequenos e centralizados (Tabelas 3, 5, 6 e 7, como no enunciado).
#let estilo = (
  t1: (cor: rgb("#E5DFEC"), texto: black, larguras: (6.6cm, 3.6cm), pequenos: false),
  t2: (cor: rgb("#DAEEF3"), texto: black, larguras: (4cm, 3.2cm, 3.2cm, 3.2cm), pequenos: false),
  t3: (cor: rgb("#DAEEF3"), texto: black, larguras: (3.8cm, 1fr, 1fr, 1fr), pequenos: true),
  t4: (cor: rgb("#FDE9D9"), texto: black, larguras: (3.6cm, 1fr, 1fr, 3fr), pequenos: false),
  t5: (cor: rgb("#FDE9D9"), texto: black, larguras: (3.8cm, 1fr, 1fr, 1fr), pequenos: true),
  t6: (cor: rgb("#E36C0A"), texto: white, larguras: (3.8cm, 4cm, 4cm), pequenos: true),
  t7: (cor: rgb("#B2A1C7"), texto: black, larguras: (3.8cm, 4cm, 4cm), pequenos: true),
)

#let titulo(texto) = block(below: 0.3em, text(weight: "bold", texto))
#let subtitulo(texto) = block(below: 0.3em, text(style: "italic", size: 9.5pt, texto))

// "{MSE}" nos rótulos vira o MSE com barra (média dos k folds), em itálico, como no enunciado.
// Uma quebra de linha separa o rótulo de uma nota menor (Tabela 7).
#let mse = $overline(M S E)$
#let rotulo(r) = {
  let (principal, ..nota) = r.split("\n")
  principal.split("{MSE}").join(mse)
  for n in nota { linebreak(); text(size: 7.5pt, n) }
}

// Célula de texto, ou [média, dpa] das Tabelas 3, 5 e 6: "0,001968 (0,000013)".
#let celula(c) = if type(c) == array {
  let (media, dpa) = c
  [#media (#dpa)]
} else { c }

#let tabela(t, e) = table(
  columns: e.larguras,
  stroke: 0.5pt + black,
  inset: (x: 5pt, y: 2.5pt),
  align: (x, y) => (if x == 0 and y > 0 and not e.pequenos { left } else { center }) + horizon,
  fill: (x, y) => if y == 0 { e.cor },
  table.header(..t.colunas.map(c => text(weight: "bold", fill: e.texto, c))),
  ..t.linhas.map(((r, ..celulas)) => (
    if e.pequenos { text(size: pequeno, rotulo(r)) } else { rotulo(r) },
    ..celulas.map(celula),
  )).flatten(),
)

#for n in range(1, 8) {
  let chave = "t" + str(n)
  let t = d.at(chave)
  // Título e tabela juntos, sem dividir a tabela entre páginas.
  block(above: 1.4em, breakable: false, {
    titulo(t.titulo)
    if "subtitulo" in t { subtitulo(t.subtitulo) }
    tabela(t, estilo.at(chave))
  })
}

// Seção 8: um gráfico de dispersão por modelo, lado a lado, sem dividir entre páginas.
#block(above: 1.4em, breakable: false)[
  #titulo(d.t8.titulo)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 0.6cm,
    ..d.t8.graficos.map(g => image(g, width: 100%)),
  )
]
