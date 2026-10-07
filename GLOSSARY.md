# SI-T02 Regressão

Estimar a probabilidade de sobrevivência de vítimas de catástrofes a partir de sinais vitais e lesões, comparando um regressor CART com um regressor MLP.

## Dados

**Vítima**:
Uma amostra (linha) do dataset: um indivíduo com seus sinais vitais, lesões e probabilidade de sobrevivência.
_Avoid_: paciente, registro, exemplo

**Feature de entrada**:
Uma das 10 características permitidas como entrada do modelo: idade, fc, fr, pas, spo2, temp, pr, sg, fx, queim.
_Avoid_: atributo, variável independente

**Feature proibida**:
Característica presente no dataset que nunca pode entrar no treino: gcs, avpu, tri (e sobr, que é o alvo).

**sobr**:
A variável alvo: probabilidade de sobrevivência da vítima, real em [0, 1].
_Avoid_: target, label, classe

**tri**:
Triagem START (verde, amarelo, vermelho, preto). Usada apenas para descrever o dataset, nunca como entrada.

**Dataset de treino/validação**:
As exatamente 10.000 vítimas geradas por `gerar_dados_vitimas.py`, usadas na validação cruzada e no retreino.
_Avoid_: dataset de treino (sozinho), base

**Dataset de teste cego**:
As 1.300 vítimas usadas somente na etapa de teste cego; qualquer outro uso é vazamento de dados.
_Avoid_: test set, holdout

**Ruído**:
Parâmetro do gerador (0 < ruído ≤ 0.05) que perturba as features de entrada e troca a tri de vítimas por classes vizinhas. Não é o ruído da própria sobr, que é fixo no gerador.
_Avoid_: noise, nivel de ruído (no relatório)

## Modelagem

**Hiperparametrização**:
Uma configuração nomeada de hiperparâmetros de um algoritmo (CART ou RN), avaliada por validação cruzada.
_Avoid_: parametrização, config

**U / E / O**:
Os três rótulos obrigatórios de hiperparametrização por algoritmo: Subajustado (underfitted), Equilibrado, Sobreajustado (overfitted).

**Melhor modelo**:
A hiperparametrização escolhida (por algoritmo) a partir dos resultados da validação cruzada, que vai para o retreino e o teste cego.

**Retreino**:
Fit da melhor hiperparametrização com todas as 10.000 vítimas, sem validação cruzada nem split.

**Teste cego**:
Avaliação dos modelos do retreino no dataset de teste cego, comparando MSE e RMSE do CART e da RN.

## Métricas

**MSE de fold**:
Erro quadrático médio das predições de sobr em um fold (treino ou validação).

**MSE médio**:
Média dos MSE de fold sobre os k folds, reportada com seu dpa.

**dpa**:
Desvio padrão amostral (divisor k−1) dos valores por fold.
_Avoid_: desvio padrão (sem qualificar), std

**Média das difs**:
Média sobre os folds de |MSE de treino − MSE de validação|; mede a distância entre treino e validação.
