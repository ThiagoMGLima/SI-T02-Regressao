# MSE sem raiz e dpa com divisor k−1

A fórmula de MSE no enunciado tem uma raiz quadrada, o que daria o RMSE, mas a tabela 7 do próprio enunciado define RMSE = √MSE. Por isso tratamos a raiz como erro de digitação: o MSE é sempre a média dos erros quadráticos, sem raiz, e o RMSE só aparece no teste cego. O dpa usa divisor k−1 (`ddof=1`), que é o único que reproduz o exemplo numérico do enunciado (folds [0.0232, 0.0011, 0.0198] → dpa 0,0119) e é o que os notebooks de referência do professor usam.
