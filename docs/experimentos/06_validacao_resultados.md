# 06 — Validação de Resultados

## Integridade
Antes de aceitar qualquer resultado:

- shapes corretos;
- labels corretos;
- ausência de NaN/Inf após preprocessing;
- número de atributos válido;
- subset binário;
- K respeitado;
- métricas dentro dos limites;
- runtime registrado;
- seed registrada.

## Resultado mínimo por execução

```text
dataset
fold
method
classifier
seed
K
n_features
reduction_rate
accuracy
precision
recall
f1
auc
runtime
qubo_energy
```

Campos não aplicáveis devem ser explicitamente `null/NA`, nunca inventados.

## Consistência
Verificar:

```text
0 <= reduction_rate <= 1
0 <= accuracy <= 1
0 <= precision <= 1
0 <= recall <= 1
0 <= f1 <= 1
0 <= auc <= 1
```

## Agregação
Reportar média e desvio padrão por:

```text
dataset × classifier × method
```

Não esconder dispersão.

## QUBO
Além de desempenho preditivo:
- energia;
- número de features;
- redução;
- estabilidade;
- runtime.

## Estabilidade
Calcular Jaccard entre seeds quando houver múltiplas execuções.

## Auditoria
Resultados suspeitos devem ser investigados antes de serem removidos ou corrigidos.

Nunca alterar um resultado para "ficar coerente".
