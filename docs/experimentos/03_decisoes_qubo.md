# 03 — Decisões QUBO

## Formulação

Para `x_i ∈ {0,1}`:

```text
E(x) =
    - α Σ_i r_i x_i
    + β Σ_{i<j} d_ij x_i x_j
    + λ (Σ_i x_i - K)^2
```

### Relevância
`r_i` = Mutual Information entre atributo `i` e o alvo, calculada exclusivamente no TRAIN.

Recomendação de normalização:

```text
r_norm = MinMax(MI)
```

### Redundância
`d_ij = |Spearman(X_i, X_j)|`.

A redundância deve representar dependência monotônica independentemente do sinal da correlação.

## Cardinalidade

A penalidade:

```text
λ (Σ_i x_i - K)^2
```

é parte da formulação e impede que a solução escolha quantidade arbitrária de atributos.

## K
Usar um conjunto pequeno, definido previamente. Exemplo:

```text
K ∈ {10, 20, 50, 100}
```

Se K for otimizado, a escolha ocorre somente no TRAIN/inner-CV.

## Pesos
Após normalização, testar poucos valores previamente definidos para `α`, `β` e `λ`. Não realizar HPO irrestrito do QUBO.

## Simulated Annealing
O valor atual de 1.000 iterações é tratado como hipótese, não como verdade.

Calibrar, quando possível:

```text
100, 500, 1000, 2000, 5000
```

e selecionar um orçamento justificável por energia, tempo e estabilidade.

## Seeds
Executar múltiplas seeds:

```text
42, 43, 44, 45, 46
```

quando o orçamento permitir.

## Estabilidade

Para subconjuntos Sa e Sb:

```text
Jaccard = |Sa ∩ Sb| / |Sa ∪ Sb|
```

Reportar estabilidade entre seeds.

## Testes de implementação
- Q é válida.
- Q é simétrica conforme a convenção adotada.
- Energia de soluções simples pode ser calculada manualmente.
- Solução contém apenas 0/1.
- Cardinalidade respeita K.
- Mesma seed + mesmo input produz comportamento reprodutível quando suportado pela implementação.

## Não fazer
- Usar TEST para escolher K.
- Usar TEST para escolher pesos.
- Escolher K observando o melhor resultado externo.
- Reexecutar QUBO por classificador sem justificativa.
- Chamar qualquer solução de "ótima" apenas porque possui menor energia encontrada pelo SA.
