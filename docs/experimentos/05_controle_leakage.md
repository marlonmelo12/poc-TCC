# 05 — Controle de Leakage

## Princípio
O outer TEST representa dados nunca vistos.

## Permitido no TRAIN
- fit de scaler;
- filtro de variância/MAD;
- MI;
- Spearman;
- construção do QUBO;
- escolha de K;
- HPO;
- seleção ANOVA;
- RFECV;
- Lasso;
- treinamento.

## Proibido antes do outer split
- normalização global;
- remoção global de features;
- MI em todos os dados;
- correlação em todos os dados;
- seleção em todos os dados;
- escolha de hiperparâmetros baseada em todos os dados.

## Estrutura correta

```text
X, y
↓
outer split
├── X_train, y_train
│   └── tudo que aprende parâmetros
└── X_test
    └── somente transformações já ajustadas + predict
```

## Testes obrigatórios
Criar testes que garantam que transformadores/seletores não recebem `X_test` durante `fit`.

## Atenção
Um pipeline pode estar sintaticamente correto e ainda possuir leakage lógico. Auditar o fluxo de dados, não apenas a API do sklearn.

## Critério de aprovação
Não executar experimento definitivo enquanto o fluxo TRAIN→TEST não estiver demonstravelmente isolado.
