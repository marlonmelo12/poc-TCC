# 02 — Protocolo Experimental

## Desenho principal

Para cada dataset:

```text
Dataset
└── Outer Stratified K-Fold (10)
    ├── TRAIN
    │   ├── preprocessamento
    │   ├── seleção
    │   ├── escolha de hiperparâmetros/K
    │   └── treinamento final
    └── TEST
        └── avaliação única
```

O TEST nunca participa de decisões.

## Métodos
1. None — baseline sem seleção.
2. ANOVA F-score.
3. RFECV.
4. Lasso.
5. QUBO + SA.

## Classificadores
RF, KNN, SVM, XGBoost, CatBoost e OPF.

## Métricas
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- `n_features`
- `reduction_rate`
- runtime

`reduction_rate = 1 - n_features / n_features_original`

## QUBO
O QUBO é calculado uma vez por `dataset × outer_fold × configuração QUBO`, não uma vez por classificador.

Depois, o mesmo subconjunto é fornecido aos classificadores.

## Inner CV
Usar inner Stratified 3-Fold para decisões internas/HPO, salvo justificativa registrada.

## HPO
Preferir orçamento controlado. Não trocar Grid Search por outro método sem registrar impacto metodológico.

## Resultado
Cada execução deve produzir registros persistentes e identificáveis por dataset, fold, método, classificador, seed e configuração.
