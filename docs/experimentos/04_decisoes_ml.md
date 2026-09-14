# 04 — Decisões de Machine Learning

## Objetivo
Garantir comparação justa entre métodos de seleção sem transformar HPO no foco principal do TCC.

## HPO
Ajustes ocorrem somente no TRAIN.

Preferência operacional:
1. validar custo do Grid Search existente;
2. se necessário, usar RandomizedSearchCV com orçamento explícito;
3. Optuna somente se houver necessidade documentada.

Evitar HPO excessivo.

## Inner CV
Preferir Stratified 3-Fold.

Não substituir automaticamente por holdout 80/20, principalmente em Colon (`N=62`).

## RFECV
Manter como representante de Wrapper.

Para reduzir custo, avaliar:
- `step=0.10`;
- RF com `n_estimators=50`;
- inner CV=3.

Qualquer alteração deve ser registrada.

## Lasso
Manter Logistic Regression + L1 conforme protocolo.

## ANOVA
Manter F-score.

A porcentagem de atributos deve ser tratada como decisão interna e nunca definida usando o TEST.

## Classificadores
Manter:
- RF
- KNN
- SVM
- XGBoost
- CatBoost
- OPF

Não substituir OPF silenciosamente.

## Reprodutibilidade
Registrar:
- random_state/seed;
- versões;
- parâmetros;
- tempo;
- falhas;
- quantidade final de atributos.

## Paralelização
Controlar threads para evitar oversubscription.

Exemplo válido:
```text
processos externos > 1
modelos internos n_jobs = 1
```

ou configuração equivalente explicitamente registrada.
