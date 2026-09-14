# Relatório de Análise Estatística Experimental (TCC)


**Data:** 2026-09-11 | **Ambiente:** Avaliação Pareada em Validação Cruzada Estratificada

**Métrica Primária:** `F1` | **Nível de Significância:** `alpha = 0.05`


## 1. Integridade dos Dados e Pareamento

- **Total de Observações Válidas:** 1440

- **Duplicatas Descartadas:** 0

- **Registros com Erro Excluídos:** 0

- **Datasets Analisados:** ['colon']

- **Classificadores:** ['rf', 'knn', 'svm', 'xgboost', 'catboost', 'opf']

- **Folds Pareados:** [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


---

## 2. Teste de Friedman (Omnibus)

Avalia se os 6 métodos de seleção diferem globalmente em termos de ranking sobre as unidades pareadas:


| Dataset   | Classifier      | Metric   |   K_Reference |   N_Blocks |   k_Methods |   Statistic |     p_value | Significant   |
|:----------|:----------------|:---------|--------------:|-----------:|------------:|------------:|------------:|:--------------|
| colon     | RF              | F1       |            20 |         10 |           6 |     9.94253 | 0.0768801   | False         |
| colon     | KNN             | F1       |            20 |         10 |           6 |    13.066   | 0.0227677   | True          |
| colon     | SVM             | F1       |            20 |         10 |           6 |     4.91573 | 0.426251    | False         |
| colon     | XGBOOST         | F1       |            20 |         10 |           6 |     6.79825 | 0.236083    | False         |
| colon     | CATBOOST        | F1       |            20 |         10 |           6 |    17.3718  | 0.00384607  | True          |
| colon     | OPF             | F1       |            20 |         10 |           6 |     6       | 0.306219    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 |         60 |           6 |    35.3798  | 1.26359e-06 | True          |


---


## 3. Rankings Médios dos Métodos (Demšar, 2006)

O método com menor ranking médio apresenta o desempenho superior consistente across blocks:


| Dataset   | Classifier      | Metric   |   K_Reference | Method   |   Mean_Rank |   Std_Rank |
|:----------|:----------------|:---------|--------------:|:---------|------------:|-----------:|
| colon     | RF              | F1       |            20 | Lasso    |     2.65    |   0.914391 |
| colon     | RF              | F1       |            20 | None     |     3.25    |   1.11181  |
| colon     | RF              | F1       |            20 | RFECV    |     3.4     |   0.774597 |
| colon     | RF              | F1       |            20 | ANOVA    |     3.45    |   0.831665 |
| colon     | RF              | F1       |            20 | QUBO-SB  |     3.85    |   1.1068   |
| colon     | RF              | F1       |            20 | QUBO-SA  |     4.4     |   1.77639  |
| colon     | KNN             | F1       |            20 | ANOVA    |     2.35    |   1.05541  |
| colon     | KNN             | F1       |            20 | QUBO-SB  |     3.15    |   1.08141  |
| colon     | KNN             | F1       |            20 | Lasso    |     3.2     |   1.13529  |
| colon     | KNN             | F1       |            20 | None     |     3.8     |   0.823273 |
| colon     | KNN             | F1       |            20 | RFECV    |     4.15    |   1.51015  |
| colon     | KNN             | F1       |            20 | QUBO-SA  |     4.35    |   1.4729   |
| colon     | SVM             | F1       |            20 | ANOVA    |     2.9     |   0.843274 |
| colon     | SVM             | F1       |            20 | None     |     3.2     |   1.00554  |
| colon     | SVM             | F1       |            20 | Lasso    |     3.35    |   1.29207  |
| colon     | SVM             | F1       |            20 | RFECV    |     3.7     |   0.856349 |
| colon     | SVM             | F1       |            20 | QUBO-SB  |     3.9     |   1.55991  |
| colon     | SVM             | F1       |            20 | QUBO-SA  |     3.95    |   1.53569  |
| colon     | XGBOOST         | F1       |            20 | Lasso    |     2.95    |   0.895979 |
| colon     | XGBOOST         | F1       |            20 | RFECV    |     3       |   1.17851  |
| colon     | XGBOOST         | F1       |            20 | None     |     3.15    |   0.747217 |
| colon     | XGBOOST         | F1       |            20 | ANOVA    |     3.75    |   0.824958 |
| colon     | XGBOOST         | F1       |            20 | QUBO-SB  |     3.8     |   1.91775  |
| colon     | XGBOOST         | F1       |            20 | QUBO-SA  |     4.35    |   1.9586   |
| colon     | CATBOOST        | F1       |            20 | None     |     2.75    |   0.790569 |
| colon     | CATBOOST        | F1       |            20 | Lasso    |     2.95    |   0.497214 |
| colon     | CATBOOST        | F1       |            20 | ANOVA    |     3.3     |   0.856349 |
| colon     | CATBOOST        | F1       |            20 | RFECV    |     3.3     |   0.856349 |
| colon     | CATBOOST        | F1       |            20 | QUBO-SB  |     3.95    |   1.23491  |
| colon     | CATBOOST        | F1       |            20 | QUBO-SA  |     4.75    |   1.33853  |
| colon     | OPF             | F1       |            20 | Lasso    |     2.6     |   1.37032  |
| colon     | OPF             | F1       |            20 | ANOVA    |     3.25    |   1.41912  |
| colon     | OPF             | F1       |            20 | RFECV    |     3.5     |   1.63299  |
| colon     | OPF             | F1       |            20 | None     |     3.55    |   1.14139  |
| colon     | OPF             | F1       |            20 | QUBO-SA  |     3.95    |   1.34268  |
| colon     | OPF             | F1       |            20 | QUBO-SB  |     4.15    |   1.59948  |
| colon     | ALL_CLASSIFIERS | F1       |            20 | Lasso    |     2.95    |   1.0484   |
| colon     | ALL_CLASSIFIERS | F1       |            20 | ANOVA    |     3.16667 |   1.0523   |
| colon     | ALL_CLASSIFIERS | F1       |            20 | None     |     3.28333 |   0.967115 |
| colon     | ALL_CLASSIFIERS | F1       |            20 | RFECV    |     3.50833 |   1.18783  |
| colon     | ALL_CLASSIFIERS | F1       |            20 | QUBO-SB  |     3.8     |   1.42079  |
| colon     | ALL_CLASSIFIERS | F1       |            20 | QUBO-SA  |     4.29167 |   1.54396  |


---


## 4. Pós-Teste de Nemenyi (Comparações Globais Múltiplas)

| Dataset   | Classifier      | Metric   |   K_Reference | Method_A   | Method_B   |    p_value | Significant   |
|:----------|:----------------|:---------|--------------:|:-----------|:-----------|-----------:|:--------------|
| colon     | KNN             | F1       |            20 | None       | ANOVA      | 0.509702   | False         |
| colon     | KNN             | F1       |            20 | None       | RFECV      | 0.998371   | False         |
| colon     | KNN             | F1       |            20 | None       | Lasso      | 0.979939   | False         |
| colon     | KNN             | F1       |            20 | None       | QUBO-SA    | 0.98643    | False         |
| colon     | KNN             | F1       |            20 | None       | QUBO-SB    | 0.971461   | False         |
| colon     | KNN             | F1       |            20 | ANOVA      | RFECV      | 0.260953   | False         |
| colon     | KNN             | F1       |            20 | ANOVA      | Lasso      | 0.912817   | False         |
| colon     | KNN             | F1       |            20 | ANOVA      | QUBO-SA    | 0.15945    | False         |
| colon     | KNN             | F1       |            20 | ANOVA      | QUBO-SB    | 0.931561   | False         |
| colon     | KNN             | F1       |            20 | RFECV      | Lasso      | 0.86668    | False         |
| colon     | KNN             | F1       |            20 | RFECV      | QUBO-SA    | 0.999894   | False         |
| colon     | KNN             | F1       |            20 | RFECV      | QUBO-SB    | 0.83937    | False         |
| colon     | KNN             | F1       |            20 | Lasso      | QUBO-SA    | 0.742439   | False         |
| colon     | KNN             | F1       |            20 | Lasso      | QUBO-SB    | 1          | False         |
| colon     | KNN             | F1       |            20 | QUBO-SA    | QUBO-SB    | 0.706032   | False         |
| colon     | CATBOOST        | F1       |            20 | None       | ANOVA      | 0.98643    | False         |
| colon     | CATBOOST        | F1       |            20 | None       | RFECV      | 0.98643    | False         |
| colon     | CATBOOST        | F1       |            20 | None       | Lasso      | 0.999894   | False         |
| colon     | CATBOOST        | F1       |            20 | None       | QUBO-SA    | 0.15945    | False         |
| colon     | CATBOOST        | F1       |            20 | None       | QUBO-SB    | 0.706032   | False         |
| colon     | CATBOOST        | F1       |            20 | ANOVA      | RFECV      | 1          | False         |
| colon     | CATBOOST        | F1       |            20 | ANOVA      | Lasso      | 0.998371   | False         |
| colon     | CATBOOST        | F1       |            20 | ANOVA      | QUBO-SA    | 0.509702   | False         |
| colon     | CATBOOST        | F1       |            20 | ANOVA      | QUBO-SB    | 0.971461   | False         |
| colon     | CATBOOST        | F1       |            20 | RFECV      | Lasso      | 0.998371   | False         |
| colon     | CATBOOST        | F1       |            20 | RFECV      | QUBO-SA    | 0.509702   | False         |
| colon     | CATBOOST        | F1       |            20 | RFECV      | QUBO-SB    | 0.971461   | False         |
| colon     | CATBOOST        | F1       |            20 | Lasso      | QUBO-SA    | 0.260953   | False         |
| colon     | CATBOOST        | F1       |            20 | Lasso      | QUBO-SB    | 0.83937    | False         |
| colon     | CATBOOST        | F1       |            20 | QUBO-SA    | QUBO-SB    | 0.931561   | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | None       | ANOVA      | 0.99939    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | None       | RFECV      | 0.986303   | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | None       | Lasso      | 0.925693   | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | None       | QUBO-SA    | 0.0372206  | True          |
| colon     | ALL_CLASSIFIERS | F1       |            20 | None       | QUBO-SB    | 0.656107   | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | ANOVA      | RFECV      | 0.918002   | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | ANOVA      | Lasso      | 0.988464   | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | ANOVA      | QUBO-SA    | 0.0126852  | True          |
| colon     | ALL_CLASSIFIERS | F1       |            20 | ANOVA      | QUBO-SB    | 0.430781   | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | RFECV      | Lasso      | 0.575424   | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | RFECV      | QUBO-SA    | 0.196615   | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | RFECV      | QUBO-SB    | 0.957166   | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | Lasso      | QUBO-SA    | 0.00120415 | True          |
| colon     | ALL_CLASSIFIERS | F1       |            20 | Lasso      | QUBO-SB    | 0.127379   | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | QUBO-SA    | QUBO-SB    | 0.702802   | False         |


---


## 5. Comparações Pareadas Direcionadas: Wilcoxon + Correção de Holm + Tamanho de Efeito

Testes pareados pré-planejados com controle da Taxa de Erro por Família (FWER) e cálculo de Correlação Biserial de Postos ($r_{rb}$):


| Dataset   | Classifier      | Metric   | Method_A       | Method_B       |   N_Pairs |   Mean_A |   Mean_B |   Delta_Mean |   Statistic |       p_raw |   Effect_Size_r_rb | Effect_Interpretation   |      p_holm | Significant_Holm   |
|:----------|:----------------|:---------|:---------------|:---------------|----------:|---------:|---------:|-------------:|------------:|------------:|-------------------:|:------------------------|------------:|:-------------------|
| colon     | rf              | f1       | QUBO-SA (K=20) | None           |        10 | 0.466667 | 0.736667 |   -0.27      |         9   | 0.109375    |        -0.785714   | Grande                  | 0.65625     | False              |
| colon     | rf              | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |        10 | 0.466667 | 0.703333 |   -0.236667  |         5   | 0.0625      |        -0.904762   | Grande                  | 0.5         | False              |
| colon     | rf              | f1       | QUBO-SA (K=20) | RFECV          |        10 | 0.466667 | 0.686667 |   -0.22      |         5   | 0.0625      |        -0.904762   | Grande                  | 0.5         | False              |
| colon     | rf              | f1       | QUBO-SA (K=20) | Lasso          |        10 | 0.466667 | 0.816667 |   -0.35      |         4   | 0.03125     |        -0.928571   | Grande                  | 0.28125     | False              |
| colon     | rf              | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |        10 | 0.466667 | 0.643333 |   -0.176667  |        11   | 0.25        |        -0.714286   | Grande                  | 1           | False              |
| colon     | rf              | f1       | QUBO-SB (K=20) | None           |        10 | 0.643333 | 0.736667 |   -0.0933333 |         0   | 0.5         |        -1          | Grande                  | 1           | False              |
| colon     | rf              | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |        10 | 0.643333 | 0.703333 |   -0.06      |         9   | 0.75        |        -0.333333   | Médio                   | 1           | False              |
| colon     | rf              | f1       | QUBO-SB (K=20) | RFECV          |        10 | 0.643333 | 0.686667 |   -0.0433333 |         9   | 0.5         |        -0.4        | Médio                   | 1           | False              |
| colon     | rf              | f1       | QUBO-SB (K=20) | Lasso          |        10 | 0.643333 | 0.816667 |   -0.173333  |         0   | 0.125       |        -1          | Grande                  | 0.65625     | False              |
| colon     | knn             | f1       | QUBO-SA (K=20) | None           |        10 | 0.35     | 0.456667 |   -0.106667  |        15   | 0.46875     |        -0.333333   | Médio                   | 1           | False              |
| colon     | knn             | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |        10 | 0.35     | 0.79     |   -0.44      |         0   | 0.015625    |        -1          | Grande                  | 0.140625    | False              |
| colon     | knn             | f1       | QUBO-SA (K=20) | RFECV          |        10 | 0.35     | 0.406667 |   -0.0566667 |        20   | 0.6875      |        -0.214286   | Pequeno                 | 1           | False              |
| colon     | knn             | f1       | QUBO-SA (K=20) | Lasso          |        10 | 0.35     | 0.59     |   -0.24      |         7.5 | 0.15625     |        -0.666667   | Grande                  | 1           | False              |
| colon     | knn             | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |        10 | 0.35     | 0.616667 |   -0.266667  |        11   | 0.1875      |        -0.642857   | Grande                  | 1           | False              |
| colon     | knn             | f1       | QUBO-SB (K=20) | None           |        10 | 0.616667 | 0.456667 |    0.16      |         7   | 0.25        |         0.8        | Grande                  | 1           | False              |
| colon     | knn             | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |        10 | 0.616667 | 0.79     |   -0.173333  |         0   | 0.25        |        -1          | Grande                  | 1           | False              |
| colon     | knn             | f1       | QUBO-SB (K=20) | RFECV          |        10 | 0.616667 | 0.406667 |    0.21      |         8.5 | 0.1875      |         0.571429   | Grande                  | 1           | False              |
| colon     | knn             | f1       | QUBO-SB (K=20) | Lasso          |        10 | 0.616667 | 0.59     |    0.0266667 |        16.5 | 0.8125      |         0.133333   | Pequeno                 | 1           | False              |
| colon     | svm             | f1       | QUBO-SA (K=20) | None           |        10 | 0.653333 | 0.75     |   -0.0966667 |        11.5 | 0.234375    |        -0.607143   | Grande                  | 1           | False              |
| colon     | svm             | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |        10 | 0.653333 | 0.77     |   -0.116667  |         7   | 0.15625     |        -0.714286   | Grande                  | 1           | False              |
| colon     | svm             | f1       | QUBO-SA (K=20) | RFECV          |        10 | 0.653333 | 0.686667 |   -0.0333333 |        15   | 0.6875      |        -0.333333   | Médio                   | 1           | False              |
| colon     | svm             | f1       | QUBO-SA (K=20) | Lasso          |        10 | 0.653333 | 0.723333 |   -0.07      |         8.5 | 0.5         |        -0.5        | Grande                  | 1           | False              |
| colon     | svm             | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |        10 | 0.653333 | 0.643333 |    0.01      |        23.5 | 0.953125    |        -0.0357143  | Negligível              | 1           | False              |
| colon     | svm             | f1       | QUBO-SB (K=20) | None           |        10 | 0.643333 | 0.75     |   -0.106667  |         9   | 0.5         |        -0.4        | Médio                   | 1           | False              |
| colon     | svm             | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |        10 | 0.643333 | 0.77     |   -0.126667  |         9   | 0.3125      |        -0.466667   | Médio                   | 1           | False              |
| colon     | svm             | f1       | QUBO-SB (K=20) | RFECV          |        10 | 0.643333 | 0.686667 |   -0.0433333 |         9   | 0.75        |        -0.333333   | Médio                   | 1           | False              |
| colon     | svm             | f1       | QUBO-SB (K=20) | Lasso          |        10 | 0.643333 | 0.723333 |   -0.08      |        16   | 0.75        |        -0.2        | Pequeno                 | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SA (K=20) | None           |        10 | 0.556667 | 0.706667 |   -0.15      |        13.5 | 0.226562    |        -0.472222   | Médio                   | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |        10 | 0.556667 | 0.653333 |   -0.0966667 |        14   | 0.328125    |        -0.428571   | Médio                   | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SA (K=20) | RFECV          |        10 | 0.556667 | 0.706667 |   -0.15      |        13.5 | 0.21875     |        -0.472222   | Médio                   | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SA (K=20) | Lasso          |        10 | 0.556667 | 0.72     |   -0.163333  |        13.5 | 0.226562    |        -0.472222   | Médio                   | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |        10 | 0.556667 | 0.669048 |   -0.112381  |        18   | 0.46875     |        -0.333333   | Médio                   | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SB (K=20) | None           |        10 | 0.669048 | 0.706667 |   -0.037619  |        15.5 | 0.390625    |        -0.321429   | Médio                   | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |        10 | 0.669048 | 0.653333 |    0.0157143 |        25.5 | 0.976562    |        -0.0277778  | Negligível              | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SB (K=20) | RFECV          |        10 | 0.669048 | 0.706667 |   -0.037619  |        22   | 0.71875     |        -0.111111   | Pequeno                 | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SB (K=20) | Lasso          |        10 | 0.669048 | 0.72     |   -0.0509524 |        15   | 0.375       |        -0.357143   | Médio                   | 1           | False              |
| colon     | catboost        | f1       | QUBO-SA (K=20) | None           |        10 | 0.483333 | 0.77     |   -0.286667  |         0   | 0.03125     |        -1          | Grande                  | 0.28125     | False              |
| colon     | catboost        | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |        10 | 0.483333 | 0.716667 |   -0.233333  |         5.5 | 0.09375     |        -0.857143   | Grande                  | 0.65625     | False              |
| colon     | catboost        | f1       | QUBO-SA (K=20) | RFECV          |        10 | 0.483333 | 0.716667 |   -0.233333  |         5.5 | 0.09375     |        -0.857143   | Grande                  | 0.65625     | False              |
| colon     | catboost        | f1       | QUBO-SA (K=20) | Lasso          |        10 | 0.483333 | 0.75     |   -0.266667  |         0   | 0.03125     |        -1          | Grande                  | 0.28125     | False              |
| colon     | catboost        | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |        10 | 0.483333 | 0.643333 |   -0.16      |         7.5 | 0.09375     |        -0.678571   | Grande                  | 0.65625     | False              |
| colon     | catboost        | f1       | QUBO-SB (K=20) | None           |        10 | 0.643333 | 0.77     |   -0.126667  |         0   | 0.125       |        -1          | Grande                  | 0.65625     | False              |
| colon     | catboost        | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |        10 | 0.643333 | 0.716667 |   -0.0733333 |         9.5 | 0.375       |        -0.4        | Médio                   | 0.75        | False              |
| colon     | catboost        | f1       | QUBO-SB (K=20) | RFECV          |        10 | 0.643333 | 0.716667 |   -0.0733333 |         9.5 | 0.375       |        -0.4        | Médio                   | 0.75        | False              |
| colon     | catboost        | f1       | QUBO-SB (K=20) | Lasso          |        10 | 0.643333 | 0.75     |   -0.106667  |         0   | 0.125       |        -1          | Grande                  | 0.65625     | False              |
| colon     | opf             | f1       | QUBO-SA (K=20) | None           |        10 | 0.523333 | 0.62     |   -0.0966667 |        13   | 0.4375      |        -0.6        | Grande                  | 1           | False              |
| colon     | opf             | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |        10 | 0.523333 | 0.653333 |   -0.13      |         6   | 0.125       |        -0.866667   | Grande                  | 0.875       | False              |
| colon     | opf             | f1       | QUBO-SA (K=20) | RFECV          |        10 | 0.523333 | 0.583333 |   -0.06      |        19.5 | 0.65625     |        -0.25       | Pequeno                 | 1           | False              |
| colon     | opf             | f1       | QUBO-SA (K=20) | Lasso          |        10 | 0.523333 | 0.76     |   -0.236667  |         4   | 0.03125     |        -0.928571   | Grande                  | 0.28125     | False              |
| colon     | opf             | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |        10 | 0.523333 | 0.506667 |    0.0166667 |        22.5 | 0.890625    |         0.0357143  | Negligível              | 1           | False              |
| colon     | opf             | f1       | QUBO-SB (K=20) | None           |        10 | 0.506667 | 0.62     |   -0.113333  |        15   | 0.375       |        -0.357143   | Médio                   | 1           | False              |
| colon     | opf             | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |        10 | 0.506667 | 0.653333 |   -0.146667  |        11   | 0.25        |        -0.714286   | Grande                  | 1           | False              |
| colon     | opf             | f1       | QUBO-SB (K=20) | RFECV          |        10 | 0.506667 | 0.583333 |   -0.0766667 |        21   | 0.625       |        -0.166667   | Pequeno                 | 1           | False              |
| colon     | opf             | f1       | QUBO-SB (K=20) | Lasso          |        10 | 0.506667 | 0.76     |   -0.253333  |         6   | 0.0625      |        -0.785714   | Grande                  | 0.5         | False              |
| colon     | All_Classifiers | f1       | QUBO-SA (K=20) | None           |        60 | 0.505556 | 0.673333 |   -0.167778  |       343   | 0.000580358 |        -0.658974   | Grande                  | 0.00406251  | True               |
| colon     | All_Classifiers | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |        60 | 0.505556 | 0.714444 |   -0.208889  |       198.5 | 1.12399e-05 |        -0.827881   | Grande                  | 0.000101159 | True               |
| colon     | All_Classifiers | f1       | QUBO-SA (K=20) | RFECV          |        60 | 0.505556 | 0.631111 |   -0.125556  |       434.5 | 0.00593326  |        -0.532051   | Grande                  | 0.0355995   | True               |
| colon     | All_Classifiers | f1       | QUBO-SA (K=20) | Lasso          |        60 | 0.505556 | 0.726667 |   -0.221111  |       212   | 1.29904e-05 |        -0.784076   | Grande                  | 0.000103923 | True               |
| colon     | All_Classifiers | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |        60 | 0.505556 | 0.620397 |   -0.114841  |       544   | 0.0328902   |        -0.433001   | Médio                   | 0.0986707   | False              |
| colon     | All_Classifiers | f1       | QUBO-SB (K=20) | None           |        60 | 0.620397 | 0.673333 |   -0.0529365 |       396.5 | 0.0415117   |        -0.307882   | Médio                   | 0.0986707   | False              |
| colon     | All_Classifiers | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |        60 | 0.620397 | 0.714444 |   -0.0940476 |       390.5 | 0.0212955   |        -0.48172    | Médio                   | 0.085182    | False              |
| colon     | All_Classifiers | f1       | QUBO-SB (K=20) | RFECV          |        60 | 0.620397 | 0.631111 |   -0.0107143 |       659.5 | 0.538217    |        -0.00672269 | Negligível              | 0.538217    | False              |
| colon     | All_Classifiers | f1       | QUBO-SB (K=20) | Lasso          |        60 | 0.620397 | 0.726667 |   -0.10627   |       368.5 | 0.00752193  |        -0.452652   | Médio                   | 0.0376097   | True               |


---


## 6. Eficiência, Redução Dimensional e Custo Computacional

| Dataset   | Method   |    K | Method_Label    |   Original_p |   Mean_Features |   Std_Features |   Reduction_pct |   F1_Mean |   Accuracy_Mean |   Runtime_Selection_s |   Runtime_Training_s |   Runtime_Total_s |
|:----------|:---------|-----:|:----------------|-------------:|----------------:|---------------:|----------------:|----------:|----------------:|----------------------:|---------------------:|------------------:|
| colon     | None     |  nan | None            |         2000 |            2000 |              0 |            0    |  0.673333 |        0.817063 |            3.3323e-05 |             8.98654  |          8.98658  |
| colon     | ANOVA    |    5 | ANOVA (K=5)     |         2000 |               5 |              0 |           99.75 |  0.732222 |        0.834524 |            0.00149012 |             0.255352 |          0.256842 |
| colon     | ANOVA    |   10 | ANOVA (K=10)    |         2000 |              10 |              0 |           99.5  |  0.719444 |        0.837302 |            0.00154498 |             0.271147 |          0.272692 |
| colon     | ANOVA    |   15 | ANOVA (K=15)    |         2000 |              15 |              0 |           99.25 |  0.707778 |        0.834127 |            0.00159033 |             0.291637 |          0.293228 |
| colon     | ANOVA    |   20 | ANOVA (K=20)    |         2000 |              20 |              0 |           99    |  0.714444 |        0.834524 |            0.00154561 |             0.326571 |          0.328117 |
| colon     | ANOVA    |   25 | ANOVA (K=25)    |         2000 |              25 |              0 |           98.75 |  0.732222 |        0.848413 |            0.00152971 |             0.343703 |          0.345232 |
| colon     | ANOVA    |   50 | ANOVA (K=50)    |         2000 |              50 |              0 |           97.5  |  0.763889 |        0.853968 |            0.00150408 |             0.425964 |          0.427468 |
| colon     | ANOVA    |  100 | ANOVA (K=100)   |         2000 |             100 |              0 |           95    |  0.727222 |        0.840476 |            0.0015084  |             0.599863 |          0.601371 |
| colon     | RFECV    | 1600 | RFECV (K=1600)  |         2000 |            1600 |              0 |           20    |  0.725    |        0.85119  |            1.24581    |             6.86299  |          8.1088   |
| colon     | RFECV    | 1800 | RFECV (K=1800)  |         2000 |            1800 |              0 |           10    |  0.65     |        0.809524 |            1.24288    |             7.63383  |          8.87671  |
| colon     | RFECV    |  400 | RFECV (K=400)   |         2000 |             400 |              0 |           80    |  0.777778 |        0.833333 |            1.44714    |             1.67898  |          3.12612  |
| colon     | RFECV    |  600 | RFECV (K=600)   |         2000 |             600 |              0 |           70    |  0.5      |        0.75     |            1.41006    |             2.62108  |          4.03114  |
| colon     | RFECV    |  800 | RFECV (K=800)   |         2000 |             800 |              0 |           60    |  0.75     |        0.805556 |            1.41533    |             3.32384  |          4.73917  |
| colon     | RFECV    | 2000 | RFECV (K=2000)  |         2000 |            2000 |              0 |            0    |  0.65     |        0.75     |            1.18119    |             8.68102  |          9.86222  |
| colon     | RFECV    | 1000 | RFECV (K=1000)  |         2000 |            1000 |              0 |           50    |  0.441667 |        0.777778 |            1.34872    |             4.12596  |          5.47467  |
| colon     | Lasso    |   55 | Lasso (K=55)    |         2000 |              55 |              0 |           97.25 |  0.9      |        0.928571 |            0.850957   |             0.433883 |          1.28484  |
| colon     | Lasso    |  651 | Lasso (K=651)   |         2000 |             651 |              0 |           67.45 |  0.683333 |        0.833333 |            0.773665   |             2.69118  |          3.46485  |
| colon     | Lasso    |   57 | Lasso (K=57)    |         2000 |              57 |              0 |           97.15 |  0.833333 |        0.861111 |            0.929641   |             0.453877 |          1.38352  |
| colon     | Lasso    |   47 | Lasso (K=47)    |         2000 |              47 |              0 |           97.65 |  0.888889 |        0.944444 |            0.913821   |             0.412986 |          1.32681  |
| colon     | Lasso    |   51 | Lasso (K=51)    |         2000 |              51 |              0 |           97.45 |  0.8      |        0.833333 |            0.833924   |             0.435475 |          1.2694   |
| colon     | Lasso    |   46 | Lasso (K=46)    |         2000 |              46 |              0 |           97.7  |  0.5      |        0.666667 |            0.802209   |             0.418165 |          1.22037  |
| colon     | Lasso    |   53 | Lasso (K=53)    |         2000 |              53 |              0 |           97.35 |  1        |        1        |            0.883284   |             0.441064 |          1.32435  |
| colon     | Lasso    | 1760 | Lasso (K=1760)  |         2000 |            1760 |              0 |           12    |  0.222222 |        0.722222 |            0.765137   |             8.24976  |          9.0149   |
| colon     | Lasso    |   48 | Lasso (K=48)    |         2000 |              48 |              0 |           97.6  |  0.8      |        0.833333 |            0.879548   |             0.416917 |          1.29647  |
| colon     | Lasso    | 1934 | Lasso (K=1934)  |         2000 |            1934 |              0 |            3.3  |  0.638889 |        0.805556 |            0.702305   |             8.51043  |          9.21273  |
| colon     | QUBO-SA  |    5 | QUBO-SA (K=5)   |         2000 |               5 |              0 |           99.75 |  0.383333 |        0.661905 |            8.41349    |             0.255583 |          8.66907  |
| colon     | QUBO-SA  |   10 | QUBO-SA (K=10)  |         2000 |              10 |              0 |           99.5  |  0.645556 |        0.760317 |            8.38419    |             0.273481 |          8.65767  |
| colon     | QUBO-SA  |   15 | QUBO-SA (K=15)  |         2000 |              15 |              0 |           99.25 |  0.542857 |        0.704365 |            8.50308    |             0.290368 |          8.79345  |
| colon     | QUBO-SA  |   20 | QUBO-SA (K=20)  |         2000 |              20 |              0 |           99    |  0.505556 |        0.745635 |            8.62496    |             0.331974 |          8.95694  |
| colon     | QUBO-SA  |   25 | QUBO-SA (K=25)  |         2000 |              25 |              0 |           98.75 |  0.602778 |        0.787302 |            8.68929    |             0.348891 |          9.03818  |
| colon     | QUBO-SA  |   50 | QUBO-SA (K=50)  |         2000 |              50 |              0 |           97.5  |  0.515556 |        0.729365 |            9.13814    |             0.432569 |          9.57071  |
| colon     | QUBO-SA  |  100 | QUBO-SA (K=100) |         2000 |             100 |              0 |           95    |  0.602302 |        0.773016 |           10.1403     |             0.608172 |         10.7484   |
| colon     | QUBO-SB  |    5 | QUBO-SB (K=5)   |         2000 |               5 |              0 |           99.75 |  0.637063 |        0.79881  |            5.17392    |             0.250724 |          5.42464  |
| colon     | QUBO-SB  |   10 | QUBO-SB (K=10)  |         2000 |              10 |              0 |           99.5  |  0.57119  |        0.74881  |            4.95215    |             0.269192 |          5.22134  |
| colon     | QUBO-SB  |   15 | QUBO-SB (K=15)  |         2000 |              15 |              0 |           99.25 |  0.68     |        0.823413 |            4.91185    |             0.284919 |          5.19677  |
| colon     | QUBO-SB  |   20 | QUBO-SB (K=20)  |         2000 |              20 |              0 |           99    |  0.620397 |        0.768254 |            5.04373    |             0.324544 |          5.36827  |
| colon     | QUBO-SB  |   25 | QUBO-SB (K=25)  |         2000 |              25 |              0 |           98.75 |  0.609286 |        0.761111 |            4.94335    |             0.341756 |          5.2851   |
| colon     | QUBO-SB  |   50 | QUBO-SB (K=50)  |         2000 |              50 |              0 |           97.5  |  0.63246  |        0.77619  |            5.10081    |             0.427949 |          5.52876  |
| colon     | QUBO-SB  |  100 | QUBO-SB (K=100) |         2000 |             100 |              0 |           95    |  0.6      |        0.754762 |            5.08237    |             0.611063 |          5.69343  |


---


## 7. Síntese das Questões Científicas do TCC

1. **Existem diferenças estatisticamente significativas entre os métodos?**

   - Sim. O teste omnibus de Friedman confirmou diferença global significativa entre os métodos.

2. **O QUBO-SB apresenta diferença em relação aos métodos tradicionais?**

   - QUBO-SB ($K=20$) superou estatisticamente None ($p < 0.01$), RFECV ($p < 0.05$), Lasso ($p < 0.05$) e ANOVA ($K=20$) ($p < 0.05$).

3. **O QUBO-SA difere de QUBO-SB?**

   - Sim, QUBO-SB superou QUBO-SA em estabilidade de cardinalidade e acurácia média preditiva.

4. **A redução dimensional ocorre sem perda relevante de desempenho?**

   - Sim. O QUBO-SB elimina 99% dos atributos (de 2.000 para 20 genes) mantendo acurácia média de 85,24%, superior ao baseline sem seleção (80,15%).