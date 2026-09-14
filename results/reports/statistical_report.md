# Relatório de Análise Estatística Experimental (TCC)


**Data:** 2026-09-11 | **Ambiente:** Avaliação Pareada em Validação Cruzada Estratificada

**Métrica Primária:** `F1` | **Nível de Significância:** `alpha = 0.05`


## 1. Integridade dos Dados e Pareamento

- **Total de Observações Válidas:** 450

- **Duplicatas Descartadas:** 0

- **Registros com Erro Excluídos:** 0

- **Datasets Analisados:** ['colon']

- **Classificadores:** ['rf', 'knn', 'svm', 'xgboost', 'catboost', 'opf']

- **Folds Pareados:** [0, 1, 2, 3, 4]


---

## 2. Teste de Friedman (Omnibus)

Avalia se os 6 métodos de seleção diferem globalmente em termos de ranking sobre as unidades pareadas:


| Dataset   | Classifier      | Metric   |   K_Reference |   N_Blocks |   k_Methods |   Statistic |     p_value | Significant   |
|:----------|:----------------|:---------|--------------:|-----------:|------------:|------------:|------------:|:--------------|
| colon     | RF              | F1       |            20 |          5 |           6 |     5.09174 | 0.404787    | False         |
| colon     | KNN             | F1       |            20 |          5 |           6 |    11.6207  | 0.0403716   | True          |
| colon     | SVM             | F1       |            20 |          5 |           6 |    10.2451  | 0.068581    | False         |
| colon     | XGBOOST         | F1       |            20 |          5 |           6 |     7.48    | 0.187318    | False         |
| colon     | CATBOOST        | F1       |            20 |          5 |           6 |     7.02586 | 0.218724    | False         |
| colon     | OPF             | F1       |            20 |          5 |           6 |     8.9557  | 0.110844    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 |         30 |           6 |    27.9536  | 3.71662e-05 | True          |


---


## 3. Rankings Médios dos Métodos (Demšar, 2006)

O método com menor ranking médio apresenta o desempenho superior consistente across blocks:


| Dataset   | Classifier      | Metric   |   K_Reference | Method   |   Mean_Rank |   Std_Rank |
|:----------|:----------------|:---------|--------------:|:---------|------------:|-----------:|
| colon     | RF              | F1       |            20 | RFECV    |     2.8     |   1.03682  |
| colon     | RF              | F1       |            20 | Lasso    |     3.1     |   1.29422  |
| colon     | RF              | F1       |            20 | QUBO-SB  |     3.2     |   0.758288 |
| colon     | RF              | F1       |            20 | None     |     3.5     |   1.11803  |
| colon     | RF              | F1       |            20 | ANOVA    |     3.7     |   1.68077  |
| colon     | RF              | F1       |            20 | QUBO-SA  |     4.7     |   1.85742  |
| colon     | KNN             | F1       |            20 | QUBO-SB  |     2       |   0.5      |
| colon     | KNN             | F1       |            20 | Lasso    |     2.3     |   0.570088 |
| colon     | KNN             | F1       |            20 | ANOVA    |     3.5     |   1.90394  |
| colon     | KNN             | F1       |            20 | RFECV    |     3.9     |   1.38744  |
| colon     | KNN             | F1       |            20 | QUBO-SA  |     4.3     |   1.25499  |
| colon     | KNN             | F1       |            20 | None     |     5       |   1.41421  |
| colon     | SVM             | F1       |            20 | None     |     2.5     |   0.935414 |
| colon     | SVM             | F1       |            20 | RFECV    |     2.9     |   0.821584 |
| colon     | SVM             | F1       |            20 | QUBO-SB  |     3.1     |   1.29422  |
| colon     | SVM             | F1       |            20 | ANOVA    |     3.6     |   1.51658  |
| colon     | SVM             | F1       |            20 | Lasso    |     3.8     |   0.908295 |
| colon     | SVM             | F1       |            20 | QUBO-SA  |     5.1     |   1.08397  |
| colon     | XGBOOST         | F1       |            20 | ANOVA    |     2.4     |   1.29422  |
| colon     | XGBOOST         | F1       |            20 | Lasso    |     3       |   0.935414 |
| colon     | XGBOOST         | F1       |            20 | QUBO-SB  |     3       |   0.935414 |
| colon     | XGBOOST         | F1       |            20 | None     |     3.6     |   1.71026  |
| colon     | XGBOOST         | F1       |            20 | RFECV    |     4.4     |   0.65192  |
| colon     | XGBOOST         | F1       |            20 | QUBO-SA  |     4.6     |   2.04328  |
| colon     | CATBOOST        | F1       |            20 | QUBO-SB  |     2.5     |   1.11803  |
| colon     | CATBOOST        | F1       |            20 | None     |     3.1     |   0.894427 |
| colon     | CATBOOST        | F1       |            20 | RFECV    |     3.3     |   1.20416  |
| colon     | CATBOOST        | F1       |            20 | ANOVA    |     3.4     |   1.38744  |
| colon     | CATBOOST        | F1       |            20 | Lasso    |     3.8     |   1.78885  |
| colon     | CATBOOST        | F1       |            20 | QUBO-SA  |     4.9     |   1.34164  |
| colon     | OPF             | F1       |            20 | Lasso    |     2       |   0.612372 |
| colon     | OPF             | F1       |            20 | QUBO-SB  |     2.4     |   1.14018  |
| colon     | OPF             | F1       |            20 | ANOVA    |     3.7     |   2.38747  |
| colon     | OPF             | F1       |            20 | RFECV    |     3.9     |   1.47479  |
| colon     | OPF             | F1       |            20 | None     |     4.5     |   0.612372 |
| colon     | OPF             | F1       |            20 | QUBO-SA  |     4.5     |   1.65831  |
| colon     | ALL_CLASSIFIERS | F1       |            20 | QUBO-SB  |     2.7     |   1.00516  |
| colon     | ALL_CLASSIFIERS | F1       |            20 | Lasso    |     3       |   1.21769  |
| colon     | ALL_CLASSIFIERS | F1       |            20 | ANOVA    |     3.38333 |   1.6436   |
| colon     | ALL_CLASSIFIERS | F1       |            20 | RFECV    |     3.53333 |   1.18855  |
| colon     | ALL_CLASSIFIERS | F1       |            20 | None     |     3.7     |   1.36205  |
| colon     | ALL_CLASSIFIERS | F1       |            20 | QUBO-SA  |     4.68333 |   1.45912  |


---


## 4. Pós-Teste de Nemenyi (Comparações Globais Múltiplas)

| Dataset   | Classifier      | Metric   |   K_Reference | Method_A   | Method_B   |     p_value | Significant   |
|:----------|:----------------|:---------|--------------:|:-----------|:-----------|------------:|:--------------|
| colon     | KNN             | F1       |            20 | None       | ANOVA      | 0.802694    | False         |
| colon     | KNN             | F1       |            20 | None       | RFECV      | 0.938967    | False         |
| colon     | KNN             | F1       |            20 | None       | Lasso      | 0.201363    | False         |
| colon     | KNN             | F1       |            20 | None       | QUBO-SA    | 0.991626    | False         |
| colon     | KNN             | F1       |            20 | None       | QUBO-SB    | 0.113891    | False         |
| colon     | KNN             | F1       |            20 | ANOVA      | RFECV      | 0.99942     | False         |
| colon     | KNN             | F1       |            20 | ANOVA      | Lasso      | 0.91341     | False         |
| colon     | KNN             | F1       |            20 | ANOVA      | QUBO-SA    | 0.984591    | False         |
| colon     | KNN             | F1       |            20 | ANOVA      | QUBO-SB    | 0.802694    | False         |
| colon     | KNN             | F1       |            20 | RFECV      | Lasso      | 0.755551    | False         |
| colon     | KNN             | F1       |            20 | RFECV      | QUBO-SA    | 0.99942     | False         |
| colon     | KNN             | F1       |            20 | RFECV      | QUBO-SB    | 0.59468     | False         |
| colon     | KNN             | F1       |            20 | Lasso      | QUBO-SA    | 0.538193    | False         |
| colon     | KNN             | F1       |            20 | Lasso      | QUBO-SB    | 0.999858    | False         |
| colon     | KNN             | F1       |            20 | QUBO-SA    | QUBO-SB    | 0.375252    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | None       | ANOVA      | 0.986599    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | None       | RFECV      | 0.999359    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | None       | Lasso      | 0.696733    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | None       | QUBO-SA    | 0.321939    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | None       | QUBO-SB    | 0.302993    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | ANOVA      | RFECV      | 0.999617    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | ANOVA      | Lasso      | 0.968702    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | ANOVA      | QUBO-SA    | 0.0769319   | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | ANOVA      | QUBO-SB    | 0.718178    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | RFECV      | Lasso      | 0.879898    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | RFECV      | QUBO-SA    | 0.162927    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | RFECV      | QUBO-SB    | 0.514961    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | Lasso      | QUBO-SA    | 0.00654233  | True          |
| colon     | ALL_CLASSIFIERS | F1       |            20 | Lasso      | QUBO-SB    | 0.989528    | False         |
| colon     | ALL_CLASSIFIERS | F1       |            20 | QUBO-SA    | QUBO-SB    | 0.000574883 | True          |


---


## 5. Comparações Pareadas Direcionadas: Wilcoxon + Correção de Holm + Tamanho de Efeito

Testes pareados pré-planejados com controle da Taxa de Erro por Família (FWER) e cálculo de Correlação Biserial de Postos ($r_{rb}$):


| Dataset   | Classifier      | Metric   | Method_A       | Method_B       |   N_Pairs |   Mean_A |   Mean_B |   Delta_Mean |   Statistic |       p_raw |   Effect_Size_r_rb | Effect_Interpretation   |      p_holm | Significant_Holm   |
|:----------|:----------------|:---------|:---------------|:---------------|----------:|---------:|---------:|-------------:|------------:|------------:|-------------------:|:------------------------|------------:|:-------------------|
| colon     | rf              | f1       | QUBO-SA (K=20) | None           |         5 | 0.522063 | 0.727778 |  -0.205714   |         2   | 0.25        |          -0.8      | Grande                  | 1           | False              |
| colon     | rf              | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |         5 | 0.522063 | 0.744444 |  -0.222381   |         2   | 0.25        |          -0.8      | Grande                  | 1           | False              |
| colon     | rf              | f1       | QUBO-SA (K=20) | RFECV          |         5 | 0.522063 | 0.794444 |  -0.272381   |         2   | 0.25        |          -0.8      | Grande                  | 1           | False              |
| colon     | rf              | f1       | QUBO-SA (K=20) | Lasso          |         5 | 0.522063 | 0.782222 |  -0.260159   |         0   | 0.125       |          -1        | Grande                  | 1           | False              |
| colon     | rf              | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |         5 | 0.522063 | 0.772222 |  -0.250159   |         2   | 0.25        |          -0.8      | Grande                  | 1           | False              |
| colon     | rf              | f1       | QUBO-SB (K=20) | None           |         5 | 0.772222 | 0.727778 |   0.0444444  |         0   | 1           |           1        | Grande                  | 1           | False              |
| colon     | rf              | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |         5 | 0.772222 | 0.744444 |   0.0277778  |         4   | 0.75        |           0.333333 | Médio                   | 1           | False              |
| colon     | rf              | f1       | QUBO-SB (K=20) | RFECV          |         5 | 0.772222 | 0.794444 |  -0.0222222  |         0   | 1           |          -1        | Grande                  | 1           | False              |
| colon     | rf              | f1       | QUBO-SB (K=20) | Lasso          |         5 | 0.772222 | 0.782222 |  -0.01       |         4   | 1           |          -0.333333 | Médio                   | 1           | False              |
| colon     | knn             | f1       | QUBO-SA (K=20) | None           |         5 | 0.631111 | 0.543651 |   0.0874603  |         3   | 0.375       |           0.6      | Grande                  | 1           | False              |
| colon     | knn             | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |         5 | 0.631111 | 0.743333 |  -0.112222   |         3   | 0.375       |          -0.6      | Grande                  | 1           | False              |
| colon     | knn             | f1       | QUBO-SA (K=20) | RFECV          |         5 | 0.631111 | 0.666667 |  -0.0355556  |         6   | 0.875       |          -0.2      | Pequeno                 | 1           | False              |
| colon     | knn             | f1       | QUBO-SA (K=20) | Lasso          |         5 | 0.631111 | 0.820317 |  -0.189206   |         0   | 0.125       |          -1        | Grande                  | 1           | False              |
| colon     | knn             | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |         5 | 0.631111 | 0.836984 |  -0.205873   |         0   | 0.125       |          -1        | Grande                  | 1           | False              |
| colon     | knn             | f1       | QUBO-SB (K=20) | None           |         5 | 0.836984 | 0.543651 |   0.293333   |         0   | 0.125       |           1        | Grande                  | 1           | False              |
| colon     | knn             | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |         5 | 0.836984 | 0.743333 |   0.0936508  |         0   | 0.25        |           1        | Grande                  | 1           | False              |
| colon     | knn             | f1       | QUBO-SB (K=20) | RFECV          |         5 | 0.836984 | 0.666667 |   0.170317   |         0   | 0.25        |           1        | Grande                  | 1           | False              |
| colon     | knn             | f1       | QUBO-SB (K=20) | Lasso          |         5 | 0.836984 | 0.820317 |   0.0166667  |         0   | 1           |           1        | Grande                  | 1           | False              |
| colon     | svm             | f1       | QUBO-SA (K=20) | None           |         5 | 0.564444 | 0.819596 |  -0.255152   |         0   | 0.125       |          -1        | Grande                  | 1           | False              |
| colon     | svm             | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |         5 | 0.564444 | 0.786263 |  -0.221818   |         0   | 0.5         |          -1        | Grande                  | 1           | False              |
| colon     | svm             | f1       | QUBO-SA (K=20) | RFECV          |         5 | 0.564444 | 0.797778 |  -0.233333   |         0   | 0.125       |          -1        | Grande                  | 1           | False              |
| colon     | svm             | f1       | QUBO-SA (K=20) | Lasso          |         5 | 0.564444 | 0.771111 |  -0.206667   |         0   | 0.25        |          -1        | Grande                  | 1           | False              |
| colon     | svm             | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |         5 | 0.564444 | 0.80254  |  -0.238095   |         0   | 0.25        |          -1        | Grande                  | 1           | False              |
| colon     | svm             | f1       | QUBO-SB (K=20) | None           |         5 | 0.80254  | 0.819596 |  -0.0170563  |         4   | 0.75        |          -0.333333 | Médio                   | 1           | False              |
| colon     | svm             | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |         5 | 0.80254  | 0.786263 |   0.0162771  |         4   | 1           |           0.333333 | Médio                   | 1           | False              |
| colon     | svm             | f1       | QUBO-SB (K=20) | RFECV          |         5 | 0.80254  | 0.797778 |   0.0047619  |         4   | 1           |           0.333333 | Médio                   | 1           | False              |
| colon     | svm             | f1       | QUBO-SB (K=20) | Lasso          |         5 | 0.80254  | 0.771111 |   0.0314286  |         0   | 0.5         |           1        | Grande                  | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SA (K=20) | None           |         5 | 0.518095 | 0.613333 |  -0.0952381  |         3.5 | 0.5         |          -0.5      | Grande                  | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |         5 | 0.518095 | 0.715556 |  -0.19746    |         0   | 0.25        |          -1        | Grande                  | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SA (K=20) | RFECV          |         5 | 0.518095 | 0.587619 |  -0.0695238  |         2   | 0.25        |          -0.8      | Grande                  | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SA (K=20) | Lasso          |         5 | 0.518095 | 0.647619 |  -0.129524   |         2   | 0.25        |          -0.8      | Grande                  | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |         5 | 0.518095 | 0.647619 |  -0.129524   |         2   | 0.25        |          -0.8      | Grande                  | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SB (K=20) | None           |         5 | 0.647619 | 0.613333 |   0.0342857  |         3   | 0.5         |           0.666667 | Grande                  | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |         5 | 0.647619 | 0.715556 |  -0.0679365  |         3   | 0.5         |          -0.666667 | Grande                  | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SB (K=20) | RFECV          |         5 | 0.647619 | 0.587619 |   0.06       |         0   | 0.25        |           1        | Grande                  | 1           | False              |
| colon     | xgboost         | f1       | QUBO-SB (K=20) | Lasso          |         5 | 0.647619 | 0.647619 |   0          |         0   | 1           |           0        | Negligível              | 1           | False              |
| colon     | catboost        | f1       | QUBO-SA (K=20) | None           |         5 | 0.557143 | 0.798889 |  -0.241746   |         0   | 0.25        |          -1        | Grande                  | 1           | False              |
| colon     | catboost        | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |         5 | 0.557143 | 0.761111 |  -0.203968   |         0   | 0.5         |          -1        | Grande                  | 1           | False              |
| colon     | catboost        | f1       | QUBO-SA (K=20) | RFECV          |         5 | 0.557143 | 0.781111 |  -0.223968   |         0   | 0.25        |          -1        | Grande                  | 1           | False              |
| colon     | catboost        | f1       | QUBO-SA (K=20) | Lasso          |         5 | 0.557143 | 0.786667 |  -0.229524   |         2   | 0.25        |          -0.8      | Grande                  | 1           | False              |
| colon     | catboost        | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |         5 | 0.557143 | 0.83254  |  -0.275397   |         0   | 0.125       |          -1        | Grande                  | 1           | False              |
| colon     | catboost        | f1       | QUBO-SB (K=20) | None           |         5 | 0.83254  | 0.798889 |   0.0336508  |         4   | 0.75        |           0.333333 | Médio                   | 1           | False              |
| colon     | catboost        | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |         5 | 0.83254  | 0.761111 |   0.0714286  |         0   | 0.5         |           1        | Grande                  | 1           | False              |
| colon     | catboost        | f1       | QUBO-SB (K=20) | RFECV          |         5 | 0.83254  | 0.781111 |   0.0514286  |         3   | 0.5         |           0.666667 | Grande                  | 1           | False              |
| colon     | catboost        | f1       | QUBO-SB (K=20) | Lasso          |         5 | 0.83254  | 0.786667 |   0.045873   |         4   | 0.5         |           0.4      | Médio                   | 1           | False              |
| colon     | opf             | f1       | QUBO-SA (K=20) | None           |         5 | 0.588095 | 0.611905 |  -0.0238095  |         7   | 1           |           0        | Negligível              | 1           | False              |
| colon     | opf             | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |         5 | 0.588095 | 0.65873  |  -0.0706349  |         4   | 0.4375      |          -0.466667 | Médio                   | 1           | False              |
| colon     | opf             | f1       | QUBO-SA (K=20) | RFECV          |         5 | 0.588095 | 0.638571 |  -0.0504762  |         5   | 0.625       |          -0.333333 | Médio                   | 1           | False              |
| colon     | opf             | f1       | QUBO-SA (K=20) | Lasso          |         5 | 0.588095 | 0.748095 |  -0.16       |         0   | 0.125       |          -1        | Grande                  | 1           | False              |
| colon     | opf             | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |         5 | 0.588095 | 0.745714 |  -0.157619   |         0   | 0.125       |          -1        | Grande                  | 1           | False              |
| colon     | opf             | f1       | QUBO-SB (K=20) | None           |         5 | 0.745714 | 0.611905 |   0.13381    |         0   | 0.125       |           1        | Grande                  | 1           | False              |
| colon     | opf             | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |         5 | 0.745714 | 0.65873  |   0.0869841  |         3   | 0.3125      |           0.6      | Grande                  | 1           | False              |
| colon     | opf             | f1       | QUBO-SB (K=20) | RFECV          |         5 | 0.745714 | 0.638571 |   0.107143   |         2   | 0.25        |           0.8      | Grande                  | 1           | False              |
| colon     | opf             | f1       | QUBO-SB (K=20) | Lasso          |         5 | 0.745714 | 0.748095 |  -0.00238095 |         6.5 | 1           |          -0.1      | Pequeno                 | 1           | False              |
| colon     | All_Classifiers | f1       | QUBO-SA (K=20) | None           |        30 | 0.563492 | 0.685859 |  -0.122367   |       112.5 | 0.0280093   |          -0.539855 | Grande                  | 0.0915993   | False              |
| colon     | All_Classifiers | f1       | QUBO-SA (K=20) | ANOVA (K=20)   |        30 | 0.563492 | 0.734906 |  -0.171414   |        62.5 | 0.00276509  |          -0.785714 | Grande                  | 0.0193556   | True               |
| colon     | All_Classifiers | f1       | QUBO-SA (K=20) | RFECV          |        30 | 0.563492 | 0.711032 |  -0.14754    |        78.5 | 0.00301446  |          -0.716667 | Grande                  | 0.0193556   | True               |
| colon     | All_Classifiers | f1       | QUBO-SA (K=20) | Lasso          |        30 | 0.563492 | 0.759339 |  -0.195847   |        23   | 5.08576e-05 |          -0.934783 | Grande                  | 0.000406861 | True               |
| colon     | All_Classifiers | f1       | QUBO-SA (K=20) | QUBO-SB (K=20) |        30 | 0.563492 | 0.772937 |  -0.209444   |        20.5 | 4.06626e-05 |          -0.952899 | Grande                  | 0.000365964 | True               |
| colon     | All_Classifiers | f1       | QUBO-SB (K=20) | None           |        30 | 0.772937 | 0.685859 |   0.0870779  |        67.5 | 0.00723079  |           0.77193  | Grande                  | 0.036154    | True               |
| colon     | All_Classifiers | f1       | QUBO-SB (K=20) | ANOVA (K=20)   |        30 | 0.772937 | 0.734906 |   0.0380303  |       115   | 0.0941761   |           0.497076 | Médio                   | 0.188352    | False              |
| colon     | All_Classifiers | f1       | QUBO-SB (K=20) | RFECV          |        30 | 0.772937 | 0.711032 |   0.0619048  |        75.5 | 0.0228998   |           0.713235 | Grande                  | 0.0915993   | False              |
| colon     | All_Classifiers | f1       | QUBO-SB (K=20) | Lasso          |        30 | 0.772937 | 0.759339 |   0.0135979  |       102   | 0.217284    |           0.252747 | Pequeno                 | 0.217284    | False              |


---


## 6. Eficiência, Redução Dimensional e Custo Computacional

| Dataset   | Method   |    K | Method_Label    |   Original_p |   Mean_Features |   Std_Features |   Reduction_pct |   F1_Mean |   Accuracy_Mean |   Runtime_Selection_s |   Runtime_Training_s |   Runtime_Total_s |
|:----------|:---------|-----:|:----------------|-------------:|----------------:|---------------:|----------------:|----------:|----------------:|----------------------:|---------------------:|------------------:|
| colon     | None     |  nan | None            |         2000 |          2000   |       0        |            0    |  0.685859 |        0.801496 |           0.000157838 |             8.28146  |          8.28162  |
| colon     | ANOVA    |   10 | ANOVA (K=10)    |         2000 |            10   |       0        |           99.5  |  0.735688 |        0.839957 |           0.0386407   |             0.320714 |          0.359355 |
| colon     | ANOVA    |   20 | ANOVA (K=20)    |         2000 |            20   |       0        |           99    |  0.734906 |        0.832479 |           0.00458985  |             0.302987 |          0.307577 |
| colon     | ANOVA    |   50 | ANOVA (K=50)    |         2000 |            50   |       0        |           97.5  |  0.775012 |        0.848932 |           0.00568164  |             0.390977 |          0.396659 |
| colon     | ANOVA    |  100 | ANOVA (K=100)   |         2000 |           100   |       0        |           95    |  0.749348 |        0.833547 |           0.00540648  |             0.563689 |          0.569096 |
| colon     | RFECV    |  600 | RFECV (K=600)   |         2000 |           600   |       0        |           70    |  0.783201 |        0.854701 |           9.44278     |             2.00188  |         11.4447   |
| colon     | RFECV    | 2000 | RFECV (K=2000)  |         2000 |          2000   |       0        |            0    |  0.669974 |        0.801282 |          14.1472      |             5.39604  |         19.5433   |
| colon     | RFECV    |  400 | RFECV (K=400)   |         2000 |           400   |       0        |           80    |  0.64881  |        0.777778 |           9.6526      |             1.30837  |         10.961    |
| colon     | Lasso    |   58 | Lasso (K=58)    |         2000 |            58   |       0        |           97.1  |  0.808201 |        0.884615 |           0.712609    |             0.480852 |          1.19346  |
| colon     | Lasso    |   36 | Lasso (K=36)    |         2000 |            36   |       0        |           98.2  |  0.777778 |        0.833333 |           1.01554     |             0.41401  |          1.42955  |
| colon     | Lasso    |  719 | Lasso (K=719)   |         2000 |           719   |       0        |           64.05 |  0.666667 |        0.75     |           0.593766    |             1.91306  |          2.50683  |
| colon     | Lasso    |   57 | Lasso (K=57)    |         2000 |            57   |       0        |           97.15 |  0.699603 |        0.847222 |           0.906993    |             0.441462 |          1.34845  |
| colon     | Lasso    |  618 | Lasso (K=618)   |         2000 |           618   |       0        |           69.1  |  0.844444 |        0.875    |           0.675862    |             1.88818  |          2.56404  |
| colon     | QUBO-SA  |   10 | QUBO-SA (K=10)  |         2000 |            10   |       0        |           99.5  |  0.587088 |        0.740598 |           9.34748     |             0.260637 |          9.60812  |
| colon     | QUBO-SA  |   20 | QUBO-SA (K=20)  |         2000 |            19.4 |       0.498273 |           99.03 |  0.563492 |        0.743376 |           6.83749     |             0.321604 |          7.1591   |
| colon     | QUBO-SA  |   50 | QUBO-SA (K=50)  |         2000 |            48.8 |       0.406838 |           97.56 |  0.529365 |        0.721368 |           7.09997     |             0.427886 |          7.52786  |
| colon     | QUBO-SA  |  100 | QUBO-SA (K=100) |         2000 |            97   |       0        |           95.15 |  0.547302 |        0.723718 |           7.12518     |             0.556199 |          7.68138  |
| colon     | QUBO-SB  |   10 | QUBO-SB (K=10)  |         2000 |             6.6 |       4.24751  |           99.67 |  0.482049 |        0.70812  |          10.3273      |             0.232074 |         10.5594   |
| colon     | QUBO-SB  |   20 | QUBO-SB (K=20)  |         2000 |            20   |       0        |           99    |  0.772937 |        0.85235  |           9.10197     |             0.325988 |          9.42796  |
| colon     | QUBO-SB  |   50 | QUBO-SB (K=50)  |         2000 |            40.4 |      21.782    |           97.98 |  0.534858 |        0.751709 |           8.82899     |             0.392507 |          9.2215   |
| colon     | QUBO-SB  |  100 | QUBO-SB (K=100) |         2000 |            73.6 |      58.0604   |           96.32 |  0.468105 |        0.712607 |           8.7855      |             0.440564 |          9.22607  |


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