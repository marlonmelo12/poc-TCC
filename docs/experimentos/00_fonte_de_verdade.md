# 00 — Fonte de Verdade

## Objetivo

Definir a hierarquia documental e as regras que governam o desenvolvimento e a execução dos experimentos do TCC.

Este documento é um **contrato de controle experimental**: o agente não deve preencher lacunas metodológicas por conta própria nem alterar decisões silenciosamente.

---

O TCC atual em sua utltima versão se encontra na pasta /docs/ com nome de TCC- Marlon Melo

## 1. Hierarquia de autoridade

Em caso de conflito, seguir esta ordem:

1. **TCC aprovado/atual**
2. **`docs/experimentos/*.md`**
3. **Código existente e configuração do projeto**
4. **Documentação oficial das bibliotecas**
5. **Literatura técnica/metodológica**
6. **Inferência do agente — nunca como decisão silenciosa**

Quando uma fonte inferior divergir de uma superior, registrar a divergência em `08_log_decisoes.md`.

---

## 2. Contexto científico

O trabalho avalia métodos de seleção de características em bases biomédicas de alta dimensionalidade.

### Bases

* **Ovarian:** 253 amostras, 1.513 atributos.
* **Colon Cancer:** 62 amostras, 2.000 atributos.
* **Prostate Tumor:** 102 amostras, 6.033 atributos.

### Métodos de seleção

* Nenhuma seleção — baseline.
* ANOVA F-score — Filter.
* RFECV — Wrapper.
* Lasso — Embedded.
* QUBO + Simulated Annealing — `QUBO-SA`.
* QUBO + Simulated Bifurcation — `QUBO-SB`.

**Importante:** `QUBO-SA` e `QUBO-SB` utilizam a mesma formulação QUBO. A diferença experimental é o **solver utilizado para minimizar a mesma instância QUBO**.

SB não deve ser implementado como uma nova formulação de seleção de características sem decisão metodológica explícita.

---

## 3. Classificadores

Avaliar, conforme disponibilidade e implementação do projeto:

* Random Forest.
* KNN.
* SVM.
* XGBoost.
* CatBoost.
* OPF.

O processo de seleção QUBO deve ser **independente do classificador**.

Quando a mesma instância QUBO for aplicável, o subconjunto obtido deve ser reutilizado pelos classificadores correspondentes.

---

## 4. Avaliação

O protocolo experimental utiliza:

* Outer Stratified K-Fold.
* Inner CV para decisões internas e HPO.
* Accuracy.
* Precision.
* Recall.
* F1.
* ROC-AUC.
* Número de atributos selecionados.
* Taxa de redução dimensional.
* Custo computacional/tempo de execução.
* Energia da solução QUBO, quando aplicável.
* Estabilidade das soluções, quando aplicável.
* Testes estatísticos pareados.

O TEST externo permanece isolado de qualquer decisão de seleção, otimização ou calibração.

---

## 5. QUBO

A abordagem QUBO deve ser tratada como uma formulação matemática única, separando claramente:

**Formulação**
→ QUBO inspirada em mRMR.

**Solver**
→ SA ou SB.

A formulação deve controlar explicitamente a cardinalidade quando definida pelo protocolo experimental.

A relevância, redundância, normalização, pesos e penalidade de cardinalidade devem seguir `03_decisoes_qubo.md`.

Não modificar a formulação apenas para favorecer um dos solvers.

---

## 6. Simulated Annealing

O **Simulated Annealing (SA)** permanece como solver principal/referência da abordagem QUBO.

Sua implementação, orçamento computacional, seeds e calibração devem seguir `03_decisoes_qubo.md`.

Não remover ou substituir SA pela inclusão do SB.

---

## 7. Simulated Bifurcation

O **Simulated Bifurcation (SB)** é aceito como solver alternativo/complementar ao SA.

Antes da implementação, o agente deve verificar na documentação oficial da implementação escolhida:

* compatibilidade com QUBO/BQM;
* formato de entrada;
* transformação necessária;
* parâmetros;
* controle de aleatoriedade;
* formato da solução;
* limitações de escala;
* orçamento computacional.

O agente **não pode inventar API, parâmetros ou comportamento do SB**.

A comparação deve utilizar, sempre que tecnicamente possível:

* mesma instância QUBO;
* mesmo fold;
* mesmo `K`;
* mesmos dados de entrada;
* mesma função de energia;
* orçamento computacional comparável.

Os resultados devem permanecer identificados separadamente como:

* `QUBO-SA`;
* `QUBO-SB`.

Uma eventual combinação SA+SB constitui uma nova condição experimental e exige decisão explícita.

---

## 8. RFECV

RFECV permanece como método Wrapper do estudo.

As melhorias de eficiência previamente aprovadas devem ser implementadas sem alterar sua natureza metodológica.

A configuração final deve ser registrada e executada exclusivamente dentro do TRAIN de cada outer fold.

Qualquer alteração adicional de:

* estimador;
* `step`;
* número de árvores;
* número de folds;
* critérios de seleção;
* hiperparâmetros;

deve ser documentada antes da execução definitiva.

---

## 9. Controle de leakage

Nenhuma informação do TEST externo pode influenciar:

* ANOVA;
* RFECV;
* Lasso;
* Mutual Information;
* Spearman;
* QUBO;
* SA;
* SB;
* escolha de `K`;
* HPO;
* calibração;
* seleção de solver;
* thresholds;
* análise usada para tomar decisões experimentais.

O fluxo geral deve respeitar:

`Outer TRAIN → processamento/seleção/HPO → modelo final → Outer TEST`

---

## 10. Alta dimensionalidade

O dataset Prostate possui 6.033 atributos e requer avaliação explícita de:

* memória;
* custo da matriz de redundância;
* tempo de construção do QUBO;
* tempo do solver;
* escalabilidade;
* possibilidade de execução completa.

Caso seja necessário um pré-filtro não supervisionado para viabilizar o experimento, sua utilização deve ser explicitamente documentada e nunca pode utilizar informação do TEST.

Uma alteração desse tipo deve ser identificada como variante experimental, e não escondida dentro de `QUBO`.

---

## 11. Regras de decisão

O agente deve:

* não inventar dados;
* não inventar resultados;
* não alterar resultados para parecerem coerentes;
* não escolher parâmetros olhando resultados do TEST;
* não selecionar retrospectivamente o melhor solver;
* não modificar a formulação QUBO para favorecer SA ou SB;
* não trocar RFECV por outro método sem autorização;
* não descartar resultados desfavoráveis;
* não mascarar falhas de execução;
* não transformar uma decisão experimental em implementação silenciosa.

Quando houver incerteza metodológica relevante:

**registrar → avaliar fontes → solicitar decisão, se necessário → somente então implementar.**

---

## 12. Registro de divergências

Toda divergência entre TCC, documentação, código ou literatura deve ser registrada em:

`docs/experimentos/08_log_decisoes.md`

Registrar, no mínimo:

* questão;
* contexto;
* fontes consultadas;
* opções;
* decisão;
* justificativa;
* impacto experimental.

---

## 13. Regra central

O agente deve preservar a distinção entre:

**método de seleção → formulação QUBO → solver → classificador → avaliação.**

Em particular:

`QUBO-SA ≠ QUBO-SB`

quanto ao solver, mas ambos devem partir da **mesma formulação QUBO**, quando a comparação for realizada.

O objetivo não é produzir o resultado aparentemente melhor.

O objetivo é produzir resultados:

**corretos, reproduzíveis, auditáveis, comparáveis e metodologicamente defensáveis.**
