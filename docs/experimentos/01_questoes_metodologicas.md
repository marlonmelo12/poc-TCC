# 01 — Questões Metodológicas

Este arquivo é um checklist de decisões. Nenhuma questão crítica pode permanecer implícita.

## A. Validação
- [ ] O outer split ocorre antes de qualquer transformação aprendida?
- [ ] Preprocessamento é ajustado somente no TRAIN?
- [ ] Seleção ocorre somente no TRAIN?
- [ ] MI ocorre somente no TRAIN?
- [ ] Spearman ocorre somente no TRAIN?
- [ ] QUBO usa somente TRAIN?
- [ ] K é escolhido somente dentro do TRAIN/inner-CV?
- [ ] HPO usa somente TRAIN?
- [ ] TEST é utilizado apenas uma vez para avaliação?

## B. QUBO
- [ ] Formulação documentada matematicamente?
- [ ] Relevância definida?
- [ ] Redundância definida?
- [ ] Escalas normalizadas?
- [ ] Redundância usa `abs(Spearman)`?
- [ ] Existe penalidade explícita de cardinalidade?
- [ ] K é definido por protocolo, não pelo TEST?
- [ ] Matriz Q é válida/simétrica?
- [ ] Convenção de energia está documentada?
- [ ] Seeds são controladas?

## C. Alta dimensionalidade
- [ ] Custo de memória da matriz foi estimado?
- [ ] Custo computacional foi medido?
- [ ] Prostate é executável na infraestrutura disponível?
- [ ] Caso exista pré-filtro, ele é explicitamente identificado como variante?
- [ ] O pré-filtro é ajustado somente no TRAIN?

## D. Machine Learning
- [ ] HPO tem orçamento explícito?
- [ ] Inner CV é apropriado ao tamanho amostral?
- [ ] RFECV não foi removido apenas por custo?
- [ ] OPF foi validado antes do experimento completo?
- [ ] Threads/processos estão sob controle?

## E. Estatística
- [ ] Unidade experimental definida?
- [ ] Comparações são pareadas?
- [ ] Múltiplas comparações são tratadas?
- [ ] Classificadores não são indevidamente tratados como observações independentes?

## Regra de bloqueio
Se qualquer item crítico não puder ser respondido com evidência, interromper a execução correspondente e registrar a lacuna.
