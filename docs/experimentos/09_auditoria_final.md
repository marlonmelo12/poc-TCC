# 09 — Auditoria Final

Preencher somente após o experimento.

## Protocolo
- [ ] Implementação corresponde ao protocolo.
- [ ] Protocolo corresponde ao TCC ou todas as divergências estão documentadas.

## Leakage
- [ ] Outer TEST permaneceu intocado.
- [ ] Preprocessing foi ajustado no TRAIN.
- [ ] Seleção foi ajustada no TRAIN.
- [ ] QUBO usou apenas TRAIN.
- [ ] HPO usou apenas TRAIN.
- [ ] K/pesos QUBO não usaram TEST.

## QUBO
- [ ] Formulação documentada.
- [ ] MI validada.
- [ ] `abs(Spearman)` validado.
- [ ] Cardinalidade implementada.
- [ ] K registrado.
- [ ] Seeds registradas.
- [ ] Energia validada.
- [ ] Estabilidade avaliada.

## Alta dimensionalidade
- [ ] Prostate teve custo avaliado.
- [ ] Pré-filtro, se usado, está documentado como variante.
- [ ] Memória/tempo foram registrados.

## Machine Learning
- [ ] RFECV mantido/documentado.
- [ ] HPO possui orçamento.
- [ ] OPF validado.
- [ ] Threads controladas.

## Resultados
- [ ] Accuracy.
- [ ] Precision.
- [ ] Recall.
- [ ] F1.
- [ ] ROC-AUC.
- [ ] Número de features.
- [ ] Reduction rate.
- [ ] Runtime.
- [ ] Energia QUBO.
- [ ] Estabilidade.

## Estatística
- [ ] Unidade experimental definida.
- [ ] Comparações pareadas.
- [ ] Múltiplas comparações tratadas quando aplicável.
- [ ] Classificadores não foram tratados como réplicas independentes indevidamente.

## Reprodutibilidade
- [ ] Seeds.
- [ ] versões.
- [ ] parâmetros.
- [ ] configuração computacional.
- [ ] checkpoints.
- [ ] resultados brutos preservados.

## Conclusão da auditoria

### Pendências
Liste qualquer item não atendido.

### Alterações metodológicas
Liste todas.

### Status
- [ ] APROVADO PARA ANÁLISE
- [ ] NECESSITA CORREÇÃO
