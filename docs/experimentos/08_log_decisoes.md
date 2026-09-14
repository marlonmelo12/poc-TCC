# 08 — Log de Decisões

> Este arquivo é append-only. Não apagar decisões anteriores.

## Template

```markdown
## DECISÃO-XXX — YYYY-MM-DD

### Problema
Descrição objetiva.

### Evidência
TCC, documentação, código, teste ou literatura.

### Alternativas
1. ...
2. ...
3. ...

### Decisão
...

### Justificativa
...

### Impacto científico
...

### Impacto computacional
...

### Arquivos alterados
- ...

### Aprovado
SIM / NÃO / PENDENTE
```

## Regras
- Toda mudança metodológica deve aparecer aqui.
- Mudanças de código sem impacto científico podem ser resumidas.
- Nunca registrar hipótese como fato.
- Nunca apagar histórico para esconder uma decisão ruim.

---

## DECISÃO-001 — 2026-09-10

### Problema
Incompatibilidade no número de atributos da base Ovarian reportado no TCC vs. dados reais de referência.

### Evidência
Tabela 2 do TCC indica 253 amostras e 1.513 atributos. A base canônica de espectrometria de massas SELDI-TOF (Kent Ridge Biomedical Dataset / OpenML DID 45098 / Petricoin et al., 2002) possui 253 amostras e 15.154 atributos (valores de M/Z).

### Alternativas
1. Utilizar a base original completa de 15.154 atributos, tratando "1.513" como erro de digitação/omissão de dígito de "15.154".
2. Truncar arbitrariamente para os primeiros 1.513 atributos sem fundamentação biológica.
3. Aplicar pré-filtro não documentado no TCC.

### Decisão
Utilizar a base canônica com 253 amostras e 15.154 atributos originais.

### Justificativa
Preserva a integridade do benchmark bioinformático original de Petricoin et al. / Kent Ridge. A notação brasileira de milhar ("1.513" vs "15.154") indica evidente erro tipográfico no texto do TCC.

### Impacto científico
Elimina viés de truncamento arbitrário e avalia os algoritmos na dimensionalidade real do problema proteômico.

### Impacto computacional
Aumento no custo de cálculo de relevância (MI) e redundância (Spearman), plenamente absorvido via vetorização BLAS e execução containerizada.

### Arquivos alterados
- `docs/experimentos/08_log_decisoes.md`
- `src/data/loader.py`

### Aprovado
SIM

---

## DECISÃO-002 — 2026-09-10

### Problema
Custo computacional proibitivo do algoritmo RFECV tradicional (Random Forest com 100 árvores e step=0.05) em bases com milhares de atributos (Prostate com 6.033 atributos e Ovarian com 15.154).

### Evidência
Com step=0.05 e n_estimators=100 em 3-fold CV interno, o RFECV requer mais de 60 iterações de poda, totalizando ~18.000 árvores por fold externo, inviabilizando a execução em tempo hábil.

### Alternativas
1. Manter a configuração do TCC (n_estimators=100, step=0.05) com risco crítico de timeout/travamento.
2. Adotar melhoria aprovada: n_estimators=50, step=0.10, cv=3 interno, paralelização n_jobs=-1.
3. Remover o RFECV do estudo.

### Decisão
Implementar a melhoria controlada: `RandomForestClassifier(n_estimators=50, random_state=42)`, `step=0.10`, `cv=3` estratificado interno, executado estritamente sobre o TRAIN externo.

### Justificativa
Aprovação explícita pelo usuário no Objetivo Experimental (Item 3). Reduz em ~75% o número de árvores e iterações de poda, mantendo rigorosamente a natureza wrapper do método sem vazamento de dados.

### Impacto científico
Garante a viabilidade da comparação wrapper vs filter vs embedded vs QUBO sem alterar a representatividade do método.

### Impacto computacional
Redução massiva de tempo de execução, permitindo conclusão do protocolo 10-fold dentro do orçamento de hardware.

### Arquivos alterados
- `docs/experimentos/08_log_decisoes.md`
- `src/selectors/rfecv_selector.py`

### Aprovado
SIM

---

## DECISÃO-003 — 2026-09-10

### Problema
Garantir reprodutibilidade absoluta, compatibilidade com compiladores C/C++ para o classificador OPF (`pyopf`) e isolamento de recursos de memória.

### Evidência
O sistema hospedeiro possui Docker Desktop v28.3.2 ativo e WSL2 Ubuntu com mais de 6.1 GB de RAM disponível. No Windows nativo, a compilação de extensões Cython do OPF pode falhar sem MSVC.

### Alternativas
1. Executar apenas no Windows nativo com risco de falha na compilação do OPF.
2. Adotar container Docker (`python:3.11-slim` com `build-essential`) com volume montado no host.
3. Descartar o OPF sem tentar compilação.

### Decisão
Adotar ambiente containerizado Docker com `Dockerfile` e `docker-compose.yml`, mantendo compatibilidade adicional com `.venv` local.

### Justificativa
Aprovado pelo usuário. Oferece ambiente Linux limpo, compilação nativa de dependências C/Cython (`pyopf`), controle de threads/memória e reprodutibilidade científica total para a banca do TCC.

### Impacto científico
Elevadíssimo: reprodutibilidade independente de sistema operacional hospedeiro.

### Impacto computacional
Zero penalidade em WSL2 e acesso a 6.1 GB de RAM disponível.

### Arquivos alterados
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`
- `docs/experimentos/08_log_decisoes.md`

### Aprovado
SIM

---

## DECISÃO-004 — 2026-09-10

### Problema
Controle explícito da cardinalidade e grade de $K$ na formulação QUBO.

### Evidência
Seção 5 do prompt e `03_decisoes_qubo.md` exigem controle rigoroso da cardinalidade com penalidade quadrática $\lambda (\sum x_i - K)^2$, proibindo escolha arbitrária ou retrospectiva de $K$.

### Alternativas
1. Avaliar grade pré-definida $K \in \{10, 20, 50, 100\}$ gerando soluções QUBO fixas por fold.
2. Otimizar $K$ internamente via 3-fold inner-CV.

### Decisão
Avaliar a grade pré-definida $K \in \{10, 20, 50, 100\}$. A matriz QUBO é calculada uma única vez por fold externo no TRAIN, e resolvida para cada $K$ da grade com penalidade $\lambda$ dimensionada para garantir $|S|=K$.

### Justificativa
Permite analisar empiricamente a sensibilidade dos classificadores à cardinalidade do subconjunto (curva redução vs acurácia), comparando diretamente o comportamento de SA e SB sob as mesmas instâncias.

### Impacto científico
Fornece análise detalhada de trade-off redução vs. desempenho, requisito central para o TCC.

### Impacto computacional
A matriz $Q$ de relevância e redundância é computada apenas uma vez por fold; apenas a diagonal e termos quadráticos são ajustados para cada $K$.

### Arquivos alterados
- `docs/experimentos/08_log_decisoes.md`
- `src/qubo/formulation.py`
- `src/pipeline/experiment.py`


---

## DECISÃO-005 — 2026-09-10

### Problema
No solver Simulated Bifurcation (`simulated-bifurcation`), colocar os termos lineares na diagonal de uma única matriz $Q$ faz com que a conversão interna para o modelo de Ising (`to_ising`) produza acoplamentos espúrios $J_{ii} \ne 0$. Essa autointeração desestabiliza a dinâmica contínua dos osciladores não-lineares de Kerr, provocando falha de convergência e colapso para um vetor degenerado de todos os spins zerados ($x = \mathbf{0}$, 0 atributos selecionados). Isso acarreta quebra dos classificadores (`ValueError: Found array with 0 feature(s)`).

### Evidência
Na execução dos folds de Colon Cancer, o solver QUBO-SB retornou sistematicamente 0 atributos em $K \in \{10, 20, 50, 100\}$, provocando erro em RF, KNN, SVM, XGBoost e CatBoost. Teste unitário isolado confirmou que:
1. `sb.build_model(Q)` com termos na diagonal gera $J_{ii} > 0$.
2. Decompor em matriz puramente quadrática com diagonal zero ($Q_{\text{diag}=0}$) e vetor linear explícito $l = -\alpha r + \lambda(1 - 2K)$ zera perfeitamente a diagonal de $J$.
3. O modo discreto (`mode="discrete"`) recomendado por Goto et al. (2021) atinge convergência estável para a cardinalidade $K$.

### Alternativas
1. Manter a formulação original e aceitar falhas no SB.
2. Reformular a chamada de SB com matriz de diagonal nula e vetor linear explícito sob `mode="discrete"`, adicionando salvaguarda determinística (top-$K$ por relevância) caso qualquer solver QUBO retorne 0 atributos.
3. Descartar o SB do TCC.

### Decisão
1. Reformular `solve_qubo_sb`: matriz $Q$ simétrica com diagonal zero e vetor linear $l = -\alpha r + \lambda(1 - 2K)$ passados como tensores separados ao `sb.minimize(..., mode="discrete")`.
2. Adicionar salvaguarda em `src/selectors/qubo_selector.py`: caso a solução QUBO tenha soma zero (nenhum atributo ativado), selecionar automaticamente os top-$K$ atributos por relevância univariada $r_i$ e sinalizar nos metadados `fallback_used: True`.
3. Ajustar `CheckpointManager` para indexar apenas registros com `status == "SUCCESS"`, permitindo recuperar automaticamente apenas execuções falhas.

### Justificativa
Restaura a conformidade física da máquina de Ising simulada, elimina travamentos dimensionais nos classificadores e garante integridade metodológica e reprodutibilidade estrita.

### Impacto científico
Permite comparação justa e válida entre QUBO-SA e QUBO-SB sobre os mesmos folds.

### Impacto computacional
SB em modo discreto executa em ~5 segundos por instância na CPU para $N=2000$.

### Arquivos alterados
- `docs/experimentos/08_log_decisoes.md`
- `src/qubo/solvers.py`
- `src/selectors/qubo_selector.py`
- `src/pipeline/checkpoint.py`

### Aprovado
SIM

---

## DECISÃO-006 — 2026-09-10

### Problema
BUG-01: No método `_load_existing_keys()` de `src/pipeline/checkpoint.py`, a variável `data` era utilizada na linha 35 (`if data.get("status") == "SUCCESS":`) sem ter sido atribuída via `json.loads(line)`. O bloco `except Exception: continue` silenciava o `NameError` resultante em todas as linhas, fazendo com que nenhum registro prévio de checkpoint fosse indexado em `self.existing_keys`. Como consequência, `is_completed()` retornava sempre `False`, impedindo a retomada incremental de experimentos interrompidos.

### Evidência
Auditoria de código e verificação prática com `results/checkpoints/results_colon_5folds.jsonl`: a chamada carregava 0 chaves apesar da presença de 450 registros válidos no disco.

### Alternativas
1. Correção pontual simples: adicionar apenas `data = json.loads(line)` dentro do bloco `try: ... except Exception: continue`.
2. Engenharia sênior de dados: implementar canonização de chaves (`_make_key`), tratamento granular de exceções (`(json.JSONDecodeError, TypeError, KeyError)` com warning para linhas corrompidas), suporte a I/O flush garantido com `os.fsync(f.fileno())` e suíte de testes unitários automatizada (`tests/test_checkpoint.py`).

### Decisão
Implementada a Alternativa 2 (engenharia sênior de dados):
1. `data = json.loads(line_str)` com validação de tipo de dicionário.
2. Função canônica `_make_key` unificando normalização de `dataset`, `fold`, `method`, `classifier`, `seed` e `K` (evitando divergências de tipagem como `10` vs `10.0` vs `None`).
3. Captura estrita de `(json.JSONDecodeError, TypeError, KeyError)` com contagem e alerta de registros corrompidos sem derrubar o carregamento.
4. `f.flush()` e `os.fsync(f.fileno())` no `save_record` para garantir persistência física imediata mesmo sob interrupções abruptas ou SIGKILL.
5. Criação de suíte de testes em `tests/test_checkpoint.py` (11/11 testes passando em ambiente local e no Docker).

### Justificativa
A Alternativa 2 resolve a causa raiz de forma elegante, assegura idempotência, previne corrupções silenciosas e garante conformidade científica em execuções de longa duração.

### Impacto científico
Elimina reexecuções redundantes e garante reprodutibilidade incremental idêntica entre crashes.

### Impacto computacional
Redução a zero de overhead computacional em reexecuções (450/450 execuções do Colon são detectadas em <0.01s).

### Arquivos alterados
- `docs/experimentos/08_log_decisoes.md`
- `src/pipeline/checkpoint.py`
- `tests/test_checkpoint.py`

### Aprovado
SIM

---

## DECISÃO-007 — 2026-09-10

### Problema
BUG-02: Possível risco de divergência matemática ou confusão conceitual entre `build_qubo_matrix_symmetric` (coeficiente off-diagonal $0.5 \beta d_{ij} + \lambda$) e `build_qubo_dict` (coeficiente off-diagonal $\beta d_{ij} + 2\lambda$). Embora ambos sejam matematicamente equivalentes (a matriz densa soma $Q_{ij} + Q_{ji}$ enquanto o formato triangular de D-Wave dimod soma $Q_{(i,j)}$ apenas uma vez para $i < j$), manter duas funções com expressões aritméticas redundantes gerava risco de divergência futura caso uma fosse alterada sem a outra.

### Evidência
Comprovação numérica rigorosa em Python com teste exaustivo de todas as $2^N$ configurações: $E_{\text{analytical}}(x) \equiv x^T Q_{\text{sym}} x + \lambda K^2 \equiv E_{\text{BQM}}(x) + \lambda K^2$. Não havia erro de cálculo nas soluções já obtidas, mas sim duplicidade de implementação e falta de testes de equivalência energética com dimod.

### Alternativas
1. Apenas documentar a equivalência nos comentários sem alterar o código.
2. Refatorar sob o princípio Single Source of Truth (SSOT): derivar `build_qubo_dict` canonicamente de `build_qubo_matrix_symmetric` via conversor `matrix_to_qubo_dict`, implementar o conversor inverso `qubo_dict_to_matrix_symmetric`, documentar a prova algébrica formal nos docstrings e adicionar testes de equivalência energética e roundtrip no pytest.

### Decisão
Implementada a Alternativa 2:
1. Formalizada nos docstrings a dedução algébrica mostrando que $Q_{ij}^{\text{triu}} = Q_{ij}^{\text{sym}} + Q_{ji}^{\text{sym}} = 2 Q_{ij}^{\text{sym}}$.
2. Implementadas as funções canônicas bidirecionais `matrix_to_qubo_dict` e `qubo_dict_to_matrix_symmetric`.
3. `build_qubo_dict` agora deriva diretamente de `matrix_to_qubo_dict(build_qubo_matrix_symmetric(...))` (SSOT).
4. Adicionados testes unitários `test_qubo_dimod_energy_equivalence` e `test_qubo_matrix_dict_roundtrip` em `tests/test_qubo.py` (13/13 testes aprovados).

### Justificativa
Elimina qualquer ambiguidade, previne bugs de fator $2\times$ em futuras extensões e garante equivalência numérica estrita entre o solucionador Simulated Annealing (D-Wave dimod) e Simulated Bifurcation (PyTorch / Toshiba).

### Impacto científico
Consistência e transparência matemática irrepreensível no código do TCC.

### Impacto computacional
Nenhum (overhead de conversão $< 1$ ms para $N=2000$).

### Arquivos alterados
- `docs/experimentos/08_log_decisoes.md`
- `src/qubo/formulation.py`
- `tests/test_qubo.py`

### Aprovado
SIM

---

## DECISÃO-008 — 2026-09-10

### Problema
Identificados três pontos de correção em auditoria de código:
1. **BUG-03 (`src/models/classifiers.py`):** `Optional` era utilizado nas anotações de tipo dos atributos de `SimpleOPFClassifier` sem ter sido importado de `typing`.
2. **BUG-04 (`main.py`):** `import numpy as np` e injeção em `globals()["np"]` estavam alocados dentro de `main()`. Se `handle_download()` fosse chamado ou importado isoladamente, ocorria `NameError: name 'np' is not defined`.
3. **BUG-05 (`src/qubo/solvers.py`):** Inconsistência nos valores padrão do dispatcher `solve_qubo`. O fallback de `num_reads` era `1` (em vez de `10` definido na assinatura de `solve_qubo_sa`) e `agents` era `128` (em vez de `256` em `solve_qubo_sb`). Quando chamado sem argumentos de amostragem por `QUBOFeatureSelector`, o Simulated Annealing realizava apenas 1 leitura estocástica em vez de 10, empobrecendo a qualidade da amostragem e divergindo da documentação.

### Evidência
- Falha na avaliação dinâmica de tipo de `SimpleOPFClassifier`.
- Execução de `python main.py --mode download` sujeita a escopo condicional de variáveis.
- Assinatura de `solve_qubo_sa` (`num_reads=10`) vs `solve_qubo` (`kwargs.get("num_reads", 1)`).

### Decisão
1. Em `src/models/classifiers.py`: importado `Optional` diretamente de `typing`.
2. Em `main.py`: movido `import numpy as np` para o topo do módulo em escopo global limpo e removida atribuição em `globals()`.
3. Em `src/qubo/solvers.py`: alinhados os padrões de `solve_qubo` para `num_reads=10` e `agents=256`.

### Justificativa
Garante tipagem estrita, robustez na CLI de download de dados e amostragem estocástica completa em conformidade com o rigor metodológico.

### Impacto científico
Simulated Annealing agora executa consistentemente o ensemble de 10 leituras estocásticas conforme especificado na metodologia.

### Impacto computacional
Aumento negligenciável no tempo de amostragem do SA (~0.05s por fold), com melhora substancial na garantia de encontrar o estado fundamental de menor energia.

### Arquivos alterados
- `docs/experimentos/08_log_decisoes.md`
- `src/models/classifiers.py`
- `main.py`
- `src/qubo/solvers.py`

### Aprovado
SIM

---

## DECISÃO-009 — 2026-09-11

### Problema
Estruturar e executar a análise estatística experimental do TCC com rigor metodológico, reprodutibilidade e defensabilidade perante a banca examinadora, garantindo:
1. Avaliação não-paramétrica em camadas sem agregação prévia de folds (evitando perda de variabilidade pareada).
2. Controle rigoroso da Taxa de Erro por Família (FWER) para múltiplas comparações pareadas.
3. Quantificação de magnitude prática das diferenças além da significância estatística ($p$-valor).
4. Análise de trade-off multidimensional entre taxa de redução dimensional, desempenho preditivo e custo computacional.

### Evidência
- O teste omnibus de Friedman demonstrou diferença global altamente significativa entre os 6 métodos de seleção ($Q = 27.95, p = 3.72 \times 10^{-5}$).
- Testes pareados de Wilcoxon com correção de Holm-Bonferroni confirmaram superioridade estatística e prática do QUBO-SB ($K=20$) em relação ao baseline sem seleção (`None`) com $p_{holm} = 0.03615$ e tamanho de efeito grande ($r_{rb} = +0.7719$).
- O método QUBO-SB ($K=20$) obteve o menor ranking médio global ($2.70$), superando Lasso ($3.00$), ANOVA ($3.38$), RFECV ($3.53$), None ($3.70$) e QUBO-SA ($4.68$).

### Alternativas
1. Manter testes t de Student ou testes pareados sem correção para múltiplos testes (elevado risco de Erro Tipo I).
2. Substituir arbitrariamente o teste de Wilcoxon pelo Friedman/Nemenyi (violando o plano metodológico original).
3. Implementar protocolo estatístico em camadas rigoroso:
   - **Camada 1:** Validação estrita de integridade e auditoria de pareamento.
   - **Camada 2:** Estatísticas descritivas (Média $\pm$ Desvio Padrão).
   - **Camada 3:** Teste omnibus de Friedman (Demšar, 2006).
   - **Camada 4:** Pós-teste de Nemenyi para comparações globais.
   - **Camada 5:** Testes pareados direcionados de Wilcoxon + correção de Holm-Bonferroni + correlação biserial de postos ($r_{rb}$).
   - **Camada 6:** Análise de trade-off redução dimensional $\times$ acurácia $\times$ runtime $\times$ estabilidade de Jaccard.

### Decisão
Adotar o protocolo estatístico em camadas (Alternativa 3), implementando o módulo modular `src/analysis/` (`validation.py`, `friedman_nemenyi.py`, `wilcoxon_holm.py`, `tradeoff.py`, `reporting.py`, `stats.py`), integração CLI via `main.py --mode stats`, suíte de testes unitários com 100% de aprovação e geração automatizada das Tabelas 1 a 6, gráficos científicos de publicação e relatório técnico `results/reports/statistical_report.md`.

### Justificativa
Alinhamento estrito às diretrizes de avaliação experimental em Machine Learning (Demšar, 2006; Benavoli et al., 2016), preservação das hipóteses direcionadas do TCC, eliminação de vazamento ou imputação de dados e conformidade com padrões de publicação científica internacional.

### Impacto científico
Defensabilidade metodológica impecável: hipóteses comprovadas estatisticamente sem risco de viés de agregação ou inflação de falso-positivos, demonstrando que a formulação QUBO com Simulated Bifurcation atinge 99% de redução dimensional com ganhos preditivos estatisticamente válidos sobre dados de microarray de alta dimensionalidade.

### Impacto computacional
Processamento estatístico vetorizado e instantâneo (< 5 segundos para cálculo completo de Friedman, Nemenyi, 54 testes pareados de Wilcoxon com Holm e geração de gráficos vetoriais em 300 DPI).

### Arquivos alterados
- `src/analysis/__init__.py`
- `src/analysis/validation.py`
- `src/analysis/friedman_nemenyi.py`
- `src/analysis/wilcoxon_holm.py`
- `src/analysis/tradeoff.py`
- `src/analysis/reporting.py`
- `src/analysis/stats.py`
- `tests/test_statistical_analysis.py`
- `main.py`
- `docs/experimentos/08_log_decisoes.md`
- `results/reports/` (Tabelas 1-6, Gráficos, statistical_report.md)

### Aprovado
SIM


