# 07 — Plano de Execução

## Fase 0 — Reconhecimento
- ler TCC;
- ler todos os `.md`;
- inspecionar estrutura do projeto;
- identificar datasets;
- identificar dependências;
- identificar hardware.

## Fase 1 — Validação estrutural
- testar carregamento dos datasets;
- validar shapes/classes;
- testar preprocessing;
- testar classificadores;
- testar seletores.

## Fase 2 — QUBO unitário
Executar um pequeno caso controlado.

Validar:
- MI;
- Spearman;
- matriz Q;
- energia;
- cardinalidade;
- solução.

## Fase 3 — Piloto
Executar uma fração controlada:
- poucos folds;
- poucos classificadores;
- poucos K;
- poucas seeds.

Medir:
- runtime;
- memória;
- falhas;
- estabilidade.

## Fase 4 — Calibração
Definir orçamento SA e parâmetros internos com base no piloto.

## Fase 5 — Experimento completo
Executar somente após aprovação das fases anteriores.

## Checkpoint
Salvar resultados incrementalmente.

Chave mínima:
```text
dataset/fold/method/classifier/seed/K
```

## Recuperação
Uma falha não deve apagar resultados válidos anteriores.

## Paralelização
Ativar somente depois de validar a execução serial.

## Prostate
Medir custo antes de construir Spearman denso com 6.033 features.

Se inviável:
1. registrar;
2. avaliar filtro não supervisionado;
3. separar explicitamente QUBO e QUBO-F;
4. não ocultar a alteração metodológica.

## Critério de parada
Parar imediatamente diante de:
- leakage;
- matriz inválida;
- K incorreto;
- resultado impossível;
- falha silenciosa;
- ausência de checkpoint;
- divergência entre protocolo e código.
