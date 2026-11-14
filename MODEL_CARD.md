# Model Card — `credit-scoring` v1 (model_version `d616328c08`)

## Uso pretendido
- **Tarefa:** estimar a probabilidade de inadimplência (PD) de um solicitante de crédito pessoa física.
- **Decisão:** aprovar se PD < 0,120 (threshold que aprova 80% da população de referência).
- **Usuários:** motor de decisão de crédito da fintech, com **revisão humana** disponível a pedido
  do titular (LGPD art. 20) e obrigatória para recusas de clientes < 25 anos próximas do threshold.
- **Fora do escopo:** crédito empresarial, precificação de taxa, cobrança, qualquer uso que não
  seja análise de risco de crédito.

## Dados
- Home Credit Default Risk (`application_train.csv`), 307.511 clientes, inadimplência 8,07%.
- 22 de 122 colunas lidas (minimização); 23 linhas em quarentena pelo contrato de dados.
- Split estratificado: 60% treino, 15% teste (referência), 25% holdout (produção simulada). Seed 42.
- **Não usados como feature:** `CODE_GENDER` (mantido só para auditoria), identificadores.

## Modelo
- Pipeline scikit-learn: engenharia de features (idade, anos de emprego, flag "não empregado",
  crédito/renda, parcela/renda, prazo implícito, bem/crédito) → one-hot (`handle_unknown="ignore"`)
  → `XGBClassifier` (400 árvores, profundidade 4, learning rate 0,05, subsample 0,8).
- Rastreado no MLflow (`credit-training`), registrado como `credit-scoring` v1.

## Métricas (referência)
| ROC-AUC | Gini | KS | PR-AUC | Aprovação | Inadimplência dos aprovados |
|---:|---:|---:|---:|---:|---:|
| 0,764 | 0,528 | 0,394 | 0,243 | 80,0% | 4,8% |

## Fairness (referência)
- Disparate impact por gênero: **0,913** (passa na regra dos 80%).
- Disparate impact por faixa etária: **0,649** (não passa): aprovação de 60,1% para < 25 anos contra
  92,5% para 55+; TPR 64,5% vs. 93,6%. Mitigação no README, seção 9.6.

## Limitações conhecidas
- Sensível a drift nos scores de bureau (`EXT_SOURCE_2/3`): eles explicam ~90% do deslocamento de
  score numa crise simulada; choques de capacidade de pagamento (renda, inflação, desemprego) quase
  não movem o score, ou seja, o modelo pouco usa comprometimento de renda.
- Degrada sob concept drift (−9,2 p.p. de AUC no 6º mês simulado). Monitorado mensalmente com alertas.
- Treinado com dados de outra população (Home Credit); o uso real exigiria revalidação local.

## Monitoramento
Contrato de dados em toda ingestão; PSI/KS/JS + Evidently por mês; AUC com rótulos atrasados;
disparate impact por gênero e idade; alertas `FeatureDriftCritical`, `DriftedShareHigh`,
`ModelAucDrop`, `ApprovalRateShift`, `FairnessDisparateImpact` com runbooks no README.

## Gatilhos de retreino
`ModelAucDrop` disparado, ou `FeatureDriftCritical` persistente por 2 meses com confirmação da área
de crédito. O novo modelo entra como challenger no MLflow e só é promovido com AUC e DI iguais ou
melhores que o champion.
