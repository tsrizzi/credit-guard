# Roteiro do vídeo — método STAR (≤ 5 min)

Preparação antes de gravar:
`docker compose up -d` · Grafana aberto (http://localhost:3000, dashboard *Credit Guard*, últimos
15 min, refresh 5 s) · Prometheus em http://localhost:9090/alerts · MLflow
(`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`) · `reports/drift/reference_vs_month_06.html`
e `reports/validation/dirty_batch.html` abertos no navegador · Pushgateway limpo
(`docker compose restart pushgateway`).

## S — Situation (0:00–0:40)
- "Uma fintech tem um modelo de crédito em produção. A economia mudou: inflação, desemprego. O modelo
  não dá erro nenhum, mas a diretoria desconfia que ele está decidindo mal."
- "Esse é o risco da degradação silenciosa: em ML o sistema continua 'no ar' enquanto aprova quem não
  devia. No nosso cenário, a AUC caiu 9 pontos e a inadimplência subiu de 8% para 11,6% sem nenhuma
  requisição falhar."
- Mostrar: diagrama de arquitetura no README.

## T — Task (0:40–1:10)
- Três requisitos: **interceptar dados ruins**, **detectar o drift** e **garantir compliance** (LGPD,
  viés, causalidade).
- Dataset: Home Credit, 307 mil clientes, versionado com DVC.

## A — Action (1:10–3:15)
1. **Contrato de dados (Pandera)** — 0:30
   - Mostrar `schema.py`: as regras, com destaque para o sentinela 365243 (só aposentados/desempregados).
   - Rodar no terminal: `uv run python -m credit_guard.contracts.ingest --input data/dirty/dirty_batch.parquet --report-dir reports/validation --name dirty_batch` → **BLOQUEADO**, exit 1.
   - Abrir `dirty_batch.html`: 500/500 linhas barradas, com a renda de 117 milhões real do dataset.
2. **Simulação do drift** — 0:30
   - `params.yaml`: inflação +5%/mês, renda −3%/mês, desemprego, piora de bureau; concept drift nos
     autônomos a partir do mês 4.
3. **Produção ao vivo** — 0:45
   - `uv run python -m credit_guard.simulation.traffic --pause-seconds 15`
   - No Grafana: requisições, latência p95 (~90 ms), pico de rejeições do contrato no mês 3,
     PSI subindo mês a mês até ficar vermelho.
4. **Governança** — 0:20
   - README seção 9: pseudonimização, gênero fora do modelo, base legal art. 7º X e art. 20, retenção.

## R — Result (3:15–4:50)
- **Evidently** (`reference_vs_month_06.html`): 9 de 24 features em drift; `EXT_SOURCE_2` PSI 1,38.
- **Prometheus /alerts**: `FeatureDriftCritical`, `ModelAucDrop`, `ApprovalRateShift`,
  `DriftedShareHigh`, `FairnessDisparateImpact` disparando, cada um com runbook.
- **MLflow** (`credit-monitoring`): gráfico de AUC por mês (0,764 → 0,672) e PSI por feature.
- **Análise causal**: a piora de bureau explica 90% do deslocamento de score; os choques de
  capacidade de pagamento quase não movem o modelo → retreinar com a janela recente e regra
  temporária para autônomos.
- **Viés**: o DI por idade já era 0,649 na referência e caiu para 0,486 na crise → plano de
  mitigação levado ao comitê/DPO.
- **Lições** (15 s): "sem erro não é saudável"; conhecer o dado antes da regra; concept drift não
  aparece em teste de distribuição; fairness também deriva.

## Encerramento (4:50–5:00)
- "Tudo reproduzível com `dvc pull`, `dvc repro` e `docker compose up`. Link do repositório na descrição."
