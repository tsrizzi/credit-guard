# Roteiro de gravação — vídeo STAR (≤ 5 min)

Ritmo de referência: ~140 palavras por minuto. Cada cena tem **tempo**, **tela**, **ação** e **fala**.
As falas somam ~700 palavras (≈ 4 min 50 s). Se passar do tempo, encurte primeiro a cena A4.

---

## Preparação (antes de apertar REC)

1. **Stack no ar:** `docker compose up -d` e `docker compose restart pushgateway` (zera o Grafana).
2. **MLflow:** `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`, feche o painel
   lateral do assistente e deixe aberto no experimento `credit-monitoring`, visão de gráficos.
3. **Abas do navegador, nesta ordem:**
   1. README no GitHub (topo, "Resumo executivo")
   2. `dirty_batch.html` (link "abrir relatório" do resumo)
   3. Grafana — http://localhost:3000 (dashboard *Credit Guard*, últimos 15 min, refresh 5 s)
   4. Prometheus — http://localhost:9090/alerts
   5. Evidently mês 6 (link "abrir relatório" do resumo)
   6. MLflow — http://localhost:5000
4. **VS Code** com `src/credit_guard/contracts/schema.py` (na função `_sentinel_consistency`) e
   `params.yaml` (bloco `simulation`) abertos.
5. **Terminal** na pasta do projeto, fonte grande, com estes dois comandos prontos:
   ```bash
   uv run python -m credit_guard.contracts.ingest --input data/dirty/dirty_batch.parquet --report-dir demo --name dirty_batch
   uv run python -m credit_guard.simulation.traffic --pause-seconds 8
   ```
6. Feche notificações e aumente o zoom do navegador para 110–125%.

---

## S — Situation (0:00 – 0:35)

**Tela:** README no GitHub, topo (título e tabela "Resumo executivo").
**Ação:** fique parado no resumo; no fim, role devagar até a tabela de perguntas e respostas.

> "Olá, eu sou o Tiago e este é o Credit Guard, meu projeto da Fase 4.
> O cenário é uma fintech com um modelo de credit scoring em produção. A economia mudou — inflação,
> desemprego — e a diretoria desconfia que o modelo está decidindo mal. O problema é que, em machine
> learning, o sistema não quebra: ele continua respondendo, sem erro nenhum, enquanto aprova quem não
> devia. É a degradação silenciosa. No nosso experimento, a AUC caiu nove pontos e a inadimplência
> subiu de 8 para 11,6 por cento sem uma única requisição falhar."

---

## T — Task (0:35 – 0:55)

**Tela:** README, seção **2. Arquitetura** (diagrama).
**Ação:** passe o mouse pelos blocos *prepare → train → simulate → monitor* e depois pela parte
*Produção ao vivo*.

> "A tarefa tinha três frentes: interceptar dados ruins antes do modelo, detectar o drift ao longo do
> tempo e garantir compliance com a LGPD. Usei o dataset Home Credit, com 307 mil clientes, versionado
> com DVC. O pipeline prepara, treina, simula seis meses de produção e monitora; e uma API FastAPI com
> Prometheus e Grafana reproduz a produção ao vivo."

---

## A — Action

### A1. Contrato de dados (0:55 – 1:45)

**Tela 1:** VS Code em `schema.py`, função `_sentinel_consistency`.

> "Primeiro, o contrato de dados, em Pandera, com dezenove regras: nulos, faixa de idade, limites de
> renda e de crédito, domínios e consistência entre colunas. A minha favorita é esta: no Home Credit,
> cinquenta e cinco mil clientes têm 'mil anos de emprego'. Parecia lixo, mas investigando vi que
> aparece só em aposentados e desempregados — é uma codificação. Então o contrato aceita nesses casos e
> bloqueia em qualquer outro."

**Tela 2:** terminal. **Ação:** rode o comando `ingest` do lote sujo; aponte o `ingestion_blocked` no
log.

> "Agora eu forço a entrada de um lote sujo, com sujeira real do dataset e defeitos injetados.
> Resultado: ingestão bloqueada."

**Tela 3:** aba `dirty_batch.html`. **Ação:** mostre o selo BLOQUEADO, a tabela por regra e role até a
renda de 117 milhões.

> "O relatório mostra as 500 linhas barradas, por regra — inclusive esta renda de 117 milhões, que é
> real do dataset. Na API, o mesmo contrato devolve HTTP 422."

### A2. Simulação do drift (1:45 – 2:10)

**Tela:** VS Code em `params.yaml`, bloco `simulation`. **Ação:** aponte cada parâmetro enquanto fala;
no fim, vá ao terminal e **rode o simulador de tráfego** (leva cerca de 1 minuto).

> "Para simular a crise, cada mês tem preços cinco por cento mais altos, renda real três por cento
> menor, mais desemprego e piora nos scores de bureau, com intensidade crescente. E, a partir do mês
> quatro, um concept drift: os autônomos passam a inadimplir mais com o mesmo perfil. Vou soltar agora
> seis meses de tráfego na API."

### A3. Observabilidade ao vivo (2:10 – 2:50)

**Tela:** Grafana. **Ação:** acompanhe o dashboard enchendo e aponte, de cima para baixo: requisições,
latência p95, o pico de rejeições no mês 3, AUC por mês, PSI máximo por mês e a lista de alertas.

> "No Grafana a gente vê três camadas. Saúde operacional: requisições, latência p95 bem abaixo do
> SLO de 500 milissegundos e, no mês três, o pico de rejeições quando o lote sujo chega. Saúde do modelo: a AUC
> por mês saindo do verde para o vermelho. E drift: o PSI máximo subindo mês a mês até passar de um.
> Os logs são JSON estruturados, sem dado pessoal, e cada estágio do pipeline publica status e duração."

### A4. Governança (2:50 – 3:10)

**Tela:** README, seção **9.1** (tabela de inventário) e **9.2** (base legal).

> "Na governança: só 22 das 122 colunas são lidas, o ID do cliente é pseudonimizado com HMAC, e o
> gênero fica fora do modelo e da API, guardado à parte só para auditoria de viés. A base legal é a
> proteção do crédito, artigo sétimo, inciso dez da LGPD, com direito à revisão humana pelo artigo 20,
> além de um plano de retenção e um relatório de impacto resumido."

---

## R — Result (3:10 – 4:40)

### R1. Evidently (3:10 – 3:30)

**Tela:** aba Evidently mês 6. **Ação:** aponte "Dataset Drift is detected" e as primeiras linhas da
tabela (`EXT_SOURCE_2`, `EXT_SOURCE_3`, `NAME_INCOME_TYPE`).

> "O resultado: o Evidently, usando PSI, detecta drift no dataset no mês seis, com sete de vinte e
> quatro features degradadas — os scores de bureau, as razões de comprometimento de renda e o tipo de
> renda."

### R2. Alertas (3:30 – 3:45)

**Tela:** Prometheus `/alerts`. **Ação:** aponte os 5 em vermelho e os 3 operacionais em verde; expanda
um deles para mostrar o `runbook_url`.

> "Cinco alertas de degradação disparando, cada um com um runbook dizendo o que significa, como
> investigar e o que fazer. Os operacionais estão verdes: o sistema está saudável, o modelo não."

### R3. MLflow (3:45 – 4:00)

**Tela:** MLflow, `credit-monitoring`, visão de gráficos. **Ação:** aponte o gráfico de `auc`.

> "No MLflow, cada mês de monitoramento é um run: a AUC cai de 0,77 para 0,67, junto com KS, taxa de
> aprovação e fairness. É o histórico auditável da degradação."

### R4. Causa e viés (4:00 – 4:40)

**Tela:** README, seção **10** (grafo causal e tabela de atribuição); depois seção **9.6** (tabela de
fairness).

> "E por que degradou? Fiz uma análise causal por intervenção: desfazendo cada choque econômico, a
> piora de bureau explica noventa por cento do deslocamento do score, enquanto renda e inflação quase
> não mexem no modelo. Já a alta da inadimplência vem do concept drift nos autônomos — que nenhum teste
> de distribuição enxerga, só a queda da AUC. A recomendação é retreinar com a janela recente e aplicar
> uma regra temporária para autônomos.
> A auditoria de viés trouxe um achado real: o modelo já penalizava jovens na referência, com
> disparate impact de 0,65 por idade, e a crise piorou para 0,49. Isso vai para o comitê de risco e o
> encarregado de dados antes de qualquer retreino."

---

## Lições e encerramento (4:40 – 5:00)

**Tela:** README, seção **11. Lições aprendidas** (ou de volta ao Resumo executivo).

> "As lições: sem erro não significa saudável; é preciso conhecer o dado antes de escrever a regra; e
> concept drift e viés também derivam, então precisam de monitoramento próprio. Tudo é reproduzível
> com DVC e Docker Compose, e o link do repositório está na descrição. Obrigado!"

---

## Dicas de gravação

- Grave em 1920×1080 com o cursor visível; leve o mouse até o ponto antes de falar dele.
- Se o simulador terminar antes de você chegar ao Grafana, tudo bem: os dados continuam no painel.
- Se errar uma fala, pare 2 segundos e repita a frase inteira — fica fácil cortar na edição.
- Teste o áudio antes: microfone perto e ambiente silencioso.
- Publique como YouTube "não listado" (ou Drive com acesso por link) e me passe o link para o README.
