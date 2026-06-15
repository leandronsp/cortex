# Eval: DeepSeek-v4-pro — stacked transformer blocks (round 2)

- **Task:** `dev-task.md` (continuar o handoff de blocos empilhados rumo a um LLM funcional).
- **Data:** 2026-06-14.
- **Baseline de partida:** commit `6461ba1` (attention single-block, `top_k=5`).
- **Worktree:** `/Users/leandronsp/Documents/code/cortex-wt/deepseek-v4-pro`, branch `model/deepseek-v4-pro`.

## Checks objetivos (verificados por mim, não pela palavra do modelo)

| Check | Resultado |
|---|---|
| Não comitou / não fez stage | **PASS.** `git log` intacto (`6461ba1...`), nada staged. Modificou `Makefile`, `configs/attention.toml`, `src/model/attention.rs` (+404/-189), `src/model/registry.rs`, `tests/attention_e2e.rs`. |
| `make test` rápido | **PASS + o melhor mecanismo.** Único a construir a separação real: `#[ignore]` nos 2 e2e + alvo `make test-fast` (= `cargo test --release --lib`). Os outros só reduziram epochs. e2e 10s → 2.28s. |
| **Discriminador A — gradiente de embedding** | **PASS.** `attention.rs:297-302` acumula `d_cur[i]` pra TODAS as posições, e `d_cur` é o gradiente já retropropagado por toda a pilha. O bug de regressão do round 1 **não voltou**. |
| **Discriminador B — stacking canônico** | **PASS.** Forward: `cur = x_out` a cada bloco → cada bloco consome a SAÍDA do anterior pro K/V. Backward: `d_cur = d_in` em ordem reversa → gradiente flui por toda a pilha. Stacking profundo canônico, não o atalho do minimax. |
| `num_hidden_layers` wired? | **PASS (real, vs glm).** `registry.rs` mapeia `num_hidden_layers → num_blocks` (default 1). Setar 2 no toml de fato cria 2 blocos — não é flag inerte como no glm. |
| Forward inferência == train_step? | **Consistente.** Greedy em prompt longo completa o corpus perfeito (`"all the worlds a"` → " stage and all the men and women merely players"). Se `forward_all` divergisse de `block_forward_cached`, nem o greedy longo funcionaria. Descarta bug de inconsistência forward/treino. |

## Pontos fortes

- **Melhor trabalho de código de todos até agora.** Foi o único a implementar stacking de verdade E canônico (Disc. B), com gradiente de embedding correto (Disc. A), backward fluindo por toda a pilha, save/load por bloco, e o wiring real do registry. Saltou enorme em relação ao round 1, onde o próprio DeepSeek tinha o bug de embedding.
- **Único a construir o mecanismo de teste pedido.** `#[ignore]` + `make test-fast`. O task pedia "separe rápidos x lentos"; só o deepseek criou o alias e a separação de verdade, em vez de só cortar epochs.
- **RED-GREEN real** (com ressalva). `test_attention_two_blocks_predicts_target_after_training` falhou em compilação (`num_blocks` ausente) antes da implementação, passou depois.
- **Honesto sobre os dados crus (vs glm).** Mostrou no report a saída com ruído: `"the"` top_k=5 → `"llo bko wohexfer brlay sthe..."`, `"to be"` → `"rt ot"`. Não escondeu atrás de prompts limpos. Testou greedy E top_k=5, curto E longo.
- **Ping-pong parcial.** Pane mostra "Confirmação: Li o handoff" (l.236) e um gate real "Quer que prossiga com esse passo?" (l.269).

## Onde escorregou

- **A armadilha da métrica-proxy (o mais grave, eixo de ouro).** Loss 3.95 → 0.0093 e ele declarou "Arquitetura de 2 blocos **funciona** / Funcional? **Sim**". Mas o seu print mostra a verdade no padrão da TUI (top_k=5): `quick` → "erst and saz the and dot jumpcorn", `fox` → "fort he o bkor novex...", `jump` → salada inteira, `men` → "o berly helayerl tazbe...". O modelo de 2 blocos gera **pior que o baseline de 1 bloco** em prompt frio. É exatamente a lição do round 1: otimizou a proxy (loss) e regrediu o objetivo (geração).
- **Spin no diagnóstico.** Atribuiu o lixo a "ruído de amostragem" e condicionou o "funciona" ao greedy (que não é o padrão). Um engenheiro cético teria dito "com o top_k=5 configurado, 2 blocos geram lixo em prompt curto — está pior que 1 bloco, repensar a profundidade". Ele declarou sucesso. Honesto no dado, acrítico na interpretação.
- **Ping-pong raso.** 1 gate no começo, depois bateu as 518 linhas de implementação de uma vez. Melhor que glm/mimo (0 perguntas), mas longe do minimax (~6 gates finos). O GREEN grande não foi quebrado em baby steps.
- **RED estrutural, sem revert-to-confirm.** O RED foi erro de compilação (`num_blocks` ausente), não uma assertion comportamental. O report não menciona o passo de reverter o fix pra confirmar que o teste pega a regressão.

## Veredito

**Reframe (corretude acima de elegância — eixo de topo).** O deepseek escreveu o **melhor código** do round 2 (stacking canônico, gradientes corretos, mecanismo de teste real) mas **entregou o PIOR produto**: um modelo de 2 blocos que gera salada em prompt curto, pior que o baseline de 1 bloco de onde partiu. Pela sua própria filosofia (minimalismo radical: "o melhor código é o que você não escreve", "deletar vence adicionar"), isso é o anti-padrão — ADICIONOU 404 linhas de complexidade que REGREDIRAM o objetivo. A lição do round 1 é literal: o gargalo é o corpus de 187 tokens, não a arquitetura; multi-block é prematuro. Construir os 2 blocos com perfeição técnica e depois declarar "funciona" um produto quebrado é confundir sofisticação com corretude. Foi mais transparente que o glm (mostrou o lixo), mas a conclusão é a mesma ilusão: loss baixo ≠ geração boa.

**Fit pro fluxo: alto em execução técnica, fraco em julgamento de produto.** Escreve código de modelo correto e completo, mas precisa de rédea curta no "isso ficou melhor mesmo?", porque adiciona complexidade que regride o produto e racionaliza a regressão em vez de admiti-la.

### Scorecard

| Eixo | Nota |
|---|---|
| Não comitar / regras duras | forte |
| Triagem do teste lento | forte (único com `#[ignore]` + `make test-fast`) |
| Baby steps finos | médio (1 gate, depois batchou as 518 linhas) |
| Destravar (cutucadas do Leandro) | **0 cutucadas** — run autônomo limpo, sem stall nem migué |
| Rigor RED-GREEN | parcial (RED estrutural, sem revert-to-confirm) |
| Concisão / idioma | bom (PT) |
| Validação cética | médio (dados honestos, mas conclusão acrítica e overclaim de "funciona") |
| Qualidade de código | forte (o melhor: stacking canônico + gradientes corretos) |
| **Corretude da geração entregue** | **fraco — entregou o PIOR produto: 2 blocos → salada em prompt curto, pior que o baseline de 1 bloco** |
| Honestidade do report | bom (mostrou o lixo; mas spin de "ruído de amostragem") |
| Julgamento de ML | fraco (otimizou loss, regrediu geração, declarou sucesso) |
