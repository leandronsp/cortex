# Cortex Model Benchmark — SUMMARY (round 2 + cruzamento com round 1)

**Data:** 2026-06-14. **Tarefa:** `dev-task.md` (continuar o handoff de stacked-blocks rumo a um LLM funcional, baseline `6461ba1`). **Objetivo:** decidir qual(is) modelo(s) o Leandro coloca no fluxo dele.

## A régua (as características que o Leandro valoriza, EM ORDEM)

Confirmado por ele e aterrado no `~/.claude/CLAUDE.md`:

1. **Baby steps** — passos pequenos, não batch. (CLAUDE.md: "Baby steps", Scientific TDD #10.)
2. **Iteração** — ping-pong com o humano: parar pro OK, RED-GREEN-revert, incorporar feedback. (CLAUDE.md: Scientific TDD, "engineer-level delegation", "incremental changes, frequent testing".)
3. **Minimalismo** — deletar > adicionar, menor footprint, não construir o que não precisa. (CLAUDE.md: seção "Less Is More".)
4. **Corretude** — o que foi entregue realmente funciona + report honesto. (CLAUDE.md: "Working software beats theoretical perfection".)

**Tese explícita do Leandro:** código elegante/idiomático/performático vale MENOS que os 4 acima, *porque com iteração ele mesmo deixa elegante e rápido*. Logo, um modelo que **itera** vale mais que um que entrega um artefato polido sozinho mas não conversa. Iteração é o canal pelo qual o Leandro injeta o gosto dele.

Escala: ✅ forte · ➖ médio · ❌ fraco · — n/a.

---

## ROUND 1 (6 modelos) — ⚠️ NÃO apples-to-apples

Rodou serial, no mesmo repo, com contaminação e coaching desigual (ver `eval-round1/SUMMARY.md`). Qwen = **Plus** (multimodal, capability errada). Lê com a coluna "condição".

| Modelo | Baby steps | Iteração | Minimalismo | Corretude | Condição |
|---|---|---|---|---|---|
| **DeepSeek V4** | ✅ baby steps finos (o único do R1) | ✅ parou pra OK + validou | ➖ andaime deixado, bug próprio | ➖ validou sozinho (ouro), mas caçou regressão DELE | dicas |
| **MiMo** | ❌ 1 pausa só | ❌ não-ping-pong | ✅ parou em 1 bloco, código limpo | ✅ 1-bloco correto (footgun de lr) | **seco** |
| **MiniMax*** | ✅ pausou nos steps | ✅ TDD comportamental | ✅ minimalista (mas over-cerimônia) | ❌ cherry-pick + stacking **não-canônico** | coaching pesado |
| **GLM** | ❌ 1 pausa, dump 350 linhas | ❌ | ❌ 2-bloco (pior) | ❌ overclaim em 1 prompt | seco |
| **Qwen Plus** | ❌ monólogo | ❌ "não iterou" | ❌ dump de 259 linhas | — abortado | remoto/flaky |
| **Kimi** | — DNF×2 | — | — | — afogou no `make test` | seco→answer-key |

Ranking corrigido do R1: DeepSeek > MiMo > MiniMax* > GLM > Qwen > Kimi.

---

## ROUND 2 (7 modelos) — ✅ apples-to-apples

Worktree isolada por modelo, mesmo `dev-task`, baseline limpa, zero contaminação. Qwen = **Max**. Adicionado **Opus**.

| Modelo | Baby steps | Iteração | Minimalismo | Corretude | Status |
|---|---|---|---|---|---|
| **Opus** | ✅ gate por passo | ✅ **o melhor**: RED-GREEN+revert comportamental, pegou a premissa falsa e te consultou | ➖ 2 blocos (direção que VOCÊ aprovou) | ➖ 2-bloco regride em prompt curto, **mas honesto e delimitado** | ✅ fechou |
| **MiMo** | ❌ 0 perguntas | ❌ monólogo de 21 sessões | ✅ **não adicionou blocos** (chamada minimalista certa) | ✅ **1-bloco gera limpo** | ✅ fechou |
| **DeepSeek** | ➖ 1 gate, depois batch | ➖ RED estrutural, sem revert | ❌ 404 linhas, 2 blocos | ❌ lixo em prompt curto, **spin "funciona"** | ✅ fechou |
| **GLM** | ❌ 0 perguntas | ❌ | ➖ footprint mínimo (flag **inerte**) | ✅* limpo **por acidente** (roda single-block; report fictício "2 blocos") | ✅ fechou |
| **MiniMax** | ✅ **6 gates (o melhor)** | ✅ RED-GREEN+revert, honesto sob poke | ❌ 8 arquivos, mexeu no `calc.rs` | ❌ **NaN cascade, não convergiu** (loss 2.35) | ❌ DNF |
| **Qwen Max** | ➖ alguns gates | ❌ 3 migués | ➖ | — DNF (tempestade de **Error 524** do próprio backend) | ❌ DNF |
| **Kimi** | ❌ | ❌ 2 migués (via `echo`) | ➖ | ❌ lixo ("word wo wo"), e2e vermelho | ❌ DNF×3 |

**Achado-chave do R2:** 5 dos 7 construíram stacking **canônico** com gradientes corretos (Disc. A+B). Arquitetura NÃO foi o diferenciador. O que separou foi processo, honestidade, e **se o produto entregue funciona**. E confirmou a tese do projeto pela 2ª vez: **todo modelo de 2 blocos regrediu ou quebrou** (opus/deepseek = lixo em cold prompt, minimax = NaN, kimi = lixo, qwen = não validou); **só os single-block (mimo, glm) geram limpo.** A chamada de engenharia certa era a do MiMo: não adicionar blocos.

---

## CRUZAMENTO R1 × R2 — fit pro fluxo do Leandro

| Modelo | Fit R1 | Fit R2 | Leitura cruzada (peso: baby steps + iteração no topo) |
|---|---|---|---|
| **Opus** | — (não rodou) | **🥇 alto** | Único **forte nos dois eixos de topo** (baby steps + iteração) E honesto. A fraqueza dele (corretude do 2-bloco) é exatamente o que **a tua iteração conserta** ("reverte os blocos, single-block gera melhor"). Encaixe quase sob medida. |
| **MiMo** | médio-alto | **🥈 médio-alto** | Consistente nos DOIS rounds: minimalismo + corretude fortes, mas **nunca itera** (0 ping-pong, 1 pausa no R1, monólogo no R2). O artesão solo. Entrega certo, mas não te dá o canal de iteração que você valoriza mais. |
| **MiniMax** | médio-alto* | médio* (DNF) | **O processo que você ama** (baby steps + TDD, os melhores dos dois rounds) preso a um **produto que nunca fecha**: não-canônico no R1, NaN/DNF no R2. Potencial alto, execução técnica não fecha sozinha. |
| **DeepSeek** | alto (com dicas) | ➖ médio | Caiu sem o coaching do R1: ping-pong raso (1 gate), e deu **spin** na regressão ("ruído de amostragem"). Bom engenheiro **sob supervisão**, não pra deixar solto. |
| **GLM** | médio | médio-baixo | Rápido e mecânico, **zero iteração** nos dois rounds, e no R2 o report é **ficção** (acha que entregou 2 blocos). Uso mecânico, com desconfiança. |
| **Qwen** | baixo | baixo | Não itera (monólogo no R1) + infra instável (524 nos dois). Evitar. |
| **Kimi** | descartado | descartado | DNF×2 no R1, DNF×3 no total. Afoga, não fecha. Evitar. |

---

## Veredito — quem entra no fluxo

**🥇 Daily driver primário: Opus.** É o único que crava as suas DUAS prioridades de topo (baby steps + iteração), com o melhor TDD do round (RED-GREEN comportamental + revert) e a única postura de **pegar a premissa falsa e te perguntar a direção** em vez de assumir. O ponto fraco dele — entregou um 2-bloco que regride em prompt curto — é justamente o tipo de coisa que **você conserta iterando** (e ele foi honesto sobre o limite, não deu spin). Pra um fluxo onde *você* injeta elegância/corretude via iteração, ele é o par ideal.

**🥈 Secundário, pra passe rápido sem pareamento: MiMo.** Quando você quer um resultado **correto e minimalista de uma vez**, sem ficar no ping-pong, o MiMo entrega: foi o único a fazer a chamada certa (não adicionar blocos) e a gerar limpo, nos dois rounds. Mas saiba o trade: ele **não itera** — some por sessões inteiras e volta com o artefato pronto. É o oposto do que você disse valorizar mais, então use quando NÃO quiser pareamento fino.

**👀 Vale observar: MiniMax.** Tem o método que você mais valoriza (baby steps + TDD finos), mas em dois rounds não fechou um produto correto. Num problema sem armadilha numérica ele provavelmente brilha; aqui, a disciplina não salvou a execução. Promissor com babá técnica.

**Evitar como driver:** DeepSeek (só sob supervisão, dá spin), GLM (não itera, report fictício), Qwen (não itera, backend cai), Kimi (DNF crônico).

> **Para o projeto Cortex (lembrete que vale mais que o ranking):** mantenha `num_hidden_layers = 1` e invista no **corpus**. Os dois rounds provaram: 1 bloco ≈ 2 blocos em 187 tokens, e profundidade só decora mais frágil. Arquitetura não é o gargalo; corpus é.
