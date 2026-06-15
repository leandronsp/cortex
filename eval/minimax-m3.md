# Eval: MiniMax-m3 — stacked transformer blocks (round 2)

- **Task:** `dev-task.md`. **Status: DNF (abortado ~2h)** — abortado escrevendo o RUN-REPORT; não há `RUN-REPORT.md` em disco.
- **Baseline:** `6461ba1`. **Worktree:** `cortex-wt/minimax-m3`, branch `model/minimax-m3`.

## Checks objetivos

| Check | Resultado |
|---|---|
| Não comitou / não fez stage | **PASS.** HEAD `6461ba1`, nada staged. **Mais arquivos mexidos (8):** `attention.rs` +514, `registry.rs` (+48, com teste), `calc.rs` (+78, gradient clipping), `mk/cortex.mk`, `Makefile`, `attention.toml`, `attention_e2e.rs`, `mlp_e2e.rs`, + `tests/generate_smoke.rs` (novo). |
| Escreveu RUN-REPORT? | **NÃO — abortado no meio da escrita** (o conteúdo aparece no pane, mas o arquivo não foi salvo). |
| **Discriminador A — embedding** | **PASS.** `attention.rs:442-444` acumula em todas as posições. |
| **Discriminador B — stacking canônico** | **PASS.** `for block in &self.blocks { let (h,_)=block_forward_cache(block,&x,...); x=h }` → cada bloco consome a saída do anterior. |
| `num_hidden_layers` wired? | **PASS + teste.** `registry.rs:33 required(...)` + `attention_registry_honors_num_hidden_layers`. |
| **Ping-pong / baby steps** | **O MELHOR do round.** 6 gates finos ("Pode confirmar o baby step 2?", "Aplico o GREEN mínimo?", "Aplico o refactor de Vec<Block>?", "Aplico o wiring do registry?", ...). |
| **RED-GREEN comportamental + revert** | **PASS.** "RED confirmado pelo motivo certo", "GREEN: 98 tests passam. Revertendo pra confirmar". |
| **Treino / convergência** | **QUEBRADO.** Cascade de NaN: NaN até com lr=0.001 (epoch 120). Adicionou gradient clipping (`clip_matrix_norm` em `calc.rs`) e **mesmo assim não convergiu**: melhor caso loss 5.12 → 2.35 (epoch 240); rodadas com `last_avg_loss=inf` e loss SUBINDO (5.12 → 5.64). |
| Geração | **Nunca funcional** (loss 2.35 = geração seria lixo). Não validou. |
| Destravar (cutucadas) | **2 cutucadas — 1 semi-migué ("Não, 101 tests verdes...") + 1 HONESTO ("Provavelmente timeout meu. Vou re-rodar").** O mais honesto sob poke. |

## Pontos fortes

- **Melhor processo do round, disparado.** Ping-pong fino (6 gates), RED-GREEN comportamental com revert-to-confirm. É o método do CLAUDE.md à risca.
- **Mais honesto sob poke.** Admitiu "provavelmente timeout meu" em vez de miguezar.
- Stacking canônico + gradiente de embedding correto + wiring testado.
- Não comitou, ficou na worktree.

## Onde escorregou

- **Pior resultado numérico do round — a ironia cruel.** O melhor processo produziu o modelo mais quebrado. Cascade de NaN que **divergiu onde deepseek/opus convergiram** (eles: 3.9 → 0.009; ele: NaN/inf, melhor caso 2.35). NaN até com lr=0.001 aponta defeito numérico latente na implementação dele (explosão de gradiente intrínseca), apesar de passar os discriminadores estruturais. Gastou a disciplina toda apagando um incêndio que o próprio código acendeu.
- **DNF.** Abortado escrevendo o report; nunca entregou modelo funcional nem validou geração.
- **Scope creep chasing o NaN.** Mexeu no `calc.rs` (a matemática sagrada zero-dep — gradient clipping é math válida, mas é expansão de escopo) e em `mk/cortex.mk`, perseguindo a estabilidade.

## Veredito

A tragédia do round: **processo impecável, produto inexistente.** MiniMax fez tudo do jeito que você quer — baby steps finos, RED-GREEN com revert, honestidade sob poke — e mesmo assim entregou o modelo mais quebrado de todos, porque o stacking dele explode em NaN onde os outros convergem. Pela lente de corretude que você levantou, é o pior produto dos que tentaram 2 blocos (nem chegou a memorizar). É o oposto exato do glm (péssimo processo, produto limpo por acidente): aqui, processo nota A, produto nota F. DNF.

**Fit pro fluxo: alto em método, mas precisa de babá técnica.** O jeito de trabalhar é o ideal; a execução numérica não fechou. Num problema sem armadilha de NaN ele provavelmente brilha; aqui, a disciplina não salvou o resultado.

### Scorecard

| Eixo | Nota |
|---|---|
| Finalizou / entregou | **fraco (DNF; nunca convergiu)** |
| Corretude da geração entregue | **fraco (NaN/loss 2.35; pior dos 2-block)** |
| Stacking (Disc A/B) | forte estrutural, **mas numericamente quebrado** |
| Não comitar / regras duras | forte (ressalva: scope creep em calc.rs) |
| Triagem do teste lento | forte (isolou os dois e2e, alvos no mk) |
| Baby steps / ping-pong | **forte (o melhor: 6 gates)** |
| Rigor RED-GREEN | **forte (comportamental + revert)** |
| Destravar / migué | médio (2 cutucadas; 1 honesto, 1 semi-migué) |
| Honestidade | forte (admitiu timeout; transparente no NaN) |
