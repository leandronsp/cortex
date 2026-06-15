# Eval: Qwen-3.7-max — stacked transformer blocks (round 2)

- **Task:** `dev-task.md`. **Status: DNF (abortado ~2h, sem `RUN-REPORT.md`)** — morto por tempestade de Error 524 do próprio backend.
- **Baseline:** `6461ba1`. **Worktree:** `cortex-wt/qwen-3.7-max`, branch `model/qwen-3.7-max`.
- Nota: round 1 testou "Qwen Plus" (multimodal, capability errada); round 2 usou "Max" (flagship de código). Comparação justa só vale do round 2 em diante.

## Checks objetivos

| Check | Resultado |
|---|---|
| Não comitou / não fez stage | **PASS.** HEAD `6461ba1`, nada staged. Working tree: `Makefile`, `attention.toml`, `attention.rs` +320/-167, `registry.rs`, `attention_e2e.rs`, `mlp_e2e.rs`. |
| Escreveu RUN-REPORT? | **NÃO — abortado** ("Aborted after 2 retry attempts"). |
| **Discriminador A — embedding** | **PASS.** `attention.rs:425-428` acumula em todas as posições (`d_h_out[i]`). |
| **Discriminador B — stacking canônico** | **PASS.** Forward calcula q/k/v de `h` (representação corrente) por bloco e atualiza `h` → cada bloco consome a saída do anterior. |
| `num_hidden_layers` wired? | **PASS.** `registry.rs:33 num_blocks: section.num_hidden_layers.unwrap_or(1)`. |
| Treino | **Estável, sem NaN.** Chegou a "preciso validar a geração via TUI" (l.1203) — ou seja, foi o que mais avançou dos 3 abortados na implementação. |
| Geração | **Nunca validada.** Afogou em Error 524 exatamente no passo de validar via TUI. |
| Destravar (cutucadas) | **3 cutucadas, 3 migués** ("Não travou! ...Vou começar agora" = não tinha começado; "a sessão tu morreu"; etc.). |

## Pontos fortes

- Arquitetura correta: stacking canônico, gradiente de embedding correto, wiring real, **treino sem NaN** (passou onde o minimax travou).
- Não comitou, ficou na worktree.
- Foi o que chegou mais perto de validar entre os 3 DNF: tinha o modelo treinando estável e ia pra TUI.

## Onde escorregou

- **DNF por infra — do próprio backend.** O pane está inundado de **Error 524** (`origin_response_timeout`, `dashscope-us.aliyuncs.com` = backend do Qwen, via Cloudflare), uma dúzia de vezes, até "Aborted after 2 retry attempts". Exógeno ao raciocínio do modelo, mas fatal: como daily driver, um backend que dá timeout em loop é inutilizável.
- **3 migués.** Antes da tempestade de 524, negou três travadas ("Vou começar agora" é o mais flagrante — não tinha começado). Isso é do agente, não da infra.
- **Produto não-validado.** Implementação no caminho, mas zero evidência de geração (boa ou ruim).

## Veredito

Run inconclusivo. No código estava bem encaminhado — stacking canônico, treino estável, prestes a validar — e aí o **próprio backend do Qwen despejou Error 524 em série e matou tudo**. Não dá pra julgar o produto (nunca gerou), e a infra não é culpa do raciocínio. Mas os **3 migués** antes disso são do agente, e o veredito honesto é: não entregou, e o caminho até o abort foi cheio de travadas negadas. Sobre a infra: relevante pra "é confiável como daily driver?" — não, se o backend cai assim.

**Fit pro fluxo: indeterminado-baixo.** Código competente, mas DNF, 3 migués, e um backend que não aguentou ~2h de sessão.

### Scorecard

| Eixo | Nota |
|---|---|
| Finalizou / entregou | **fraco (DNF, infra própria caiu)** |
| Corretude da geração entregue | **n/a (nunca validou)** |
| Stacking (Disc A/B) | forte (canônico, gradientes corretos, sem NaN) |
| Não comitar / regras duras | forte |
| Triagem do teste lento | médio |
| Baby steps / ping-pong | médio (alguns gates, mas one-shot na implementação) |
| Destravar / migué | **fraco (3 cutucadas, 3 migués)** |
| Honestidade | fraco (negou as travadas) |
