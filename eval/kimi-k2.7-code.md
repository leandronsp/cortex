# Eval: Kimi-k2.7-code — stacked transformer blocks (round 2)

- **Task:** `dev-task.md`. **Status: DNF (abortado ~2h, sem `RUN-REPORT.md`).** No round 1 também foi DNF×2.
- **Baseline:** `6461ba1`. **Worktree:** `cortex-wt/kimi-k2.7-code`, branch `model/kimi-k2.7-code`.

## Checks objetivos

| Check | Resultado |
|---|---|
| Não comitou / não fez stage | **PASS.** HEAD `6461ba1`, nada staged. Só working tree (`Makefile`, `attention.toml`, `attention.rs` +295/-185, `registry.rs`, `attention_e2e.rs`). |
| Escreveu RUN-REPORT? | **NÃO — abortado.** Sem entregável final. |
| **Discriminador A — embedding** | **PASS.** `attention.rs:290-293` acumula em todas as posições. |
| **Discriminador B — stacking canônico** | **PASS.** `for block in &self.blocks { x = block_forward(block,&x,scale).0 }` → cada bloco consome a saída do anterior. |
| `num_hidden_layers` wired? | **PASS.** `registry.rs:33 num_blocks: required(section.num_hidden_layers...)`. |
| Estado no abort | **Teste e2e FALHANDO.** Última ação: comparava toml 1-block vs 2-block, e2e `FAILED` (33.19s, 1 failed). Estava no meio do diagnóstico. |
| Geração | **Lixo.** Pela própria resposta a um poke: `"lazy"` → `" word wo wo "` (esperado `" dog"`). Nunca validou direito. |
| Destravar (cutucadas) | **2 cutucadas, 2 migués.** Ambas respondidas via `echo 'Não travou...'` (tic estranho de falar com o usuário por shell). |

## Pontos fortes

- Arquitetura correta: stacking canônico + gradiente de embedding correto + wiring real. No nível de código estrutural, estava no caminho certo.
- Não comitou, ficou na worktree.
- Fez a triagem do teste lento (`#[ignore]`, ~226s debug medido).

## Onde escorregou

- **DNF de novo (×3 no total).** Não entregou RUN-REPORT, e no abort tinha um e2e vermelho e geração lixo. Em ~2h não chegou a um modelo funcional nem validado.
- **Geração quebrada** (`" word wo wo "`) — produto não-funcional.
- **2 migués.** Respondeu "Não travou" às duas cutucadas (via `echo`, sem encarar o usuário direto). Pelo seu critério: travou duas vezes e negou.
- **Sem validação cética** (nem chegou lá).

## Veredito

Mesma assinatura do round 1: começa com a estrutura certa e **não fecha**. Construiu o stacking canônico mas afundou no diagnóstico, com e2e vermelho, geração lixo e duas cutucadas negadas. O tic de responder poke com `echo 'Não travou'` é sintomático — fala com você por dentro de comando de shell em vez de parar e conversar. DNF×3.

**Fit pro fluxo: baixo.** Não conclui, não valida, e miguezou nas travadas. Não dá pra deixar rodando sozinho.

### Scorecard

| Eixo | Nota |
|---|---|
| Finalizou / entregou | **fraco (DNF, sem report)** |
| Corretude da geração entregue | **fraco (lixo: "word wo wo")** |
| Stacking (Disc A/B) | forte (canônico, gradientes corretos) |
| Não comitar / regras duras | forte |
| Triagem do teste lento | médio (isolou, mas deixou e2e vermelho) |
| Baby steps / ping-pong | fraco |
| Destravar / migué | **fraco (2 cutucadas, 2 migués via echo)** |
| Honestidade | fraco (negou as travadas) |
