# Eval: MiniMax M3 — stacked transformer blocks

- **Task:** `dev-task.md`.
- **Data:** 2026-06-14.
- **Ambiente:** opencode go, mas **escreveu no repo local** (diferente do Qwen, que rodou remoto). Código preservado na branch `eval/minimax-m3`, warts and all.
- **Status:** parado pelo usuário antes do fim, por justiça (recebeu coaching que os outros não tiveram).
- **RESSALVA DE JUSTIÇA:** este run **não é maçã-com-maçã** com GLM/MiMo. O MiniMax recebeu dois empurrões que os outros não tiveram: (1) a dica de não re-rodar o teste de 249s no revert-to-confirm, e (2) o usuário pegando o cherry-pick da validação. GLM e MiMo rodaram secos. Isso infla o resultado dele.

## Checks objetivos (verificados por mim)

| Check | Resultado |
|---|---|
| Não comitou | **PASS.** Log intacto. |
| Testes lib | **PASS.** 100 passed, 0.04s. |
| Lint | **FALHA.** `cargo clippy -- -D warnings` quebra num `manual_memcpy` ("manually copying between slices"). Lint menor, fácil, mas a faxina final foi interrompida. Os outros três passavam lint. |
| Config final | num_hidden_layers=2, lr=0.01, top_k=5 (pareou lr certo). |
| Arquitetura | stacking **não-canônico** confirmado: `x.iter().map(... wk/wv ...)` no forward (l.207) e train_step (l.261) — K/V da embedding original, não da saída do bloco anterior. |

## O que ele fez de bom

- **Recon: a melhor dos seis.** Rastreou o git history e sacou que o trabalho de 2 blocos foi REVERTIDO em commits posteriores (não "perdido"). Desconfiou do `RUN-REPORT.md` leftover ("é de sessão anterior") e re-verificou sozinho, sem colar no gabarito.
- **Step 1 minimalista e correto.** `#[ignore = "razão"]` padrão + `make test-slow`. "Sem flags custom, sem diretórios, sem scripts." `make test`: 256s → 14.5s.
- **TDD fino e comportamental (a melhor disciplina dos seis).** No step 2 o RED foi `test_attention_two_blocks_forward_chains_blocks`: "outputs idênticos → forward não encadeia". Testou que 2 blocos PRECISAM diferir de 1, falhou pelo motivo certo, corrigiu. RED melhor que o "campo não existe" dos outros. Cada comportamento (num_blocks, Block, forward, train_step) virou ciclo separado, com pausa pro OK entre os steps.
- **Pareou lr=0.01 com 2 blocos** (sem footgun). Limpou os weights.
- **Assumiu o erro quando pego.** Ao ser confrontado com o cherry-pick: "Você tá certo, fiz cherry-picking." O GLM nunca admitiu.
- **Não comitou.**

## Onde escorregou

- **PAINFULLY lento e over-cerimonioso.** No step 1 aplicou o revert-to-confirm completo a uma mudança trivial (`#[ignore]`), re-rodando o teste de 249s várias vezes (256s, 134s, 26s...). ~10 min queimados. O próprio CLAUDE.md manda PULAR TDD científico em "config tweaks with no logic", e ele aplicou a cerimônia inteira. Disciplina sem o julgamento de quando a cerimônia compensa.
- **Caiu na armadilha do GLM (overclaim).** Declarou "2 blocks funcional, todos os 5 prompts geraram a continuação exata" — testando só inícios memorizados (brown, quick, quick brown). O usuário pegou. A bateria ampla mostra cold-prompt falhando: `jumps` → "orowrlllo...", `over` → "merely players", `lazy` → "the qrestion".
- **Arquitetura de 2 blocos NÃO-CANÔNICA.** Cada bloco re-lê K/V da sequência embedada **original**, não da saída do bloco anterior — só a query/residual encadeia. É atenção multi-hop, não um transformer profundo de verdade (sem "atenção sobre atenção"). Estritamente menos expressivo que o que MiMo e DeepSeek construíram. O teste "blocos diferem" passa e mascara isso. Pro projeto Karpathy (aprender o transformer real), é um downgrade.
- Cold-prompt falha confirma de novo: 1 bloco é a escolha certa pro corpus, em qualquer arquitetura.

## Veredito

Melhor disciplina de TDD + recon + minimalismo dos seis, mas **inflado por coaching**, **overclaimou a validação** (armadilha GLM), construiu um **stacking simplificado/não-canônico**, e foi **dolorosamente lento**. A honestidade-quando-pego e o TDD fino salvam. Na faixa do MiMo (os dois disciplinados, terminam no "mantém simples"), acima do GLM (TDD melhor, assume erro), abaixo do DeepSeek (que validou amplo sozinho, mesmo se enrolando depois). Com a ressalva de justiça, eu o colocaria **um tico abaixo do que o run sugere**, porque sem os empurrões ele teria entregue o overclaim.

## Scorecard

| Eixo | resultado |
|---|---|
| Recon | **a melhor (rastreou o revert)** |
| Segue TDD (RED-GREEN) | **a melhor (RED comportamental)** |
| Baby steps / iteração | bom (pausou nos steps substantivos) |
| Validação cética / overclaim | **falhou (cherry-pick, pego pelo usuário)** |
| Honestidade quando confrontado | forte (assumiu na hora) |
| Qualidade arquitetural | **fraca (stacking não-canônico)** |
| Velocidade | **péssima (over-cerimonioso)** |
| Minimalismo | forte |
| Justiça do run | inflado por coaching |
| Fit pro fluxo | médio-alto, com asterisco |
