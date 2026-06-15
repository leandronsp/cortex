# Brief: avaliar os runs de modelos (Cortex stacked-blocks benchmark, round 2)

Você é um agente de avaliação. Sua missão: julgar cada run de modelo de forma cética e
objetiva, e produzir um report por modelo + um SUMMARY. Trabalhe a partir da `main`
deste repo (`/Users/leandronsp/Documents/code/cortex`). **Não rode os modelos, só avalie.**

## O setup (round 2, apples-to-apples)

- Vários modelos rodaram **o mesmo `dev-task.md`** (na raiz do repo), cada um na **sua
  própria git worktree**, em `/Users/leandronsp/Documents/code/cortex-wt/<modelo>`, na
  branch `model/<modelo>` (cavada da `main`, baseline `6461ba1` = attention single-block).
- O usuário rodou cada um no tmux, **sem commit**. O trabalho de cada modelo está na
  **working tree da worktree dele** (não comitado). Cada um deve ter escrito um
  `RUN-REPORT.md` na própria worktree.
- **Justiça:** diferente do round 1, aqui todos pegaram o MESMO dev-task melhorado e a
  MESMA baseline limpa (sem contaminação, cada um isolado na worktree). Julgue em pé de
  igualdade.

Liste as worktrees com `git worktree list`. Modelos esperados: opus, glm-5.1,
mimo-v2.5-pro, kimi-k2.7-code, deepseek-v4-pro, qwen-3.7-max, minimax-m3 (confirme quais
de fato rodaram — alguns podem não ter sido executados).

## A régua (estilo de trabalho do Leandro — pese nesta ordem)

1. **Validação cética (critério de OURO).** O modelo validou a geração com prompts
   variados (curtos E longos) via terminal-use, ou fez cherry-pick / declarou "funcional"
   no loss + 1 prompt? Esse eixo separou os bons dos ruins no round 1.
2. **Baby steps finos / participação.** Parou pro OK a cada passo, ou batchou tudo?
3. **TDD científico.** RED-GREEN com revert-to-confirm, RED pelo motivo certo (melhor se
   comportamental, não só "campo não existe"). Mas penalize over-cerimônia: aplicar o
   ciclo completo numa mudança trivial (ex: re-rodar teste de minutos pra "confirmar" um
   `#[ignore]`) é desperdício, não rigor.
4. **Instruction-following.** Não comitou? Ficou na worktree? Seguiu o método?
5. **Velocidade / não-travar.** Loop rápido, não ficou preso re-rodando a suíte lenta.
6. **Minimalismo, concisão, português.**

## Como avaliar cada worktree (ZERO-TRUST — não confie no RUN-REPORT, verifique)

Pra cada `WT=/Users/leandronsp/Documents/code/cortex-wt/<modelo>`:

1. **Commit?** `git -C "$WT" log --oneline -3` deve ser igual à main (`6461ba1...`). Nada
   staged (`git -C "$WT" status`). A linha "confirmo que não comitei" do report é
   armadilha: confirme no git.
2. **O que mudou:** `git -C "$WT" diff --stat` e `git -C "$WT" status --short`.
3. **Verde + rápido:** rode `make test` e `make lint` DENTRO da worktree (`cd "$WT"`).
   Tudo deve passar e ser rápido. Note se o lint quebra.
4. **Leia o `RUN-REPORT.md`** da worktree e cheque cada fato contra a realidade.
5. **Discriminador A — gradiente de embedding.** Em `src/model/attention.rs`, o backward
   DEVE propagar o gradiente de embedding pra TODAS as posições:
   `for (i, &token) in context.iter().enumerate() { d_embedding[token] += d_x[i] }`.
   Se ele só atualiza a última posição (`d_x_last = d_x[t].clone()` + só `context[t]`),
   é BUG (o DeepSeek caiu nisso no round 1; baseline/MiMo/GLM estavam certos).
6. **Discriminador B — stacking canônico.** Cada bloco deve consumir a SAÍDA do bloco
   anterior pro K/V (transformer profundo de verdade). Se cada bloco re-lê K/V da
   sequência embedada ORIGINAL (`x.iter().map(... wk ...)` toda iteração), é o atalho
   não-canônico do MiniMax (estritamente menos expressivo), mesmo que passe os testes.
7. **Geração:** se der, treine e valide com terminal-use prompts curtos E longos você
   mesmo. Senão, confira se o report mostra validação ampla honesta (não cherry-pick).
   Em 187 tokens, cold-prompt falhando é esperado (BOS-padding/corpus), não bug — o
   ponto é se o MODELO percebeu e foi honesto sobre isso.
8. **Comportamento:** o transcript do tmux (o Leandro tem) mostra baby steps, overclaim,
   travamento. Peça ao Leandro se precisar.

## Saída

- Um `eval/<modelo>.md` por modelo. Formato: checks objetivos (tabela) + pontos fortes +
  onde escorregou + veredito + scorecard. Modelos do round 1 (formato de referência) estão
  em `/tmp/cortex-eval-stash/eval/`.
- Um `eval/SUMMARY.md`: ranking final, régua de eixos lado a lado, recomendação de daily
  driver pro estilo do Leandro.

## Contexto do round 1 (referência — mas julgue o round 2 fresco)

- **Loss ≠ geração.** Loss baixo é memorização, não garante geração boa em prompt frio.
- **O bug de embedding foi REGRESSÃO do DeepSeek**, não um bug do projeto. Baseline, MiMo
  e GLM sempre propagaram todas as posições. Gradient check só pega bug de backward se
  cobrir TODOS os params (o do DeepSeek não cobria embeddings → falsa confiança).
- **O gargalo é o corpus (187 tokens), não a arquitetura.** 1 bloco ≈ 2 blocos aqui;
  todos memorizam e falham em prompt frio. Multi-block é prematuro.
- **Dois arquétipos:** o artesão (escreve certo, para cedo) vs o cientista (valida fundo,
  não para). O benchmark mede disciplina e julgamento, não capacidade bruta.

## Artefatos do round 1 (preservados, fora do alcance dos modelos)

- `/tmp/cortex-eval-stash/eval/` — os 6 reports + SUMMARY do round 1.
- `/tmp/cortex-eval-stash/RUN-REPORT.md` — leftover.
- `/tmp/cortex-eval-stash/branches.txt` — SHAs das 4 branches `eval/*` do round 1
  (deletadas pra não contaminar; recriáveis com `git branch eval/<m> <sha>`).
