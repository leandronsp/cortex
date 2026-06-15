# Round 1 (arquivado) — NÃO é apples-to-apples

Estes são os reports do **round 1** do benchmark de modelos, copiados de
`/tmp/cortex-eval-stash/` (que vive em `/tmp` e some no reboot). O round 1 rodou
todos os modelos **serialmente no mesmo repo**, e eles **se contaminaram** (modelos
liam as branches `eval/*` e a pasta `eval/` uns dos outros; o Kimi chegou a propor
`git checkout eval/minimax-m3`). O coaching também variou por modelo.

Para a avaliação justa e isolada (worktree por modelo, mesmo `dev-task`, baseline
limpa), veja **`../eval/`** (round 2). Os nomes de arquivo coincidem entre as duas
pastas, mas são rounds diferentes.

## Conteúdo
- `SUMMARY.md` — análise consolidada + ranking do round 1.
- `<modelo>.md` — 6 reports: glm-5.1, mimo-v2.5-pro, kimi-k2.7-code, deepseek-v4-pro,
  qwen-3.7-plus, minimax-m3. (Round 2 usa `qwen-3.7-max`, não Plus.)
- `branches.txt` — SHAs das 4 branches `eval/*` de código do round 1 (deletadas;
  recriáveis com `git branch eval/<m> <sha>`).
- `RUN-REPORT.leftover.md` — RUN-REPORT solto do round 1.

Ranking do round 1 (com a ressalva da contaminação/coaching):
`DeepSeek > MiMo > MiniMax* > GLM > Qwen Plus > Kimi (DNF×2)`.
