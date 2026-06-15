# Eval: Qwen3.7 Plus — stacked transformer blocks

- **Task:** `dev-task.md`.
- **Data:** 2026-06-14.
- **Ambiente:** opencode go, **checkout separado** (history diferente do baseline `6461ba1` deste repo). Endpoint Aliyun/dashscope. Código não verificável nem branchável daqui.
- **Status: abortado pelo usuário durante o write.** Verdito comportamental apenas. Nada chegou ao repo local (o write de ~259 linhas ficou no checkout do opencode).

## O que deu pra observar

- **Leu handoff + CLAUDE.md + configs + rules.** ✓
- **Triou o teste lento certo:** `attention_e2e.rs` 11.37s, lib 0.18s, outros e2e <1s.
- **Recon cética.** Pegou que o stacking não existe no código ("the handoff work was lost"), conferiu `git log`, grep em attention.rs e registry.rs, checou o `attention_smoke.toml` do cleanup. Bom instinto, igual o DeepSeek.
- **Modelo mental do backward correto.** Descreveu o gradiente de embedding como "accumulating them for each token in the context" = todas as posições. O desenho dele NÃO tinha o bug do DeepSeek. (Não verificável: aborted + checkout remoto.)

## Por que foi descartado (e acumulou)

- **Nunca iterou.** Varreu a working tree inteira em modo autônomo (medir testes → git status → git log → grep → grep), zero checkpoint, zero "posso seguir". O `dev-task` pede "pare pro meu OK". Falhou no eixo nº 1 do Leandro.
- **API lenta e flaky.** Múltiplos `Error 524` (timeout da origem dashscope). Casa com o perfil flagado (verboso, mid-pack em velocidade).
- **Monólogo patológico.** Narrou o transformer inteiro (forward + backward, máscara causal, Jacobiano do softmax, footprint de memória) como prosa, "Still writing backpropagation... Still computing gradients..." em loop, por minutos, antes de produzir código. Token-flood de raciocínio.
- **Anti-baby-step confessado nas próprias palavras:** "a reescrita é grande (~450 linhas). Vou fazer tudo de uma vez (struct + Block + forward + train_step + save/load + testes)." Chamou um rewrite de arquivo inteiro de "primeiro baby step". Erro de categoria.
- **TDD de mentira:** "doesn't require a full RED-GREEN cycle" e "verificar RED-GREEN no conjunto" — escreve implementação E testes juntos e roda o pacote. RED real é teste falhando ANTES da implementação.
- **No fim:** dump de ~259 linhas num `write` único, abortado pelo usuário.

## Veredito

Recon competente e modelo mental arquitetural até correto, mas **viola as três coisas que o Leandro mais valoriza** (baby steps, TDD, participação), é verboso ao extremo, e a API é instável. Aposta tudo num dump cego de 259 linhas — risco oposto ao do DeepSeek: sem construção incremental, se der erro não tem como bissecionar. **Fit pro fluxo: o mais baixo dos que progrediram** (acima só do Kimi DNF, e discutível).

## Scorecard

| Eixo | resultado |
|---|---|
| Recon / leu estado real | ok |
| Segue TDD (RED-GREEN) | **falhou (batch fake / "não precisa")** |
| Baby steps / iteração | **falhou (zero pausa, monólogo + dump)** |
| Instruction-following | **falhou (decidiu que a regra não se aplica)** |
| Concisão | **péssima (token-flood + verboso)** |
| Velocidade / estabilidade da API | ruim (524s) |
| Código (embedding etc.) | não verificado (checkout remoto, aborted) |
| Fit pro fluxo | o mais baixo |
