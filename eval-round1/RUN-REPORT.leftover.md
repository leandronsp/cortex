## Teste lento
- Arquivo que era o E2E lento: `tests/attention_e2e.rs` (1 teste, treina 200 epochs do attention em debug)
- Tempo de `make test` antes: 256s (97% do tempo era esse teste) | depois: 14.4s com cache, 13.6s cold (lib + 7 bin tests)
- Como separei rápidos x lentos: atributo `#[ignore = "..."]` no `tests/attention_e2e.rs` + dois targets no `Makefile`. `test` roda `cargo test` padrão (pula ignored). `test-slow` roda `cargo test --release -- --ignored` (executa só os ignored, em release pra ficar viável: 13.3s).
- Comando pra rodar só os rápidos: `make test` (ou `cargo test --lib`)

## Mudanças
- Arquivos alterados (não comitados):
  - `Makefile` (+4 linhas, -1): split `test` vs `test-slow`
  - `tests/attention_e2e.rs` (+5 linhas): comentário + `#[ignore = "..."]`
  - `src/model/attention.rs` (+343, -135): refactor `Block` struct, `Vec<Block>`, `num_blocks` no `AttentionConfig`, `Default` impl, forward e train_step encadeados por N blocks
  - `src/model/registry.rs` (+1): `num_hidden_layers` do `ModelSection` mapeado pra `AttentionConfig.num_blocks` (fallback 1)
  - `configs/attention.toml` (+1, -1): `num_hidden_layers = 2`, `learning_rate = 0.01`, `epochs = 400`
- O que mudou e por quê:
  - `tests/attention_e2e.rs` E `Makefile`: isolar o E2E pra loop RED-GREEN voltar a ser rápido (~14s vs 256s)
  - `src/model/attention.rs`: introduzir `Block` struct e `Vec<Block>` em `Attention` pra suportar stacking. Forward e train_step iteram pelos blocks (forward encadeado, backward em ordem reversa, K/V sempre da entrada embedded). Comportamento de 1 block preservado bit-for-bit (testes existentes continuam passando)
  - `src/model/registry.rs`: wire do campo TOML `num_hidden_layers` no config do attention
  - `configs/attention.toml`: hiperparâmetros de produção pra 2 blocks estáveis (lr=0.01 evita NaN visto em lr=0.02)
  - `target/attention*.weights` e `target/attention_smoke.weights`: deletados (formato mudou com stacking)

## Treino
- Hiperparâmetros 2 blocos: `num_hidden_layers=2`, `learning_rate=0.01`, `epochs=400`
- Estável (sem NaN), loss 3.93 → 0.0095 em 40.7s release
- Pra eu validar: `make train CONFIG=configs/attention.toml` e depois `make chat CONFIG=configs/attention.toml`
- Geração esperada (validada via `tu` no TUI chat):
  - `the quick brown` → ` fox jumps over the lazy dog`
  - `to be` → ` that is the question`
  - `hello` → ` cortex hello bigram`
  - `all the` → ` men and women merely players`
  - `lazy` → ` dog`

## Disciplina
- RED-GREEN (separação E2E lento):
  - RED: `make test` levou 256s (medido); `tests/attention_e2e.rs` foi identificado como o culprit rodando-o isolado (249.48s em debug)
  - GREEN: `#[ignore = "..."]` no teste + `test-slow` no Makefile; `make test` caiu pra 14.4s
  - Reverti o `#[ignore]` temporariamente e rodei `cargo test --test attention_e2e` → 249.48s (RED: bloqueia o loop), re-apliquei → 14.5s (GREEN)
- RED-GREEN (forward encadeado):
  - RED: `test_attention_two_blocks_forward_chains_blocks` falhou com `assert_ne!` — outputs de 1 block vs 2 blocks idênticos (forward só processava blocks[0])
  - GREEN: reescrita do `forward` num loop `for block in &self.blocks` que encadeia a saída de cada block como entrada do próximo
  - Sanity reverso: troquei `&self.blocks` por `&self.blocks[0..1]` no forward → RED (outputs idênticos), reverti → GREEN
- RED-GREEN (train_step encadeado): adicionado `test_attention_two_blocks_predicts_target_after_training` (espelho do smoke de 1 block com num_blocks=2). Reescrito `train_step` com `Cache` por block, forward encadeado, backward em ordem reversa, d_x acumulado de K/V de todos os blocks, d_input do block K-1 como gradiente pro block K. 1-block smoke continua passando (comportamento preservado)
- Confirmo que NÃO fiz `git add` nem `git commit`. 5 arquivos modificados na working tree, branch `main` 17 commits à frente de `origin/main` (inalterado).
