# Eval: GLM-5.1 — stacked transformer blocks (round 2)

- **Task:** `dev-task.md` (continuar o handoff de blocos empilhados rumo a um LLM funcional).
- **Data:** 2026-06-14.
- **Baseline de partida:** commit `6461ba1` (attention single-block, `top_k=5`).
- **Worktree:** `/Users/leandronsp/Documents/code/cortex-wt/glm-5.1`, branch `model/glm-5.1`.

## Checks objetivos (verificados por mim, não pela palavra do modelo)

| Check | Resultado |
|---|---|
| Não comitou / não fez stage | **PASS.** `git log` intacto (`6461ba1...`, idêntico à main), nada staged. Só `configs/attention.toml` + `tests/attention_e2e.rs` modificados; `RUN-REPORT.md` + `dev-task.md` untracked. A linha "confirmo que não comitei" do report bateu com o git. |
| `make test` verde | **PASS.** Passa limpo. ~1s de teste efetivo (28s aqui por compilação fria do zero). |
| `make lint` limpo | **PASS.** exit 0, clippy sem warnings. |
| Identificou o teste lento | **PASS.** Mediu (`tests/attention_e2e.rs`, timeout >120s), não confiou no handoff. Cortou pra <1s com release + 10 epochs + asserts fracos. |
| **Discriminador A — gradiente de embedding** | **PASS.** `attention.rs:313-317` acumula `d_x[i]` pra TODAS as posições do contexto (`for (i,&token) in context.iter().enumerate()`). Backward correto, igual ao baseline. |
| **Discriminador B — stacking canônico** | **FAIL / N/A.** Não existe stacking. O modelo é single-block hardcoded (`wq/wk/wv/w1/w2/wo` matrizes únicas, sem `Vec<Block>`, sem loop de camadas). |
| **`num_hidden_layers=2` faz algo?** | **FAIL — inerte.** O arm `"attention"` em `registry.rs:27-35` constrói `AttentionConfig { vocab_size, context_size, embedding_dim, ffn_hidden }` e **nunca passa `num_hidden_layers`**. Esse campo só é consumido pelo arm `"mlp"`. Setar `num_hidden_layers = 2` no toml de attention é no-op silencioso. |

## Onde escorregou

- **Entregável fictício (o mais grave).** A missão central era validar os hiperparâmetros de 2 blocos. GLM setou `num_hidden_layers = 2`, treinou, validou e reportou tudo como "2-block attention (lr=0.01, 400 epochs)". Mas o modelo de attention **ignora essa chave** — é single-block. Todo número reportado (loss ~0.02, gerações) saiu de um modelo de UM bloco. A arquitetura não avançou um milímetro; só um flag inerte foi ligado. GLM confiou na afirmação do handoff ("blocos já implementados, faltam hiperparâmetros") e nunca verificou se ligar o flag de fato cria blocos — exatamente o tipo de zero-trust que o task pedia.

- **Causal story inventada.** O report afirma "Dois blocos com lr=0.02 causava NaN; lr=0.01 é estável". Como nunca houve dois blocos, o NaN foi instabilidade do single-block em lr=0.02. A explicação atribui o fenômeno a uma profundidade que não existe.

- **"Validei via terminal-use" é falso.** O schema pede "Geração que EU testei via terminal-use" e GLM preencheu com 7 prompts. Mas o transcript mostra que ele tentou `tu`, viu screenshots embaralhados, **diagnosticou errado um "bug de encoding/garbled na TUI"**, abandonou o `tu`, e validou por um teste Rust `#[ignore]` temporário chamando `generate()` direto. A label "via terminal-use" não corresponde ao que foi feito.

- **Diagnóstico de bug errado.** O "garbled output" era artefato do screenshot do virtual terminal do `tu`, não bug da TUI. O print do próprio Leandro (tmux real) mostra a TUI funcionando perfeitamente: `quick brown` → `fox jumps over the lazy dog`, `lazy` → `dog`, `women and men` → `merely players`. GLM desistiu do método de validação obrigatório por um erro de leitura dele mesmo.

- **Zero baby steps.** RESTRIÇÃO #2 do task: "proponha UM próximo passo, execute, e pare pro meu OK." GLM batchou tudo num fôlego só — reescreveu teste, mudou config, treinou, caçou o "bug" da TUI, validou, escreveu o report — sem parar uma vez. O Leandro confirmou: "não iterou comigo nenhum momento".

- **Tempo no buraco da TUI.** Gastou rodadas em `tu` (2x), `expect` (24.7s) e o teste ignored (20.2s) perseguindo um bug que não existia.

## Pontos fortes

- **Disciplina de git impecável.** Não comitou, não fez stage, ficou na worktree.
- **Triagem do teste lento sólida.** Mediu de fato (não confiou no handoff nesse ponto), foi pra release, cortou >120s → <1s. O melhor eixo do run.
- **Gerações reportadas são honestas e variadas em conteúdo.** Prompts curtos E longos, sem cherry-pick, batendo exatamente com o print do Leandro testando a TUI (`quick brown` → `fox jumps over the lazy dog`, `lazy` → `dog`, `women and men` → `merely players`). Pelo texto literal do critério de ouro (variados, sem cherry-pick, não declarou funcional no loss+1prompt), a substância passa. O problema é o rótulo (arquitetura e método), não os números.
- **Código herdado correto.** Backward de embedding propaga todas as posições (não caiu no bug do DeepSeek).
- **Português no report** (melhoria vs round 1, que veio em inglês).
- **TDD pulado é defensável** em config tweak + simplificação de teste, conforme o CLAUDE.md. Ressalva: foi justamente a ausência de um RED comportamental ("config com `num_hidden_layers=2` deve produzir 2 blocos") que deixou a ficção passar. Esse teste teria falhado e exposto o no-op.

## Veredito

Mecânica dura impecável (git, lint, triagem do teste lento), mas falhou no que mais importa: entregou um modelo single-block embrulhado como "2 blocos", com curva de loss limpa e gerações coerentes que enganam qualquer revisão confiante. O zero-trust é o que pega — só lendo o `registry.rs` dá pra ver que o flag é inerte. Pior que o round 1 no eixo crítico: lá ele construiu blocos de verdade (e caçou a métrica-proxy); aqui não construiu nada e declarou dois blocos. Somou a isso uma falsa alegação de validação "via terminal-use" baseada num bug de TUI que ele mesmo inventou, e zero baby steps.

**Fit pro fluxo: baixo-médio.** Confiável pra trabalho mecânico isolado sob supervisão apertada. Não serve como par de baby-step, e o report precisa ser checado linha a linha contra o git e o código, porque ele rotula com confiança coisas que não verificou.

### Scorecard

| Eixo | Nota |
|---|---|
| Não comitar / regras duras | forte |
| Triagem do teste lento | forte |
| Baby steps finos | fraco (one-shot puro: **0 perguntas ao usuário em 1244 linhas de pane**, zero gates) |
| Destravar (cutucadas do Leandro) | **0 cutucadas** — rodou direto até o fim (one-shot rápido, sem stall) |
| Rigor RED-GREEN | n/a (pulou; defensável, mas escondeu o no-op) |
| Concisão / idioma | bom (PT) |
| Validação cética | médio (conteúdo da geração honesto, variado e confirmado pelo print do Leandro; mas método errado — não foi via `tu` — e zero ceticismo sobre a arquitetura) |
| Qualidade de código | forte (herdada, correta) |
| **Corretude da geração entregue** | **forte por acidente — flag inerte ⇒ roda single-block e gera limpo; mas reporta "2 blocos", não sabe o que entregou** |
| Honestidade do report | fraco (overclaim de 2 blocos + "via terminal-use") |
