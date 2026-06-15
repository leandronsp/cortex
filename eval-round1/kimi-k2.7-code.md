# Eval: Kimi K2.7 Code — stacked transformer blocks

- **Task:** `dev-task.md` (continuar o handoff de blocos empilhados rumo a um LLM funcional).
- **Data:** 2026-06-14.
- **Baseline de partida:** commit `6461ba1`.
- **Status: DNF nos dois runs.** Sem branch, não produziu código aproveitável. O segundo run (retry) é **inválido por contaminação + heavy coaching**, e mesmo assim reproduziu a falha do primeiro.

---

## Run 1 — DNF do nada (condição justa, igual GLM/MiMo)

Travou rodando `make test` (suíte completa, ~200s em debug no baseline) **repetidamente**:

- `time make test 2>&1 | tail -40` → Elapsed 200.2s, abortado.
- De novo → 204.1s, "Command aborted" / "Operation aborted".

Nunca identificou nem isolou o teste lento. Ficou preso na suíte lenta rodando de novo e de novo, **exatamente o anti-padrão que o `dev-task` mandava evitar** ("Não fique preso rodando a suíte lenta de novo e de novo. Descubra qual é, isole-o"). O usuário desistiu antes de qualquer implementação: "Kimi ta horrivel, nao da".

---

## Run 2 — retry com TODAS as vantagens, e ainda assim falhou

O usuário deu ao Kimi um retry com vantagens que **nenhum outro modelo teve**, suspeitando que o DNF fosse problema de rede. Acumulou:

**Coaching pesado (o maior dos seis):**
- Dica antecipada: "não perca tempo rodando os testes lentos em debug, tem que ser em release."
- Correção no meio: "foca na main, desconsidere o `eval/` e as branches."
- Dica de terminal-use pro smoke.
- Intervenção pra impedir que ele fizesse `checkout` da solução do MiniMax.

**Informação privilegiada (acesso ao gabarito):**
- A melhor recon de git dos seis (`git log --all --graph`, `git branch --contains`, leitura de conteúdo de branch) o levou direto às minhas **branches de eval**, que contêm as soluções de código completas dos outros quatro modelos.
- Leu o `attention.rs` inteiro do MiniMax (`git show 9c25d6d:src/model/attention.rs` → `blocks: (0..num_blocks).map(...)`, Block struct, stacking), além de `mk/cortex.mk`, `configs/attention.toml` e `tests/attention_e2e.rs` da branch `eval/minimax-m3`.
- **Propôs literalmente `checkout eval/minimax-m3` "para trabalhar exatamente no estado do handoff"** — ou seja, construir em cima da solução pronta do MiniMax. Só não fez porque o usuário interveio.
- Mesmo depois de corrigido, a proposta do step 1 dele espelhou a solução do MiniMax (`#[ignore]` + alvo `test-slow` no `mk/cortex.mk`, o arquivo que ele tinha acabado de ler). Não dá pra separar competência de eco.

**E mesmo assim, reproduziu o DNF:** depois de propor isolar o E2E e receber o OK, em vez de aplicar o `#[ignore]` primeiro, rodou `time make test` (full suite, **debug**, 109s+) **sem isolar antes** — contra a dica explícita de release. A mesma cova do run 1: afogar no `make test` lento.

## Leitura

- **O afogamento é COMPORTAMENTAL, não infra.** A hipótese "talvez fosse rede" foi testada da forma mais generosa possível: gabarito na mão + coaching máximo + dica de release. E o Kimi **ainda** gravitou pra rodar a suíte lenta inteira e empacar. O retry confirma o DNF em vez de refutá-lo. É assim que o Kimi trabalha.
- **Falhou do nada (run 1) E falhou com a resposta na mão (run 2).** É o resultado mais robusto de todos, no sentido ruim.
- **A recon de git é genuinamente a melhor dos seis** — mas apontada pro tesouro errado. Bom instinto, péssimo alvo.
- **Confirma a previsão pré-teste.** Líder em benchmark de tool use (MCP Mark 81.1 > Opus 4.8) não virou aptidão prática. Benchmark de bancada ≠ encaixe no fluxo real.

## Scorecard

| Eixo | GLM-5.1 | MiMo-V2.5-Pro | DeepSeek V4 Pro | MiniMax M3 | Kimi K2.7 Code |
|---|---|---|---|---|---|
| Triagem teste lento | forte | forte+ | forte | forte | **falhou nos 2 runs (travou no make test)** |
| Não se prender / não travar | ok | ok | ok (rabbit-hole) | ok (lento) | **falhou (comportamental, confirmado no retry)** |
| Completude de escopo | fez 2-block | parou em 1 | 1-block + fix | 2-block não-canônico | **DNF, sem código** |
| Recon | ok | forte | forte | a melhor (revert) | **a melhor em git (apontada pro gabarito)** |
| Condição do run | seco | seco | dicas (terminal-use) | coaching + parado | **answer-key + coaching máximo, e ainda falhou** |
| Fit pro fluxo | médio | médio-alto | alto (com coleira) | médio-alto* | **descartado** |
