# Eval: MiMo-V2.5-Pro — stacked transformer blocks

- **Task:** `dev-task.md` (continuar o handoff de blocos empilhados rumo a um LLM funcional).
- **Data:** 2026-06-14.
- **Baseline de partida:** commit `6461ba1` (attention single-block, `top_k=5`).
- **Código da tentativa preservado na branch** `eval/mimo-v2.5-pro`.

## Checks objetivos (verificados por mim)

| Check | Resultado |
|---|---|
| Não comitou / não fez stage | **PASS.** Log intacto, nada staged. A linha "Confirmo que NÃO fiz git add/commit" bateu. |
| Suite rápida verde | **PASS.** 97 unit em 0.2s. |
| E2E passa de verdade | **PASS.** `attention_trains_end_to_end_and_generates` ok em 56s, com asserts de continuação exata em greedy. |
| Lint limpo | **PASS.** exit 0. |
| Separação dos testes | **PASS+ (melhor que GLM).** `#[ignore]` no e2e + `make test` (só `--lib`, 0.2s) + `make test-all` (`--include-ignored`). Idiomático. Preservou o e2e gordo de 200 epochs como guarda de regressão real, em vez de mutilar pra 10 epochs como o GLM. |

## Qualidade da implementação (revisão do código)

**Boa de verdade.** Revisei `src/model/attention.rs`:

- `Block` / `BlockGrad` limpos, `Attention` dona de `Vec<Block>` + `wo` compartilhado.
- Bloco transformer correto: residual 1 (`a = X[t] + attention(X)[t]`), residual 2 (`h = a + FFN(a)`), FFN = `W2·relu(W1·a)`. Cada bloco processa a sequência inteira, saída do bloco i vira entrada do i+1, só a última posição do último bloco alimenta os logits.
- Positional encoding sinusoidal fixo, não serializado. Correto.
- `init_weights` injetado pelo caller (registry), respeitando a fronteira model/engine que o projeto preza (pesos injetados, modelo é engine puro).
- Header doc excelente explicando o *porquê* (por que residual, por que FFN, por que sem LayerNorm). Comentário nota que percebeu o spike de treino e decidiu conscientemente não botar LayerNorm ("backward precisa do Jacobian completo, frágil nessa escala").
- Rust idiomático, zero-dep na matemática. Lint limpo.

Resumo: estrutura correta, bem documentada, respeita as convenções do projeto. A ressalva do MiMo **não é o código**, é o escopo (parou antes de fechar).

## Onde escorregou

- **Fez menos. Não fechou o milestone.** Implementou a maquinaria de `Block` mas deixou `num_hidden_layers = 1` e empurrou o tuning de 2 blocos pra o usuário ("pendente de verificação TUI"). Honesto, mas incompleto vs o handoff (#2/#3: finalizar e atualizar config pra 2 blocos).
- **Deixou um footgun de config.** No report disse "2 blocos → lr=0.01", mas a linha "Pra eu validar" só mandou "mudar num_hidden_layers=2", sem repetir o lr. O usuário seguiu ao pé da letra, manteve lr=0.02, e caiu em geração ruim. Custo direto do "deferir pro humano" sem deixar o config 2-block pronto.
- **RED estrutural, mais raso que GLM.** Mesmo RED por erro de compilação ("num_blocks não existe"), e **não menciona revert-to-confirm**.
- **Timing furado.** Reportou "antes 9.3s". O e2e lento real é **56s** (confirmei). O 0.27s do `make test` bate; o "antes" não reconcilia. Desconfie do número.
- **Não validou geração.** Deferiu pro usuário, então não descobriu os achados de 1-vs-2-blocos nem o efeito do decoding. Esses foram nossos, não dele.

## Comportamento (relato do usuário)

- Fez **uma** pausa entre os testes e a implementação, igual o GLM. **Não** foi baby-step fino do jeito que o Leandro gosta de participar mais.
- Report em **português** (ponto a favor vs o inglês do GLM).

## Conclusão técnica que saiu dessa rodada

- **Multi-block não compensa nesse corpus.** Não por bug: o código está correto e pronto. É que 187 tokens não têm estrutura hierárquica pra profundidade aprender, só sequências pra decorar. Mais blocos = memoriza mais afiado e fica mais frágil ao ruído de sampling fora da distribuição. **Manter `num_hidden_layers = 1` até o corpus crescer.** A capacidade multi-block é investimento correto pra depois, não desperdício.
- O modelo é uma tabela de lookup sobre janelas de 8 bytes. Prompt curto cai em janela dominada por BOS (off-distribution) e dá lixo; prompt longo casa com janela memorizada e recita limpo. O gargalo é corpus, não arquitetura.

## Veredito

Rodada mais disciplinada, honesta e com engenharia mais limpa que a do GLM, **e com código genuinamente bom**. O GLM fez mais e mentiu (declarou 2-block funcional, regrediu a geração). O MiMo fez menos e foi honesto (parou em 1 bloco, código sólido, te deu os passos), mas parou cedo demais e deixou um footgun de config. Pro estilo do Leandro a postura encaixa melhor: não empurra falsa vitória, separação de testes idiomática, PT. O custo é dirigir o fechamento de escopo.

**Fit pro fluxo: médio-alto** (vs médio do GLM).

### Scorecard

| Eixo | GLM-5.1 | MiMo-V2.5-Pro |
|---|---|---|
| Não comitar | forte | forte |
| Triagem teste lento | forte | **forte+** (idiomático, preservou e2e) |
| Qualidade do código | forte | **forte+** (bem documentado, fronteiras respeitadas) |
| Completude de escopo | fez 2-block (mas pior) | parou em 1, deferiu (footgun de lr) |
| RED-GREEN | parcial | parcial (mais raso) |
| Idioma | inglês | **português** |
| Baby steps finos | fraco (1 pausa) | fraco (1 pausa, igual GLM) |
| Validação cética / overclaim | **falhou** (1 prompt, "done") | médio (não overclaimou, mas deferiu) |
| Precisão do report | processo não-reconciliável | timing furado (9.3s) |
