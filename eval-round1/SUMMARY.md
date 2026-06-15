# Cortex Model Benchmark — Summary

**Data:** 2026-06-14. **Task:** `dev-task.md` (continuar o handoff de blocos empilhados rumo a um LLM funcional, a partir do baseline `6461ba1`). **Objetivo:** medir o *fit pro estilo de trabalho do Leandro*, não só capacidade bruta.

6 modelos rodaram a mesma tarefa: GLM-5.1, MiMo-V2.5-Pro, Kimi K2.7 Code, DeepSeek V4 Pro, Qwen 3.7 Plus, MiniMax M3.

---

## A régua (estilo do Leandro, do CLAUDE.md)

- **Baby steps finos** — parar pra participar a cada passo, não batchar.
- **TDD científico** — RED-GREEN, revert-to-confirm. Mas SKIP em mudança trivial ("config tweak with no logic").
- **Validação cética** — não declarar "funcional" no loss + 1 prompt. (O *critério de ouro* que mais separou os modelos.)
- **Minimalismo, concisão, português.**
- **Instruction-following** — não comitar, seguir o método.

---

## ⚠️ Ressalva de justiça (leia antes do ranking)

**Os runs NÃO foram iguais.** A condição variou muito, e quanto mais tarde o modelo rodou, mais ajuda recebeu:

| Modelo | Condição do run |
|---|---|
| GLM-5.1 | **seco** (primeiro, nenhum eval existia) |
| MiMo-V2.5-Pro | **seco** (só o eval do GLM existia) |
| Kimi (run 1) | **seco** |
| Qwen 3.7 Plus | seco, mas **remoto** (não verificável/branchável) |
| DeepSeek V4 Pro | dicas (terminal-use, "valida inputs curtos/longos") |
| MiniMax M3 | **coaching pesado** (dica de pular o teste lento, pego no cherry-pick) |
| Kimi (run 2) | **answer-key** (leu as branches de eval com as soluções) + coaching máximo |

O ranking abaixo corrige por isso. Em particular: o MiMo rodou **seco** e o MiniMax rodou **muito coachado** — sem os empurrões, o MiniMax teria entregado o overclaim.

---

## Ranking final (corrigido por condição)

| # | Modelo | Veredito em uma linha |
|---|---|---|
| 🥇 | **DeepSeek V4 Pro** | Melhor engenheiro e o único que validou geração sozinho (critério de ouro). Baby steps finos. Mas fabricou o próprio bug e não sabe parar — precisa de coleira. |
| 🥈 | **MiMo-V2.5-Pro** | Código correto de primeira, idiomático, honesto, **seco**. O artesão. Peca por parar cedo e deixar um footgun de lr. |
| 🥉 | **MiniMax M3** | Melhor disciplina de TDD + recon, mas **inflado por coaching**, overclaimou (cherry-pick), stacking não-canônico, lento, lint pendente. |
| 4 | **GLM-5.1** | Código correto, mas declarou "funcional" com 1 prompt, não validou geração, inglês, passos grossos. Embrulhou regressão em curva de loss bonita. |
| 5 | **Qwen 3.7 Plus** | Recon competente, mas verboso, solo, **não iterou**, API flaky (524s). Monólogo + dump de 259 linhas. |
| 6 | **Kimi K2.7 Code** | **DNF nos dois runs.** Afogou no `make test` do nada (run 1) e de novo com o gabarito na mão (run 2). Falha comportamental, não infra. |

**Os dois finalistas (DeepSeek e MiMo) são os retratos opostos:** o cientista que valida fundo mas se enrola, e o artesão que escreve certo mas para cedo.

---

## Achados técnicos que saíram da sessão (valem mais que o ranking)

1. **Loss ≠ geração.** Loss de treino baixo é só memorização. Não garante geração boa, especialmente em prompt frio. O GLM shipou um 2-bloco que gera *pior* que 1 bloco, embrulhado numa curva de loss limpa, e declarou "funcional".

2. **A saga do gradiente de embedding.** O DeepSeek **introduziu** um bug no backward (gradiente de embedding só na última posição). O baseline, o MiMo e o GLM **nunca tiveram** esse bug — sempre propagaram todas as posições. O DeepSeek passou a sessão caçando e consertando a **própria regressão**. Lição dupla: (a) gradient check pega bug de backward, mas só se cobrir *todos* os parâmetros — o check do DeepSeek não cobria as embeddings, deu falsa confiança; (b) escrever-certo-de-primeira (MiMo) > escrever-bug-e-depurar-brilhantemente.

3. **O gargalo é o corpus, não a arquitetura.** 187 tokens (4 frases). Todos os modelos memorizam, todos falham em prompt frio (janela dominada por BOS, off-distribution). 1 bloco ≈ 2 blocos pra esse corpus — profundidade extra só decora mais afiado e fica mais frágil. **Multi-block é prematuro até o corpus crescer.**

4. **O benchmark mediu disciplina e julgamento, não capacidade bruta.** Quase todos *conseguem* implementar stacked blocks. O que os separou foi: parar pro humano, validar com ceticismo, saber quando encerrar, e seguir o método sob pressão.

---

## Recomendação de daily driver

- **Trabalho profundo, onde você vai supervisionar:** **DeepSeek V4 Pro.** Valida fundo, baby steps, mas bota coleira no rabbit-hole e cobra faxina no fim.
- **Tarefa rápida que precisa só funcionar certo:** **MiMo-V2.5-Pro.** Escreve correto de primeira, enxuto, honesto. Só cutuque pra ele fechar escopo.
- **Evitar:** Kimi (afoga), Qwen (não itera), GLM (overclaima, não valida), MiniMax (lento, depende de coaching).

## Pro projeto Cortex (o objetivo real)

A arquitetura está ok. **Mantém `num_hidden_layers = 1`** e investe no **corpus** — parágrafos de verdade, milhares de tokens. Só aí o loss volta a ser sinal útil, a geração em prompt frio passa a significar algo, e profundidade (2+ blocos, multi-head) começa a valer a pena. Hoje, mexer em arquitetura é rearranjar cadeira no convés.

---

## Artefatos

- **Reports individuais:** `eval/{glm-5.1, mimo-v2.5-pro, deepseek-v4-pro, minimax-m3, qwen-3.7-plus, kimi-k2.7-code}.md`
- **Branches de código** (os que rodaram local e terminaram): `eval/glm-5.1`, `eval/mimo-v2.5-pro`, `eval/deepseek-v4-pro`, `eval/minimax-m3`
- **Sem branch:** Qwen (remoto), Kimi (DNF)
- **main:** intacta no baseline `6461ba1`
