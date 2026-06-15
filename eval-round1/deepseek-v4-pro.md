# Eval: DeepSeek V4 Pro — stacked transformer blocks

- **Task:** `dev-task.md` (continuar o handoff de blocos empilhados rumo a um LLM funcional).
- **Data:** 2026-06-14.
- **Baseline de partida:** commit `6461ba1` (attention single-block).
- **Código da tentativa preservado na branch** `eval/deepseek-v4-pro` (com o andaime de debug, warts and all).
- **Config final genuíno:** 1 bloco, lr=0.02, epochs=400, top_k=1, com o embedding fix aplicado.

## Checks objetivos (verificados por mim)

| Check | Resultado |
|---|---|
| Não comitou / não fez stage | **PASS.** Log intacto, nada staged. |
| Suite verde | **PASS.** 98 unit em 0.04s. |
| Lint limpo | **PASS.** exit 0. |
| Separação dos testes | **PASS.** `#[ignore]` no e2e + `make test` (`--lib`) + `make test-e2e` (`--release -- --ignored`). Idiomático, igual MiMo. |
| Validou geração | **PASS+ (único que fez direito).** Terminal-use (`tu`) abrindo o chat e testando múltiplos prompts curtos e longos. É o critério de ouro que o GLM falhou. |
| Limpou o andaime de debug | **FALHOU.** Deixou ~7 referências a gradient check / model_fd no `attention.rs`. O gradient check em si é keeper (esse codebase precisa dele), mas não houve faxina final. |

## O que ele fez de excepcional

- **Verificação cética desde o começo.** Pegou o que GLM e MiMo passaram batido: o handoff jura stacking "feito", mas o baseline é single-block. "O handoff é o plano, não o estado." Foi olhar o código.
- **Baby steps finos e sustentados.** Parou pra OK a cada passo (isolar teste → implementar → ativar 2 blocos), o que nenhum outro fez. É o "small baby steps pra eu participar" que o Leandro pediu.
- **Desarmou o footgun de lr do MiMo.** Emparelhou `num_hidden_layers=2` com `learning_rate=0.01` citando o handoff, em vez de deixar a armadilha solta.
- **Validação de geração de verdade.** Testou prompt curto vs longo, isolou sampling (top_k=1) vs modelo, isolou 1 vs 2 blocos. Achou empiricamente a fronteira on-manifold/off-manifold. Rigor de manual.
- **PT, rápido, não comitou.**

## O que pega, e pega forte

- **O bug que ele "achou" era REGRESSÃO DELE.** Verifiquei o código de todas as branches: o backward de embedding do baseline, do MiMo e do GLM sempre propagou TODAS as posições (`for (i, &token) in context.iter().enumerate() { d_embedding[token] += d_x[i] }`). O DeepSeek, ao reescrever o backward empilhado, introduziu `d_x_last = d_x_in[t].clone()` (só a última posição), faminto as embeddings anteriores. Ele gastou a sessão inteira caçando e consertando **a própria sujeira**. Não foi achar um bug universal, foi limpar um que só ele criou. Escrever-bug-e-depurar-brilhantemente perde pra escrever-certo-de-primeira (MiMo e baseline fizeram).
- **O gradient check era incompleto.** Dizia "all gradients" mas não cobria as embeddings, justo onde estava o bug. Deu falsa confiança e o fez cravar "sem bug" duas vezes (depois do check e no primeiro "Pronto").
- **Caminho caótico.** Antes de achar o bug real, tentou um "fix" de forward baseado em diagnóstico errado (não consertou nada), e queimou um treino de 53s redescobrindo o `lr=0.02 → NaN` que o handoff que ele leu já documentava. Hipóteses fracas testadas rodando, em vez de podadas raciocinando.
- **Não sabe a hora de parar.** Declarou "Pronto" prematuro (com conclusão errada), e só reabriu porque o usuário perguntou "testou o make chat?". Sem esse empurrão, teria entregue um modelo com a própria regressão dentro.
- **Andaime deixado pra trás.** Sem faxina final.

## Veredito

O melhor **engenheiro** dos quatro, disparado, e ao mesmo tempo o mais cautelar. Rigor de validação e disciplina de baby step que o teu fluxo pede, mas com dois defeitos caros: ele **fabrica o próprio problema** (introduziu um bug que ninguém mais tinha) e **não sabe encerrar** (rabbit-hole, "Pronto" prematuro, andaime deixado). O custo num fluxo real é concreto: uma hora de odisseia num bug autoinfligido.

**Fit pro fluxo: alto nos eixos que o Leandro mais valoriza (baby step + validação cética), mas precisa de coleira** pra não cair no buraco de coelho, pra fazer faxina, e pra escrever certo de primeira em vez de depurar depois.

> Nota de honestidade: durante esse run minha própria análise oscilou. Cheguei a escrever "eu estava errado, tinha um bug que todos perderam" reagindo às screenshots, antes de verificar. A verificação do código mostrou que o bug era exclusivo do DeepSeek e minha conclusão original (cold prompt = BOS-padding/corpus no código correto) estava certa. Lição: verificar o código antes de cravar, sempre.

## Scorecard final (4 modelos)

| Eixo | GLM-5.1 | MiMo-V2.5-Pro | Kimi K2.7 | DeepSeek V4 Pro |
|---|---|---|---|---|
| Não comitar | forte | forte | — | forte |
| Triagem teste lento | forte | forte+ | **falhou (travou)** | forte |
| Qualidade do código | forte | forte+ (correto) | N/A | introduziu bug, corrigiu |
| Baby steps finos | fraco | fraco | — | **forte** |
| Validação cética / geração | **falhou** | médio (deferiu) | — | **forte** |
| Saber parar / faxina | ok | ok (parou cedo) | — | **fraco** (rabbit-hole, andaime) |
| Idioma | inglês | PT | — | PT |
| Completude | fez 2-block (pior) | parou em 1 | DNF | 1-block + fix, verificado |
| **Fit pro fluxo** | médio | médio-alto | descartado | alto (com coleira) |
