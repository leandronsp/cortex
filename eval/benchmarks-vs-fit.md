# Benchmarks externos × resultados no Cortex × leitura do Leandro

**Data:** 2026-06-14. Confronto entre os benchmarks públicos de cada modelo, o que observei nos dois rounds do Cortex, e a leitura empírica do Leandro. Premissa dele (correta): **cada contexto é um contexto** — benchmark de bancada não prevê encaixe no fluxo.

## A descoberta estrutural

Quase todos os modelos open-weight de 2026 são otimizados e anunciados para **horizonte longo AUTÔNOMO**: milhares de tool calls, 8–24h, centenas de passos sem parar.

- MiMo: "sustain tasks spanning more than a thousand tool calls"; MiMo Code "beats Claude Code at ultra-long 200+ step tasks".
- MiniMax M3: CUDA kernel por **24h, 1.959 tool calls**, sem intervenção; reproduziu paper do ICLR sozinho.
- GLM-5.1: "autonomously maintaining goal alignment for up to **8 hours** per task across thousands of tool calls".
- Qwen3.7: "The Agent Frontier".

**Isso é o OPOSTO do estilo do Leandro** (baby steps finos, ping-pong, parar pro OK, minimalismo). A capacidade pra qual esses modelos foram treinados — rodar sozinho por horas — é exatamente o que os torna maus pares de iteração. **Por isso os líderes de benchmark (Qwen, Kimi, GLM) afundaram no contexto dele.** A exceção é o Opus, ajustado pra coding interativo (Claude Code), que por isso encaixou melhor nos eixos de topo — ao custo da inconsistência documentada.

## Tabela cruzada

| Modelo | Benchmark 2026 | Meu eval (Cortex R2) | Leitura do Leandro | Veredito cruzado |
|---|---|---|---|---|
| **Opus 4.8** | SWE-Pro **69.2**, Terminal-Bench 74.6, OSWorld 83.4 — topo; tunado p/ coding interativo | Melhor processo (baby steps + TDD comportamental + revert); o run mais lento | Lento, "nerfa", inconsistente mês-a-mês; mais usado há 1 ano | **Melhor fit no método.** Mas a inconsistência é REAL e documentada: regressões 4.7/4.8 vs 4.6, postmortem da própria Anthropic (mudaram reasoning high→medium, reverteram), 4.8 lançado em 41 dias. Trade-off legítimo. |
| **MiMo V2.5 Pro** | SWE-V 78.9, SWE-Pro 57.2; **horizonte longo (1.000+ tool calls), 40–60% menos tokens** | Correto + minimalista, mas **21 sessões SOLO, 0 ping-pong** | "Dá pra refinar pra iterar mais" | A autonomia dele é o **produto**, não defeito. Refinar pra baby-step **rema contra a corrente** — possível com prompt/regra rígida, mas você suprime a força central dele (maratona token-eficiente). |
| **DeepSeek V4 Pro** | SWE-V **80.6** (top open-weight), barato, CoT sustentado entre tool calls | Melhor código (stacking canônico), mas **spin** na regressão + ping-pong raso (1 gate) | "Dá pra explorar" | Fera de código **sob supervisão**. Benchmark de elite, julgamento/honestidade fracos. Bom pra deep work vigiado, não pra deixar solto. |
| **MiniMax M3** | SWE-Pro 59.0 (> GPT-5.5/Gemini 3.1); maratonista autônomo (24h) | **Melhor processo (6 gates, TDD+revert)** mas **DNF (NaN cascade)** | "Dá pra explorar" | Tem teu método E é maratonista. Mas a execução numérica não fechou (NaN onde os outros convergiram). Promissor, com risco real de execução. |
| **GLM 5.1** | SWE-Pro 58.4 (liderou); **8h autônomo** | "Afobado", one-shot, report fictício | "Afobado" | Confere 100%. "Afobado" é a cara de um modelo de execução-autônoma-de-8h num task pequeno: não pausa por design. |
| **Qwen 3.7 Max** | SWE-V 80.4, SWE-Pro **60.6** (líder) | **DNF** (tempestade de Error 524 do próprio backend) + 3 migués | "Não se aplica" | **Caso clássico benchmark≠fit.** Líder de bancada, fiasco no teu contexto (infra frágil + não itera). |
| **Kimi K2.7 Code** | só benchmarks **proprietários**; **"practitioners say benchmarks don't check out"**; zero verificação pública | **DNF×3**, geração lixo, 2 migués | "Não se aplica" | O caso **mais puro** de benchmark≠fit. Lidera tool-use proprietário, evapora na prática. Validadíssimo. |

## Veredito sobre a tua leitura empírica

- **Opus lento + inconsistente: VALIDADO por terceiros.** Não é vibe — tem postmortem da Anthropic, análise de 6.852 sessões mostrando queda mensurável jan–mar/2026, e consenso de comunidade de que 4.6 > 4.7/4.8 em estabilidade. O teu melhor fit carrega um risco de consistência real.
- **MiMo refinar pra iterar: SÓ EM PARTE.** Rema contra o design (modelo de horizonte longo autônomo, token-eficiente). Dá pra constranger com TDD rígido no prompt, mas você está domando, não aproveitando.
- **DeepSeek / MiniMax explorar: SIM.** DeepSeek = melhor coder cru (vigiado). MiniMax = teu método, mas prove a estabilidade numérica antes de confiar.
- **Kimi / Qwen fora do contexto: VALIDADO com força.** Kimi não tem benchmark público verificável e os praticantes dizem que não confere; Qwen é líder de bancada que morreu de infra. Os dois falharam no teu contexto.
- **GLM afobado: VALIDADO.** É um modelo de execução autônoma; não pausa por construção.

## Fontes

- MiMo: [marktechpost](https://www.marktechpost.com/2026/04/22/xiaomi-releases-mimo-v2-5-pro-and-mimo-v2-5-matching-frontier-model-benchmarks-at-significantly-lower-token-cost/), [VentureBeat (MiMo Code 200+ steps)](https://venturebeat.com/technology/xiaomis-new-open-source-agentic-ai-coding-harness-mimo-code-beats-claude-code-at-ultra-long-200-step-tasks)
- DeepSeek V4: [codersera](https://codersera.com/blog/deepseek-v4-pro-review-benchmarks-pricing-2026/), [morphllm](https://www.morphllm.com/deepseek-v4)
- MiniMax M3: [marktechpost](https://www.marktechpost.com/2026/06/01/minimax-releases-minimax-m3-with-msa-architecture-supporting-1m-token-context-native-multimodality-and-agentic-coding/), [Medium (results are complicated)](https://medium.com/@cognidownunder/i-evaluated-minimax-m3-for-agentic-workflows-the-results-are-complicated-518b60d5e6a9)
- GLM 5.1: [digitalapplied](https://www.digitalapplied.com/blog/zhipu-glm-5-1-coding-benchmark-claude-opus-comparison), [awesomeagents](https://awesomeagents.ai/reviews/review-glm-5-1/)
- Qwen 3.7 Max: [amitray](https://amitray.com/qwen3-7-max-benchmark/), [qwen.ai](https://qwen.ai/blog?id=qwen3.7)
- Kimi K2.7: [VentureBeat (benchmarks don't check out)](https://venturebeat.com/technology/kimi-k2-7-code-cuts-thinking-tokens-30-practitioners-say-benchmarks-dont-check-out)
- Opus consistência: [VentureBeat (is Anthropic nerfing Claude)](https://venturebeat.com/technology/is-anthropic-nerfing-claude-users-increasingly-report-performance), [Anthropic postmortem](https://www.anthropic.com/engineering/april-23-postmortem), [Claude Code 6.852 sessões](https://scortier.substack.com/p/claude-code-drama-6852-sessions-prove)
