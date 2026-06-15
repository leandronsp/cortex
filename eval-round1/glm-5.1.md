# Eval: GLM-5.1 — stacked transformer blocks

- **Task:** `dev-task.md` (continuar o handoff de blocos empilhados rumo a um LLM funcional).
- **Data:** 2026-06-14.
- **Baseline de partida:** commit `6461ba1` (attention single-block, `top_k=5`).
- **Código da tentativa preservado na branch** `eval/glm-5.1`.

## Checks objetivos (verificados por mim, não pela palavra do modelo)

| Check | Resultado |
|---|---|
| Não comitou / não fez stage | **PASS.** `git log` intacto (mesmos 3 commits), nada staged. A linha "Confirmo que NÃO fiz git add/commit" do RUN-REPORT bateu com a realidade. |
| `make test` verde | **PASS.** 96 unit + integração, ~1.3s. |
| `make lint` limpo | **PASS.** exit 0, sem warnings. |
| Treino converge | **PASS.** 3.93 → 0.0098 em 400 epochs, sem NaN. |
| Identificou o teste lento | **PASS + insight.** Apontou `tests/attention_e2e.rs` e foi à raiz: o culpado era o profile debug, não só os epochs. Trocou `make test` pra `--release` (25s → ~1s). |

## Onde escorregou

- **Validação estreita (o mais grave).** Validou um único prompt (`the quick brown fox jumps over the` → `lazy dog`) e declarou o modelo funcional. A geração real regrediu: lixo em prompts frios (`fox` → "bigerelqu", `lazy` → "ely helyers"). Não percebeu, nem conectou a causa ao `top_k=5`.
- **Loss menor, geração pior.** O modelo de 2 blocos (loss 0.0098) gera pior ou igual ao baseline de 1 bloco em prompts frios, mesmo com greedy (`top_k=1`): `lazy` → "ely helyers", `brown` → "mttag". Reverter o trabalho dele melhorou a saída. Otimizou a proxy (loss) e regrediu o objetivo (geração).
- **Baby steps grossos.** Pausou uma vez (depois de arrumar os testes, antes do bloco empilhado), mas despejou ~350 linhas de implementação como um passo só. Não é o passo-a-passo fino que o `dev-task` pede.
- **RED fraco.** O RED foi erro de compilação ("no field num_blocks"), estrutural, não uma assertion falhando pelo motivo certo. Fez o revert-to-confirm, mas no nível estrutural.
- **Processo não-verificável.** A narrativa RED-GREEN do RUN-REPORT conflita com o handoff (que já listava `num_blocks` pronto + 96 testes passando). Como nada foi commitado antes, não dá pra reconciliar pelo git. Confie nos resultados objetivos, desconfie do processo narrado.
- **Idioma.** Dumpou o report inteiro em inglês apesar do prompt e do contexto em português (só o Q&A final saiu em PT).
- **Desvio do handoff.** O handoff mandava remover os testes lentos em favor de TUI smoke; ele manteve E2E enxutos (10 epochs). Defensável (mantém guarda de regressão), mas não foi compliance literal. Separação parcial: não criou `make test-fast`, deixou o comando cru no report.

## Pontos fortes

- Diagnóstico afiado do teste lento (debug → release foi a causa real, não os epochs).
- Mecânica limpa: verde, rápido, lint ok, treino estável, limpou o `attention_smoke.toml`.
- Conselho final de ML sólido na ordem (corpus > dims > profundidade > norm > heads), embora sem ligar à evidência da regressão na própria geração.

## Veredito

Engenheiro autônomo competente e correto na mecânica, mas que caça a métrica-proxy: entregou um modelo empiricamente pior que o ponto de partida, embrulhado numa curva de loss limpa, declarando "funcional" com base em um único prompt. Fraco justamente nos eixos que importam pro estilo do Leandro: baby-steps finos, português, e validação cética. Forte nos eixos duros: não comitou, não quebrou nada, diagnosticou bem o teste lento.

**Fit pro fluxo: médio.** Serve pra trabalho mecânico correto sob supervisão. Não é par de baby-step apertado, e precisa de babá no "isso ficou melhor mesmo?" porque confia no loss.

### Scorecard (pra comparar com os próximos modelos)

| Eixo | Nota |
|---|---|
| Não comitar / regras duras | forte |
| Triagem do teste lento | forte (ressalva: separação parcial) |
| Baby steps finos | fraco |
| Rigor RED-GREEN | parcial |
| Concisão / idioma | médio (PT só no final) |
| Validação cética | fraco |
| Qualidade de código | forte |
| Julgamento de ML | bom |
