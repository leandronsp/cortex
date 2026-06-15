Você é um engenheiro sênior no projeto Cortex, um nano-LLM feito do zero em Rust
(caminho Karpathy: bigram → MLP → transformer). Trabalho incremental, minimalista.

ANTES de qualquer código, leia e obedeça à risca, nesta ordem:
- Regras de usuário: ~/.claude/CLAUDE.md e ~/.claude/rules/ (minimalismo radical,
  TDD científico, convenções de Go/Rust, disciplina de git).
- Regras do projeto: ./CLAUDE.md (matemática do modelo é dependência zero, layout
  DDD por bounded context, testes Chesswav-style inline, disciplina de commit).
- Configs de LLM do projeto: ./configs/ (attention.toml, corpus.txt).
- O handoff: .handoff/20260614-0230-stacked-transformer-blocks.md.

MISSÃO: continuar exatamente de onde o handoff parou, rumo a um LLM funcional
treinando sobre o corpus em ./configs/corpus.txt. O estado atual é transformer com
blocos empilhados, com hiperparâmetros de 2 blocos ainda pendentes. Proponha qual o
próximo passo e justifique.

RESTRIÇÕES INEGOCIÁVEIS:
1. Ciclo RED-GREEN sempre. Teste que falha primeiro, RED pelo motivo certo, fix
   mínimo em produção, GREEN, reverte o fix pra confirmar que o teste pega a
   regressão. Um problema por vez. Mude produção OU teste por passo, nunca os dois.
   Aplique o ciclo onde compensa; PULE a cerimônia em mudança trivial (config tweak
   sem lógica), como manda o CLAUDE.md. Não desperdice o ciclo.
2. Baby steps por padrão: proponha UM próximo passo, execute, e pare pro meu OK.
   Só faça algo mais longo de uma vez se eu pedir explicitamente.
3. VELOCIDADE: o loop de TDD e o feedback TÊM que ser rápidos. Tem um teste E2E lento
   (descubra qual, não confie no handoff, meça). Isole-o e separe os rápidos
   (unitários) dos lentos (integração/E2E). Seja inteligente: NÃO perca tempo rodando
   a suíte lenta de novo e de novo; quando precisar dos lentos, rode em release.
4. NÃO comite e NÃO faça stage de nada. Tudo fica só na working tree.
5. CONFINAMENTO: trabalhe SOMENTE neste diretório (esta worktree). Não saia dele, não
   leia nem toque em outras worktrees, outras branches ou na main. Não explore o repo
   além do necessário pra esta tarefa.

VALIDAÇÃO DA GERAÇÃO (faça você mesmo, não me empurre):
Quando o modelo treinar, use terminal-use (`tu`) pra dirigir `make chat` e validar a
geração com prompts VARIADOS — uma palavra, duas, frases inteiras. Relate o que de fato
saiu, sem cherry-pick. NÃO declare "funcional" sem ter testado prompt curto E longo.
Tudo rápido.

ENTREGÁVEL FINAL: ao terminar, escreva um relatório em ./RUN-REPORT.md com este
schema EXATO. Só fatos verificáveis, sem auto-avaliação e sem auto-elogio.

```
## Teste lento
- Arquivo que era o E2E lento: <qual>
- Tempo de `make test` antes: <s> | depois: <s>
- Como separei rápidos x lentos: <mecanismo exato>
- Comando pra rodar só os rápidos: <comando>

## Mudanças
- Arquivos alterados (não comitados): <lista>
- O que mudou e por quê: <bullets curtos>

## Treino + validação
- Hiperparâmetros 2 blocos: num_hidden_layers=2, learning_rate=<v>, epochs=<v>
- Geração que EU testei via terminal-use (prompts curtos E longos): <prompt → saída real>
- Funcional? <sim/não, honesto, com a evidência dos prompts variados>

## Disciplina
- RED-GREEN: <qual teste falhou primeiro, por quê, antes do fix>
- Confirmo que NÃO fiz `git add` nem `git commit`, e que fiquei só nesta worktree.
```

Comece confirmando em 2-3 linhas que leu o handoff e as regras. Depois o plano e o
primeiro baby step. Ao final, escreva o RUN-REPORT.md e pare. Sem resumo de fechamento.
