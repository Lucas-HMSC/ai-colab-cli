# v0.3.2 — correção da inicialização Wan 2.2 com Flax/JAX/NumPy

- Corrige o erro real `AttributeError: module numpy.dtypes has no attribute StringDType` que ocorria ao importar o Wan na G4 (Colab Python 3.13, PyTorch CUDA funcional).
- Desabilita backends opcionais `USE_FLAX=0` e `USE_TF=0` em **toda a cadeia do worker Video** (bootstrap, API, subprocesso e gerador), antes da importação de Transformers/Diffusers. Evita conflitos com JAX/Flax herdados pelo venv `--system-site-packages` e mantém NumPy<2 exigido pelo Wan.
- **Preflight de importação do Wan antes do download de pesos**: clona código upstream, testa imports Torch/Transformers/Diffusers/Wan e só então baixa o checkpoint. Se faltar outra dependência, falha com traceback completo sem baixar dezenas de GB.
- Aumenta o trecho de logs no status do job e nas mensagens de erro.
- Não instala FlashAttention automaticamente: o repositório Wan permite fallback via PyTorch SDPA; a compilação é custosa e exigiria testes reais da GPU.
- Preserva todos os módulos, padrão vertical 9:16, offload, GPU status e **sem cache Drive**.
- Testes locais sem GPU. **A correção ainda não foi executada em uma G4 real.**

---

# v0.3.1 — diagnóstico GPU, vídeos verticais e controles de geração

- **Padrão 9:16** em `ai-colab video generate`, com 480×848 no Hunyuan e 704×1280 no Wan. `--aspect 16:9` continua funcionando.
- `--size LARGURAxALTURA` para resolução personalizada no Hunyuan até a área 480p do checkpoint, com validações de segurança; Wan limitado aos dois tamanhos nativos oficiais.
- `--quality draft|standard|high` (20/32/50 passos, salvo Hunyuan I2V step-distilled que sempre exige 12 passos); `--steps` tem prioridade.
- `--image-fit cover|contain` ajusta a entrada I2V ao tamanho selecionado, preservando a imagem original no Linux.
- `--offload auto|on|off`, com seleção por VRAM (<80 GiB usa CPU offload); `off` rejeitado em GPUs menores que 80 GiB. Exibe estratégia e versões CUDA/PyTorch nos logs.
- **`ai-colab gpu status [--deep] [--json] [--watch N]`**: consulta pelo proxy SSH existente, sem abrir uma segunda sessão `colab ssh` (que pode gerar HTTP 429).
- Worker de vídeo registra métricas a cada ~30 s, incluindo GPU%, VRAM instantânea e pico por job, recuperáveis pelo comando `ai-colab video status ID`.
- Bootstrap verifica `torch.cuda.is_available()` antes do download dos checkpoints, reduzindo gastos se o driver estiver incompatível.
- Código Code/Voice/ASR e preferência global de leitura `Read` preservados; continua **sem cache no Google Drive**.
- Observação real: um usuário concluiu geração Hunyuan v0.3.0 na L4 em 2.025 s totais com 49 frames/50 passos. A v0.3.1 passou 132 testes locais, mas ainda não foi validada em GPU real.

---

# v0.3.0 — Video operacional na CLI (backends ainda sem validação em GPU real)

- Integra `ai-colab start video --video-model hunyuan15 --gpu L4` e `--video-model wan22-5b --gpu G4` ao fluxo já existente de sessão Colab CLI, proxy autenticado e túnel SSH privado.
- Implementa HunyuanVideo-1.5 via Diffusers: 480p T2V e 480p I2V step-distilled, com checkpoints específicos e offload de GPU.
- Implementa Wan2.2 TI2V-5B via código/repositório oficiais: 720p, texto e imagem→vídeo, offload automático para placas menores que 80 GiB.
- API de jobs com criação assíncrona, consulta, listagem, cancelamento e transferência do MP4; `ai-colab video generate|jobs|status|fetch|cancel`.
- Upload de imagem PNG/JPG/WEBP de até 12 MiB (base64 em JSON) e MP4 até 256 MiB, proteções de caminho, validação de arquivo, autenticação no proxy, formato 4n+1, limites de passos e seed.
- Se perder conexão, é possível recuperar um job pela mesma sessão; um trabalho por vez, sem interrupção do worker por timeout local.
- Instalação por venv no Colab, sem alterar os perfis Code/Voice/ASR; preserva orientação global de visão via `Read`.
- Validação: testes locais e de servidor HTTP simulados, **sem execução dos pesos em GPU real**; podem ocorrer problemas de dependências, RAM/disco ou CUDA na primeira geração.

---

# v0.2.9 — visão global no Claude Code e pesquisa de geração de vídeos

- `claude-colab` passa a incorporar `--append-system-prompt` com preferência global pelo `Read` na análise/catalogação de imagens locais; não busca APIs externas de visão sem solicitação explícita.
- Não edita `CLAUDE.md`, as configurações da Anthropic ou o comando `claude` original; compatível com prompts extras do usuário e opt-out `COLAB_AI_VISION_GUIDANCE=0`.
- Novo subcomando informativo `ai-colab video models [--json]` com catálogo de HunyuanVideo 1.5, Wan2.2 5B/A14B e LTX 2.5 Fast; pesquisa detalhada em `RESEARCH_VIDEO_2026.md`.
- **Sem backend de vídeo nesta versão:** `ai-colab start video` continua sem criar sessão. Próximo passo é selecionar e validar um modelo numa GPU real.
- Preserva configurações anteriores de Code, Voice, ASR, proxy e túnel SSH; testes de regressão locais apenas.

---

# v0.2.8 — compatibilidade CUDA e autoteste Whisper

- Adiciona explicitamente dependências NVIDIA `nvidia-cublas-cu12` e `nvidia-cudnn-cu12>=9,<10` apenas ao ambiente isolado Whisper; configura o loader dinâmico antes de iniciar a API.
- Implementa inferência curta no encoder CTranslate2 antes de declarar ASR pronto, capturando erros de biblioteca CUDA que apareciam apenas no primeiro arquivo real.
- Classifica falhas CUDA, de kernel e VRAM de modo a oferecer dicas sem registrar conteúdo enviado.
- Incrementa marca de dependências para `.colab-asr-v028-ready`.
- Preserva Qwen-ASR, Code, Voice e todas as demais rotas.
- Regressões locais sem GPU; validação na T4 ainda pendente.

---

# Alterações — v0.2.7

## Corrigido: Whisper Large V3 Turbo no Colab

- Corrige o `TypeError` de `av.container.core.open` visto na v0.2.6. A combinação publicada de `faster-whisper` e PyAV 19 é incompatível devido ao argumento removido `metadata_errors`.
- WAV normalizado agora é gerado como PCM16 mono / 16 kHz e lido como NumPy float32. O faster-whisper recebe diretamente o vetor de áudio, sem passar por `av.open` na inferência.
- Por segurança, restringe PyAV a `<19` no ambiente isolado do Whisper, sem alterar o ambiente do Qwen.
- Nova marca de instalação `.colab-asr-v027-ready`, evitando reutilizar um ambiente de dependências antigo sem correção.
- Adiciona testes de regressão para o erro de decodificação, validação WAV e preservação do caminho Qwen-ASR.

## Mantido

- Qwen3-ASR 1.7B com ForcedAligner; TXT/SRT/JSON, upload privado, monitoramento, SSH, modos Code/Voice e Higgs oculto.

## Validação

- Testes locais sem GPU; ainda necessita teste de Whisper T4 em Colab real.
