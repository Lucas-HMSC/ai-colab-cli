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
