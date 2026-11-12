# Colab AI CLI v0.3.2

Ferramenta Linux para operar modelos por demanda no **Google Colab Pro** através da **Colab CLI** e um **túnel SSH privado**. A v0.3.1 melhora a modalidade **Video** com HunyuanVideo-1.5 (L4) e Wan2.2 TI2V-5B (G4): T2V/I2V, jobs assíncronos, cancelamento, retomada de download e MP4 local. Mantém a orientação global Read e os backends anteriores de Code, Voice e ASR.

## Instalação

Pré-requisitos: Linux, Python 3, `uv`, Google Colab CLI autenticada, OpenSSH (`ssh` e `ssh-keygen`) e acesso a GPU no Colab. A GPU é criada apenas ao executar `ai-colab start ...`.

```bash
uv tool install google-colab-cli
colab --auth=oauth2 sessions
bash install.sh
ai-colab --version
ai-colab models
```

A atualização pode ser instalada sobre a versão anterior. **Não execute outra modalidade enquanto uma GPU estiver ativa**; encerre primeiro usando `ai-colab stop`. O modo Higgs permanece oculto. O cache Google Drive continua desativado; os downloads utilizam `hf_xet` com `HF_XET_HIGH_PERFORMANCE=1` por padrão.

## Claude Code: orientação global de visão (v0.2.9)

O `claude-colab` agora adiciona **`--append-system-prompt`**, uma orientação aplicada em **todos os projetos abertos por esse launcher**. Quando a tarefa envolve capturas de tela ou arquivos de imagem, ele instrui o agente a **preferir a ferramenta `Read`** do Claude Code para observar imagens nativas e não procurar chaves de APIs de visão, serviços externos ou OCR de terceiros, salvo solicitação expressa. A instrução orienta a catalogação em lotes e proíbe descrições inventadas de imagens não lidas.

- Aplica-se somente a `claude-colab` / `ai-colab claude`; **o `claude` original, o `~/.claude/settings.json` e os `CLAUDE.md` dos seus projetos não são modificados**.
- Usa a flag documentada `--append-system-prompt`, preservando o sistema do Claude Code, ferramentas e políticas originais. Se você passar sua própria flag `--append-system-prompt`, a orientação é mesclada em um único argumento.
- **Opt-out por sessão:** `COLAB_AI_VISION_GUIDANCE=0 claude-colab`. Também aceita `false`, `no` e `off`.
- A orientação **não é uma barreira técnica**: agentes podem escolher outras ferramentas; a eficácia deve ser confirmada com um arquivo PNG/JPG real na GPU.
- Para ver a regra imediatamente, inicie **uma conversa nova** em `claude-colab`. Em conversas retomadas com `--continue`/`--resume`, o Claude Code pode manter o prompt de sistema registrado até a compactação; em versões recentes, `--system-prompt-snapshot off` força a reconstrução. Consulte [a documentação oficial](https://code.claude.com/docs/en/cli-reference#system-prompt-flags-in-resumed-conversations).

```bash
ai-colab start code --gpu G4 --context 262144
claude-colab
# Dentro da sessão, peça para catalogar imagens de uma pasta normalmente.

# Opcional: desativar somente a orientação visual nessa execução:
COLAB_AI_VISION_GUIDANCE=0 claude-colab
```

## Vídeos: HunyuanVideo-1.5 e Wan2.2 TI2V-5B (v0.3.1)

**Os dois backends estão implementados; o Hunyuan v0.3.0 já concluiu uma geração real de 49 quadros e 50 passos no ambiente do usuário em 2.025 segundos (33 min e 45 s, incluindo o preparo e os downloads). As alterações da v0.3.1 não foram testadas ainda em GPU real.** O Wan também permanece sem validação real; possíveis incompatibilidades de dependências, VRAM ou disco precisam ser investigadas com logs. `ai-colab video models` lista também Wan A14B e LTX 2.5 como candidatos **não implementados**.

| Perfil | GPU para começar | Resolução / FPS | Texto→vídeo | Imagem→vídeo |
|---|---|---|---|---|
| `hunyuan15` | L4 | 480p, 24 fps | `HunyuanVideo15Pipeline` (50 passos sugeridos) | `HunyuanVideo15ImageToVideoPipeline` step-distilled (12 passos) |
| `wan22-5b` | G4 | 1280×704 ou 704×1280, 24 fps | Mesmo checkpoint TI2V-5B | Mesmo checkpoint com `--image` |

O Hunyuan usa **dois conjuntos de pesos distintos** no Hugging Face (`hunyuanvideo-community`); seu checkpoint Diffusers 480p T2V pode ocupar **mais de 50 GB** em disco. O Wan usa o repositório oficial, clonando o código e transferindo `Wan-AI/Wan2.2-TI2V-5B` no primeiro trabalho. Sem cache permanente no Google Drive, download e instalação podem se repetir após desligar a VM. Ambos geram **MP4 sem áudio**.

### 1. Instalar e iniciar Hunyuan (L4)

```bash
# Na pasta extraída do ZIP, primeiro instale:
bash install.sh
ai-colab --version

# Iniciar servidor de jobs (os pesos são baixados na primeira geração):
ai-colab start video --video-model hunyuan15 --gpu L4 --wait 1800

# Primeiro teste curto: 49 frames (aprox. 2 segundos a 24 fps):
mkdir -p ~/Videos
ai-colab video generate \
  --prompt "A pequena nave futurista cruza um campo ao pôr do sol, câmera cinematográfica" \
  --frames 49 --quality draft \
  --output ~/Videos/hunyuan-teste-vertical.mp4
```

Para **animar uma imagem** com o mesmo perfil Hunyuan:

```bash
ai-colab video generate \
  --image ~/Pictures/foto.png \
  --prompt "Movimento sutil de câmera, vento suave, movimento natural" \
  --frames 49 --steps 12 --output ~/Videos/hunyuan-animacao.mp4
```

A I2V Hunyuan é a versão **step-distilled** e exige exatamente 12 passos; na v0.3.1 a imagem de entrada é ajustada para a dimensão solicitada, com corte central (`--image-fit cover`) ou preenchimento sem corte (`--image-fit contain`). O original no Linux não é alterado. **Usar Hunyuan em T4 não é recomendado neste projeto** (a implementação usa BF16; selecione L4/G4). Além da VRAM, pode precisar de muita RAM e espaço em disco para checkpoint e offload de CPU. Evite usar em uma instância com pouco armazenamento. O modelo possui termos de licença da Tencent que precisam ser considerados para uso comercial.

### 2. Iniciar Wan2.2 TI2V-5B (G4)

Ao trocar de perfil, encerre a GPU anterior para evitar conflito:

```bash
ai-colab stop
ai-colab start video --video-model wan22-5b --gpu G4 --wait 1800

ai-colab video generate \
  --prompt "Um beija-flor azul voa entre flores ao amanhecer, luz natural" \
  --frames 49 --quality draft \
  --output ~/Videos/wan-teste-vertical.mp4
```

Para **imagem→vídeo** no mesmo checkpoint:

```bash
ai-colab video generate \
  --image ~/Pictures/referencia.jpg \
  --prompt "A câmera se aproxima lentamente, objetos com movimentos sutis" \
  --frames 49 --steps 50 --aspect 9:16 \
  --output ~/Videos/wan-animacao.mp4
```

Wan usa `--offload_model`, `--convert_model_dtype` e `--t5_cpu` automaticamente quando a GPU tem menos de 80 GiB; na G4 de ~96 GB, tenta executar sem offload. O `--aspect` define 1280×704 ou 704×1280; em I2V, a imagem é preparada nessa mesma dimensão para evitar mudança inesperada de proporção. O corte central pode alterar o enquadramento (use `--image-fit contain` para preservar todo o original). Uma L4 com offload é possível em teoria, mas é significativamente mais exigente; comece pela G4.

### Novidades v0.3.1: vídeo vertical, qualidade e dimensões

**Formato padrão: vertical `9:16`.** Para reproduzir um vídeo horizontal antigo, informe `--aspect 16:9` explicitamente. Os tamanhos nativos são:

| Modelo | Padrão 9:16 | Horizontal 16:9 | Resolução personalizada |
|---|---|---|---|
| Hunyuan 1.5 (checkpoint 480p) | 480×848 | 848×480 | `--size 448x768` por exemplo (múltiplos de 16, mínimo 256 px/lado, área até 848×480); experimental |
| Wan TI2V-5B (checkpoint 720p) | 704×1280 | 1280×704 | Apenas os dois tamanhos oficiais são aceitos |

**Qualidade por passos:** `--quality draft` usa 20, `--quality standard` 32, `--quality high` 50 (padrão); `--steps N` sempre tem prioridade sobre o preset. Menos passos tende a reduzir tempo de inferência, mas não há garantia de velocidade proporcional nem da mesma qualidade visual. **Hunyuan I2V destilado sempre usa 12 passos**, independentemente do preset.

```bash
# Vertical padrão, rascunho para economizar unidades:
ai-colab video generate --prompt "Flores ao vento, câmera lenta" \
  --quality draft --frames 49 --output ~/Videos/vertical-rascunho.mp4

# Vertical Hunyuan com resolução menor (o checkpoint permanece 480p):
ai-colab video generate --prompt "Cachoeira na floresta" \
  --size 448x768 --quality standard --output ~/Videos/cachoeira.mp4

# Horizontal Wan com dimensões oficiais:
ai-colab video generate --prompt "Uma estrada de montanha" \
  --aspect 16:9 --quality high --output ~/Videos/estrada.mp4
```

**Offload:** `--offload auto` (padrão) coloca partes do modelo na CPU quando a VRAM total da GPU é inferior a 80 GiB (ex.: L4). `--offload on` força essa economia de VRAM. `--offload off` requer pelo menos 80 GiB no worker e pode provocar OOM mesmo acima desse limite; use com cuidado. O nome e a VRAM da GPU, a disponibilidade de PyTorch/CUDA e a estratégia escolhida aparecem no log de inferência. `offload CPU` **não significa que toda a geração esteja rodando exclusivamente na CPU**.

**Imagem de entrada:** `--image-fit cover` (padrão) recorta centralmente para preencher o formato; `--image-fit contain` preserva a imagem inteira e adiciona bordas quando necessário. A entrada enviada não é regravada no computador. Para imagens com texto nas bordas, prefira `contain`.

### Diagnóstico da GPU sem uma segunda conexão SSH

A CLI mantém uma sessão SSH dedicada, e uma segunda chamada `colab ssh` pode receber HTTP 429. **Use o proxy já autenticado:**

```bash
ai-colab gpu status              # modelo da GPU, uso, VRAM, temperatura e processos
ai-colab gpu status --deep       # consulta opcional ao torch.cuda do ambiente remoto
ai-colab gpu status --json       # métricas estruturadas
ai-colab gpu status --watch 10   # acompanhamento contínuo sem reiniciar a GPU
```

Você pode executar esse comando em outro terminal enquanto o vídeo é gerado. `--deep` executa um pequeno subprocesso PyTorch remoto e pode levar alguns segundos; para monitoramento contínuo prefira apenas `--watch`.

**Atualizações em sessão ativa:** `bash install.sh` substitui arquivos locais, mas **não substitui o worker remoto que já estava rodando na v0.3.0**. Para ativar o novo monitoramento integrado aos jobs e a preparação de imagens, baixe o MP4 já pronto, encerre a sessão com `ai-colab stop` e inicie outra com a v0.3.1. O comando `gpu status` básico pode funcionar imediatamente em sessão antiga porque usa métricas já suportadas pelo proxy anterior; `--deep` exige o proxy novo. Encerrar uma instância remove seus pesos e MP4 efêmeros, pois não usamos Google Drive como cache.

### 3. Acompanhar e recuperar trabalhos

O cliente imprime o identificador do trabalho imediatamente após o envio; durante a geração, consulta status a cada 10 segundos, sem manter uma conexão HTTP aberta. Se você pressionar Ctrl+C **na espera da CLI**, o servidor remoto poderá continuar gerando vídeo.

```bash
ai-colab video jobs                  # listar todos os jobs na sessão
ai-colab video status ID_DO_TRABALHO # estado e fim dos logs da geração
ai-colab video fetch ID_DO_TRABALHO --output ~/Videos/resultado.mp4
ai-colab video cancel ID_DO_TRABALHO # interromper uma geração ativa
```

Se o túnel SSH cair, execute `ai-colab reconnect` e consulte `ai-colab video jobs` novamente. **Baixe o MP4 antes de encerrar a GPU**; os vídeos ficam no diretório efêmero `/content/colab-ai-videos/` e desaparecem com o runtime. O histórico em memória dos jobs não sobrevive à reinicialização do worker, mesmo se a VM permanecer ativa.

Para diagnóstico:

```bash
ai-colab test # verifica worker/health, não executa a geração de vídeo
ai-colab logs --source bootstrap --lines 150
ai-colab logs --source video --lines 150
ai-colab stats
ai-colab gpu status  # nvidia-smi pelo mesmo túnel: NÃO abre outro colab ssh
ai-colab gpu status --deep  # também verifica CUDA do PyTorch
ai-colab gpu status --watch 10  # a cada 10 segundos
```

O status de cada job inclui o trecho final de seus logs, as dimensões do vídeo, o modo de offload solicitado e amostras de uso da GPU/VRAM (registradas a cada 30 s) com pico de memória. **Não reinicie automaticamente em caso de erro**: primeiro consulte logs e disponibilidade de VRAM, RAM e disco. Se a CLI do Google retornar HTTP 503 na etapa `colab new`, os scripts de vídeo nem terão iniciado: a capacidade de criar GPU pelo navegador não garante que a CLI oficial consiga provisionar a mesma GPU.

**Limites e privacidade:** somente um job ativo por VM; PNG/JPG/WEBP de até 12 MiB; 17–121 frames no formato `4n+1`, máximo de 60 passos, MP4 até 256 MiB, 8 jobs por sessão. O envio da imagem e o download do vídeo usam o túnel SSH autenticado, com portas remotas em `127.0.0.1`. A imagem temporária é excluída após a geração; o MP4 permanece na VM até encerrar a sessão. O comando oficial do Wan pode incluir o *prompt* textual nos logs do job; não utilize dados confidenciais sem avaliar essa retenção e os termos da plataforma.

Fontes de implementação: [Hunyuan 1.5 Diffusers](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5#usage-with-diffusers), [checkpoint I2V destilado](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled), [Wan 2.2 oficial](https://github.com/Wan-Video/Wan2.2), [Wan TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

## Transcrição: dois perfis

| Perfil CLI | Modelo no Hugging Face | GPU padrão | Engine |
|---|---|---|---|
| `whisper` | `deepdml/faster-whisper-large-v3-turbo-ct2` | T4 | `faster-whisper` (CTranslate2) |
| `qwen` | `Qwen/Qwen3-ASR-1.7B` | L4 | `qwen-asr` + `Qwen3-ForcedAligner-0.6B` |

São GPUs de partida, não obrigatórias. Whisper produz segmentos de fala com timestamps próprios. Qwen usa alinhamento forçado oficial para fornecer tempos de palavras e permitir legendas. **Timestamps não são identificação de falantes**; diarização ainda não foi implementada.

### Teste 1: Whisper na T4

```bash
ai-colab start transcribe --asr-model whisper --gpu T4
ai-colab test                     # verifica API; não inventa áudio de teste
ai-colab transcribe --file ~/Downloads/amostra.mp3 \
  --output ~/Downloads/whisper.txt --format txt
ai-colab transcribe --file ~/Downloads/amostra.mp3 \
  --output ~/Downloads/whisper.srt --format srt
ai-colab stop
```

### Teste 2: Qwen3-ASR na L4

```bash
ai-colab start transcribe --asr-model qwen --gpu L4
ai-colab test
ai-colab transcribe --file ~/Downloads/amostra.mp3 \
  --output ~/Downloads/qwen.json --format json
ai-colab transcribe --file ~/Downloads/amostra.mp3 \
  --output ~/Downloads/qwen.srt --format srt
ai-colab stop
```

Use **o mesmo arquivo de áudio** nos dois perfis para comparar erros de palavras, sotaque, pontuação e tempo total.

### Todos os formatos em uma execução

```bash
ai-colab transcribe --file ~/Downloads/amostra.m4a --format all \
  --output ~/Downloads/teste-asr
```

Cria `teste-asr.txt`, `teste-asr.srt` e `teste-asr.json`. Se quiser substituir arquivos existentes, adicione `--overwrite`. O padrão é forçar português (`--language pt`); `--language auto` permite detecção automática do idioma. Para arquivos maiores, ajuste `--timeout 3600`.

Limites: upload de até **150 MiB**, duração de até **120 minutos**, formatos `wav`, `mp3`, `m4a`, `mp4`, `flac`, `ogg`, `opus`, `webm`, `aac`, `mov`, `mpeg`, `mpga`, `wma`. O servidor usa FFmpeg para mono PCM16 a 16 kHz antes da inferência. No Whisper, passa o áudio já decodificado como NumPy float32 para `faster-whisper`, evitando o erro de API `av.open(metadata_errors=...)` com PyAV 19. O ambiente também restringe `av<19` como defesa adicional.

## Diagnóstico e gerenciamento

```bash
ai-colab status --remote
ai-colab logs --source bootstrap --lines 80
ai-colab logs --source vllm --lines 100   # logs de ASR nesta modalidade
ai-colab stats --usage
ai-colab wait --seconds 1800
ai-colab reconnect
ai-colab stop
```

Se a inicialização terminar em erro, a GPU **não é automaticamente desligada**. Preserve os logs e execute `ai-colab stop` para evitar consumo desnecessário.

## Segurança e privacidade

A conexão remota usa exclusivamente um túnel SSH autenticado, associado à sessão da ferramenta; a porta de inferência escuta somente em `127.0.0.1` na VM. Arquivos ASR são transferidos em blocos; vídeos são baixados via streaming. Arquivos temporários de áudio no Colab são excluídos ao concluir a requisição, inclusive quando há erro tratado, e não são gravados textos completos em logs ASR; **o Wan pode registrar prompts de vídeo em seus logs de inferência**. Isso **não elimina** os riscos inerentes ao processamento em nuvem ou às políticas da plataforma. Para áudios confidenciais, obtenha autorização, avalie a necessidade de usar Colab e garanta um tratamento compatível com suas obrigações profissionais e legais.

## Instalação técnica

- Cliente Linux: `ai_colab.py` (biblioteca padrão).
- Inicialização remota: `remote/launch.py` → `remote/transcribe_bootstrap.py` / `remote/video_bootstrap.py`.
- Serviço ASR: `remote/transcribe_api.py` com Python **3.12 isolado** por modelo.
- Serviço Video: `remote/video_api.py`, `remote/video_generate.py`, Python isolado em venv com PyTorch da VM; tarefas assíncronas e HTTP privado.
- API privada: `POST /_colab_ai/transcribe`, com streaming de upload no proxy autenticado.
- Saída: estrutura JSON contendo texto, idioma e segmentos reais; o cliente salva TXT/SRT/JSON no disco Linux.

O Qwen utiliza a API oficial `Qwen3ASRModel.from_pretrained(...).transcribe(..., return_time_stamps=True)` e seu modelo de alinhamento forçado. O Whisper usa `WhisperModel(...).transcribe(...)`. O download depende do Hugging Face; a execução da v0.3.1 não foi validada diretamente na GPU (testes locais/simulados disponíveis).

## Modalidades preservadas

```bash
# Programação (sem mudanças funcionais)
ai-colab start code --gpu G4 --context 262144
claude-colab
ai-colab stop

# Voz (Qwen3-TTS, VoxCPM2, Chatterbox, Kokoro)
ai-colab start voice --voice-model voxcpm2 --gpu L4
ai-colab voice generate --text 'Olá!' --output voz.wav
ai-colab stop
```

O comando `claude-colab` preserva o modo de permissões anterior (incluindo `--dangerously-skip-permissions` por padrão). O comando `claude` original não é modificado.

## Fontes técnicas

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Whisper Turbo para CTranslate2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)

### Diagnóstico CUDA no Whisper (v0.2.8)

O worker do Whisper agora instala cuBLAS CUDA 12 e cuDNN 9 em um Python 3.12 isolado, estabelece o `LD_LIBRARY_PATH` antes da inicialização e realiza uma inferência curta de autoteste. O servidor **não será marcado como pronto** se o encoder falhar na primeira execução. Isso detecta casos em que os pesos carregam, mas falta `libcublas.so.12` / `libcudnn.so.9` ou existe incompatibilidade de kernel CUDA.

Após `ai-colab start transcribe --asr-model whisper --gpu T4`, aguarde a mensagem de API pronta. Se falhar, **não inicie novamente nem desligue imediatamente**: use `ai-colab logs --source vllm --lines 120` e `ai-colab logs --source bootstrap --lines 80`. A ferramenta agora informa a categoria do erro e dicas de correção. Os arquivos de áudio não são incluídos nos logs.

O Qwen3-ASR não foi alterado; continua sendo alternativa funcional até a validação do Whisper na GPU real.


## Wan 2.2 — correção do conflito Flax/JAX/NumPy (v0.3.2)

O Colab fornece Flax/JAX no Python do sistema. Como o ambiente de vídeo usa `--system-site-packages` para compartilhar o PyTorch CUDA da VM, esses pacotes também ficam visíveis. O Wan usa PyTorch e requer NumPy<2; uma combinação incompatível de JAX/Flax e NumPy interrompia o import antes do carregamento dos pesos.

Na v0.3.2, o serviço define `USE_FLAX=0` e `USE_TF=0` antes dos imports e faz um preflight do Wan **antes do download dos pesos**. Isso evita o erro observado, mas não substitui a validação real na G4.

```bash
ai-colab start video --video-model wan22-5b --gpu G4 --wait 1800
ai-colab video generate --prompt "Uma nave futurista ao pôr do sol" --frames 49 --quality standard --aspect 9:16 --output ~/Videos/wan-vertical.mp4
```

Se o preflight acusar outro pacote incompatível, envie o `log_tail` exibido ou o log completo antes de tentar novo download. A GPU deve ser reiniciada para carregar a nova versão do worker remoto; essa reinicialização perde os pesos efêmeros já baixados.
