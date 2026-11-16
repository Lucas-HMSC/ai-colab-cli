# AI Colab CLI (`ai-colab`)

CLI para Linux que transforma GPUs sob demanda do **Google Colab** (via `google-colab-cli` + túnel SSH privado) em um servidor local unificado para **programação com agentes (Claude Code ou OpenCode)**, **síntese/clonagem de voz (TTS)**, **transcrição de áudio (ASR)** e **geração de vídeo**.

- **Zero dependências externas no host:** escrita apenas com a biblioteca padrão do Python 3.
- **Conexão privada:** todo o tráfego passa por um túnel SSH exclusivo em `127.0.0.1` com chave Ed25519 própria da ferramenta (sem `ngrok` e sem alterar suas chaves SSH pessoais).
- **Fluxo padronizado:** `ai-colab start <modalidade> --model <modelo>` → usar a modalidade → `ai-colab stop`.

---

## Como funciona

1. **`ai-colab start <modalidade> --model <modelo>`** aloca uma GPU no Colab, envia os scripts remotos de bootstrap e proxy autenticado, abre o túnel SSH em `127.0.0.1:18001` e inicia um *watchdog* local para manter o túnel ativo.
2. **Uso local transparente:** você consome todas as modalidades pelo próprio `ai-colab` (ex.: `ai-colab code`, `ai-colab tts`, `ai-colab asr`, `ai-colab video`).
3. **`ai-colab stop`** libera a GPU remota no Colab e remove o estado temporário local.

> [!IMPORTANT]
> O armazenamento da VM no Colab é efêmero e não utiliza Google Drive como cache. Sempre baixe seus arquivos gerados antes de executar `ai-colab stop` e encerre a GPU atual antes de iniciar outra modalidade.

---

## Pré-requisitos e instalação

### Requisitos
- **Linux** com **Python 3** (3.10+) e **OpenSSH** (`ssh` e `ssh-keygen`)
- **`uv`** instalado (para instalar a `google-colab-cli`)
- Conta **Google Colab** com acesso a GPU (ex.: Colab Pro / unidades de computação)
- **Claude Code (`claude`)** e/ou **OpenCode (`opencode`)** no `PATH` *(necessário apenas para a modalidade `code`; o OpenCode não exige conta da Anthropic)*

### Passo a passo

1. **Instale e autentique a CLI oficial do Google Colab:**
   ```bash
   uv tool install google-colab-cli
   colab --auth=oauth2 sessions
   ```

2. **Clone o repositório e execute o instalador:**
   ```bash
   git clone https://github.com/Lucas-HMSC/ai-colab-cli.git
   cd ai-colab-cli
   bash install.sh
   ```
   O script instala os arquivos em `~/.local/share/colab-ai/` e cria o executável `ai-colab` em `~/.local/bin/`. Certifique-se de que `~/.local/bin` está no seu `PATH`:
   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```

3. **Valide a instalação (nenhuma GPU é criada nesses comandos):**
   ```bash
   ai-colab --version
   ai-colab ssh-check
   ai-colab models
   ai-colab --help
   ```

---

## Catálogo de modalidades e modelos

Consulte este catálogo a qualquer momento no terminal com `ai-colab models`, `ai-colab models <modalidade>` ou `ai-colab models --json`.

| Modalidade | Alias | Modelos (`--model` / `-m`) | Aliases do modelo | GPU padrão | Observações |
|---|---|---|---|---|---|
| `code` | — | `qwen` *(padrão)* | `code`, `27b` | `G4` | `Qwen/Qwen3.8-27B-FP8` (131k contexto padrão) |
| `code` | — | `qwen-next` | `code-next`, `next`, `80b` | `G4` | `Qwen/Qwen3-Coder-Next-FP8` (32k contexto; experimental) |
| `code` | — | `qwen-next-a100` | `code-next-a100`, `next-a100`, `a100` | `A100` | `cyankiwi/Qwen3-Coder-Next-AWQ-4bit` (256k contexto na A100 80 GB; sem modo thinking) |
| `voice` | `tts` | `custom` *(padrão)* | `qwen` | `L4` | `Qwen3-TTS-12Hz-1.7B-CustomVoice` (`generate`) |
| `voice` | `tts` | `design` | — | `L4` | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` (`design`) |
| `voice` | `tts` | `clone` | — | `L4` | `Qwen3-TTS-12Hz-1.7B-Base` (`clone` com áudio + texto) |
| `voice` | `tts` | `voxcpm2` | `voxcpm` | `L4` | `openbmb/VoxCPM2` (`generate`, `design`, `clone`) |
| `voice` | `tts` | `chatterbox-br` | `chatterbox` | `L4` | `ResembleAI/Chatterbox-Multilingual-pt-br` (`generate`, `clone`) |
| `voice` | `tts` | `kokoro` | — | `T4` | `hexgrad/Kokoro-82M` (`generate`, vozes PT-BR) |
| `transcribe` | `asr` | `whisper` *(padrão)* | — | `T4` | `faster-whisper-large-v3-turbo-ct2` (segmentos) |
| `transcribe` | `asr` | `qwen` | `qwen-asr` | `L4` | `Qwen3-ASR-1.7B` + `ForcedAligner-0.6B` (tempos por palavra) |
| `video` | — | `hunyuan15` *(padrão)* | `hunyuan` | `L4` | `HunyuanVideo-1.5` 480p (T2V 50 passos / I2V 12 passos; sem áudio) |
| `video` | — | `wan22-5b` | `wan`, `wan22` | `G4` | `Wan2.2-TI2V-5B` 720p, 24 fps (T2V / I2V; sem áudio) |

> [!NOTE]
> `ai-colab video models` também exibe candidatos futuros ainda não implementados (`wan22-14b`, `ltx25`), e `image` aparece como modalidade planejada — nenhum deles aloca GPU ao ser consultado.

### Opções comuns do `ai-colab start`

```bash
ai-colab start <modalidade> [--model MODELO] [--gpu GPU] [--wait SEGUNDOS] [--hf-mode fast|normal]
```

- `--model` / `-m`: seleciona o modelo da modalidade (se omitido, usa o padrão da modalidade).
- `--gpu`: sobrescreve a GPU recomendada (ex.: `T4`, `L4`, `G4`, `A100`).
- `--wait`: tempo máximo de espera para download e carregamento inicial em segundos (padrão `900`).
- `--context`: tamanho da janela de contexto em tokens *(exclusivo da modalidade `code`)*.
- `--auth`: método de autenticação da `colab` CLI (`oauth2` padrão ou `adc`).

---

## Guia de uso por modalidade

### 1. Programação com Claude Code ou OpenCode (`code`)

Inicia um servidor vLLM compatível com as APIs Anthropic e OpenAI no Colab e conecta o cliente local escolhido através do túnel privado pelo comando `ai-colab code`:
- **Claude Code** (`ai-colab code` ou `ai-colab code --harness claude`): injeta variáveis de ambiente sem alterar seu `claude` padrão nem `settings.json`, e garante `hasCompletedOnboarding` em `~/.claude.json` em instalações novas.
- **OpenCode** (`ai-colab code --harness opencode`): **não exige conta da Anthropic**; registra automaticamente o provedor `colab-qwen` em `~/.config/opencode/opencode.json` preservando os demais provedores e inicia em `--standalone`. Se `claude` não estiver instalado no `PATH`, `ai-colab code` usa o OpenCode automaticamente quando disponível.

```bash
# Iniciar GPU com o modelo padrão (Qwen3.8-27B-FP8)
ai-colab start code --model qwen --gpu G4

# Alternativa para A100 80 GB (Qwen3-Coder-Next 4-bit AWQ, contexto máximo 256K)
ai-colab start code --model qwen-next-a100

# Opcional: ajustar janela de contexto (padrão: 131072 em code; 262144 em qwen-next-a100)
ai-colab start code --model qwen --gpu G4 --context 262144

# Abrir sessão com o cliente padrão (Claude Code, ou OpenCode se claude estiver ausente)
ai-colab code

# Forçar Claude Code ou OpenCode (sem conta Anthropic)
ai-colab code --harness claude
ai-colab code --harness opencode

# Passar flags extras diretamente ao cliente escolhido
ai-colab code -- -p "Explique a arquitetura deste repositório"
ai-colab code --harness opencode -- run "Explique a arquitetura deste repositório"

# Ao terminar, liberar a GPU
ai-colab stop
```

**Variáveis de ambiente do launcher:**
- `COLAB_AI_VISION_GUIDANCE=0`: desativa a orientação global do Claude Code que instrui o agente a inspecionar imagens locais com `Read` em vez de buscar APIs externas de visão/OCR.
- `COLAB_AI_SKIP_PERMISSIONS=0`: desativa `--dangerously-skip-permissions` (Claude Code) e `--auto` (OpenCode), preservando os prompts interativos de confirmação.

---

### 2. Síntese, Design e Clonagem de Voz (`voice` / `tts`)

Os comandos `voice` e `tts` são equivalentes. Escolha o modelo conforme a operação desejada:

- **`generate`** (voz predefinida): `custom`, `voxcpm2`, `chatterbox-br`, `kokoro`
- **`design`** (voz descrita por texto): `design`, `voxcpm2`
- **`clone`** (clonagem a partir de áudio de referência): `clone`, `voxcpm2`, `chatterbox-br`

```bash
# Listar modelos de voz sem alocar GPU
ai-colab tts models

# Exemplo 1: TTS rápido em PT-BR na T4 com Kokoro
ai-colab start tts --model kokoro --gpu T4
ai-colab tts generate \
  --text "Olá! Este áudio foi sintetizado no Google Colab." \
  --speaker pf_dora \
  --speed 1.0 \
  --output ~/voz-kokoro.wav
ai-colab stop

# Exemplo 2: Design + clonagem com VoxCPM2 na L4
ai-colab start tts --model voxcpm2 --gpu L4
ai-colab tts design \
  --description "Voz masculina grave, calma e com boa dicção" \
  --text "Bem-vindo ao projeto." \
  --output ~/locucao.wav

ai-colab tts clone \
  --reference ~/amostra-autorizada.wav \
  --reference-text "Transcrição do áudio de referência" \
  --consent \
  --text "Texto falado com a voz clonada." \
  --output ~/clonada.wav
ai-colab stop
```

**Detalhes importantes:**
- Aceita `--text "..."` ou `--file caminho.txt` (entre 1 e 1.800 caracteres por requisição; divida textos longos).
- `--speaker`: no Qwen (`custom`), aceita `Ryan`, `Aiden`, `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ono_Anna`, `Sohee`; no Kokoro, aceita `pf_dora`, `pm_alex`, `pm_santa`.
- `--clone` exige `--consent` confirmando autorização expressa do titular da voz (referência de até 4 MB; Qwen `clone` exige `--reference-text` não vazio).

---

### 3. Transcrição de Áudio e Vídeo (`transcribe` / `asr`)

Os comandos `transcribe` e `asr` são equivalentes.

```bash
# Listar modelos ASR sem alocar GPU
ai-colab asr models

# Iniciar com Whisper (T4) ou Qwen3-ASR (L4)
ai-colab start asr --model whisper --gpu T4

# Gerar TXT, SRT e JSON de uma só vez
ai-colab asr --file ~/Downloads/entrevista.mp3 \
  --format all \
  --language pt \
  --output ~/Downloads/entrevista

# Saída estruturada em stdout (sem gravar arquivo), com tempos por palavra e falantes
HF_TOKEN=hf_... ai-colab start asr --model whisper --gpu T4
ai-colab asr --file ~/Downloads/entrevista.mp3 --json --word-timestamps --diarize

# Encerrar a GPU após concluir
ai-colab stop
```

**Detalhes importantes:**
- `--format`: `txt` (padrão), `srt`, `json` ou `all` (gera os três arquivos). Use `--overwrite` para sobrescrever arquivos existentes.
- `--stdout`: imprime apenas a transcrição no `stdout` sem gravar arquivos em disco (incompatível com `--format all`).
- `--json`: com `--file`, equivale a `--format json --stdout` (ideal para automação/pipelines).
- `--word-timestamps`: inclui `words` e `word_timestamps: true` no JSON de resposta (suportado em `whisper` e `qwen`).
- `--diarize`: identifica falantes (`Falante 1`, `Falante 2`, ...) usando `pyannote/speaker-diarization-community-1`. Requer aceitar a licença do modelo e exportar `HF_TOKEN` antes de `ai-colab start asr` para provisionar `pyannote.audio` na VM.
- `--language`: `pt` (padrão, força português) ou `auto` (detecção automática).
- **Formatos aceitos:** `wav`, `mp3`, `m4a`, `mp4`, `ogg`, `oga`, `opus`, `flac`, `aac`, `webm`, `mov`, `mpeg`, `mpga`, `wma` (upload de até **150 MiB**).
- Nenhum conteúdo transcrito é gravado nos logs do servidor.

---

### 4. Geração de Vídeo (`video`)

Gera arquivos MP4 sem áudio em **texto→vídeo (T2V)** ou **imagem→vídeo (I2V)**. O formato padrão é vertical `9:16`.

```bash
# Consultar modelos e requisitos sem alocar GPU
ai-colab video models

# Iniciar HunyuanVideo-1.5 (L4) — ajuste --wait pois o download inicial é grande
ai-colab start video --model hunyuan15 --gpu L4 --wait 1800

# Texto -> Vídeo (rascunho rápido: 20 passos, 49 quadros ≈ 2s a 24fps)
ai-colab video generate \
  --prompt "Uma pequena nave futurista cruza um campo ao pôr do sol, câmera cinematográfica" \
  --frames 49 --quality draft --aspect 9:16 \
  --output ~/Videos/teste-vertical.mp4

# Imagem -> Vídeo (Hunyuan I2V step-distilled usa 12 passos; Wan usa 50 por padrão)
ai-colab video generate \
  --image ~/Pictures/referencia.png \
  --prompt "Movimento sutil de câmera, vento suave" \
  --frames 49 --steps 12 --image-fit contain \
  --output ~/Videos/animacao.mp4

# Ao finalizar os downloads, liberar a GPU
ai-colab stop
```

**Jobs assíncronos e retomada:**
- Se você pressionar `Ctrl+C` ou a conexão cair durante `video generate`, o job continua rodando na VM. Use:
  ```bash
  ai-colab reconnect
  ai-colab video jobs
  ai-colab video status <JOB_ID>
  ai-colab video fetch <JOB_ID> --output ~/Videos/resultado.mp4
  ai-colab video cancel <JOB_ID>
  ```
- **Dimensões e qualidade:**
  - `hunyuan15`: `480x848` (`9:16`), `848x480` (`16:9`) ou `--size LARGURAxALTURA` múltiplo de 16 (mín. 256 px, área até `848x480`). T2V aceita `--quality draft|standard|high` (`20`/`32`/`50` passos) ou `--steps 1..60`; **I2V step-distilled exige `--steps 12`**.
  - `wan22-5b`: aceita apenas `704x1280` (`9:16`) ou `1280x704` (`16:9`).
  - `--frames`: formato `4n+1` entre `17` e `121` (padrão `49`).
  - `--offload`: `auto` (padrão; ativa offload de CPU quando a GPU tem < 80 GiB), `on` ou `off` (exige >= 80 GiB).
  - `--image`: PNG/JPG/WEBP de até 12 MiB; `--image-fit cover` (recorte central, padrão) ou `contain` (preserva a imagem inteira com bordas).

---

## Monitoramento, diagnóstico e ciclo de vida

Nenhum dos comandos abaixo inicia uma nova GPU:

```bash
ai-colab status --remote               # Estado local + sessão no Colab
ai-colab status --json                 # Estado local, porta, token e runtime em JSON
ai-colab gpu status                    # Modelo da GPU, uso, VRAM, temperatura e processos
ai-colab gpu status --watch 10         # Acompanhamento contínuo a cada 10s
ai-colab gpu status --deep --json      # Inclui verificação PyTorch/CUDA em JSON
ai-colab stats --usage                 # GPU, RAM, disco e unidades de computação
ai-colab monitor                       # Monitor contínuo de recursos e unidades
ai-colab logs --source bootstrap --lines 120   # Logs de preparação (vllm | bootstrap | video)
ai-colab diagnostics                   # Logs locais do túnel/watchdog + estado remoto
ai-colab wait --seconds 1800           # Retoma espera de carregamento sem reiniciar GPU
ai-colab reconnect                     # Recria apenas o túnel SSH
ai-colab test                          # Testa saúde / inferência da modalidade ativa
ai-colab usage                         # Consulta unidades disponíveis no Colab
ai-colab hf status                     # Configura hf_xet (fast | normal | status)
ai-colab stop                          # Libera a GPU no Colab e limpa estado local
ai-colab stop --force-local            # Limpa apenas estado local se a VM já expirou
```

---

## Estrutura do projeto e testes

```text
ai-colab-cli/
├── ai_colab.py                # Cliente CLI local (stdlib Python 3)
├── install.sh                 # Instalador para ~/.local/share/colab-ai e ~/.local/bin
├── remote/                    # Bootstrap, proxy HTTP autenticado e workers de inferência na VM
│   ├── bootstrap.py / launch.py / serve.py / proxy.py
│   ├── voice_bootstrap.py / voice_api.py
│   ├── transcribe_bootstrap.py / transcribe_api.py
│   └── video_bootstrap.py / video_api.py / video_generate.py
└── tests/                     # Suíte de testes unitários e regressões
```

Para executar os testes locais da CLI (sem necessidade de GPU ou dependências remotas como `torch`/`soundfile`):

```bash
python3 -m unittest tests/test_cli_standardization.py tests/test_cli.py tests/test_catalog_v025.py -v
```
