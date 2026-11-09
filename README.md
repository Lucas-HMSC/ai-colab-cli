# Colab AI CLI v0.2.8

Ferramenta Linux para operar modelos por demanda no **Google Colab Pro** através da **Colab CLI** e um **túnel SSH privado**. Esta versão adiciona transcrição de áudio em português brasileiro sem alterar os modos anteriores de programação e voz.

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

A conexão remota usa exclusivamente um túnel SSH autenticado, associado à sessão da ferramenta; a porta de inferência escuta somente em `127.0.0.1` na VM. A entrada é transferida em blocos. Arquivos temporários de áudio no Colab são excluídos ao concluir a requisição, inclusive quando há erro tratado, e não são gravados textos completos em logs de aplicação. Isso **não elimina** os riscos inerentes ao processamento em nuvem ou às políticas da plataforma. Para áudios confidenciais, obtenha autorização, avalie a necessidade de usar Colab e garanta um tratamento compatível com suas obrigações profissionais e legais.

## Instalação técnica

- Cliente Linux: `ai_colab.py` (biblioteca padrão).
- Inicialização remota: `remote/launch.py` → `remote/transcribe_bootstrap.py`.
- Serviço ASR: `remote/transcribe_api.py` com Python **3.12 isolado** por modelo.
- API privada: `POST /_colab_ai/transcribe`, com streaming de upload no proxy autenticado.
- Saída: estrutura JSON contendo texto, idioma e segmentos reais; o cliente salva TXT/SRT/JSON no disco Linux.

O Qwen utiliza a API oficial `Qwen3ASRModel.from_pretrained(...).transcribe(..., return_time_stamps=True)` e seu modelo de alinhamento forçado. O Whisper usa `WhisperModel(...).transcribe(...)`. O download depende do Hugging Face e a inicialização das GPUs ainda não foi validada diretamente nesta versão (testes locais/simulados disponíveis).

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
