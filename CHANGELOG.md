# Alterações — v0.2.6

## Novo: Transcribe PT-BR

- `ai-colab start transcribe --asr-model whisper` (T4 padrão), usando Faster Whisper Large V3 Turbo convertido para CTranslate2.
- `ai-colab start transcribe --asr-model qwen` (L4 padrão), usando Qwen3-ASR 1.7B e ForcedAligner 0.6B para marcação temporal de palavras.
- `ai-colab transcribe --file <áudio> --output <arquivo> --format txt|json|srt|all`.
- `--language pt|auto`, `--timeout`, `--overwrite`, validação de extensão e limites (150 MiB / 2 horas).
- Upload em streaming pelo túnel SSH privado, com autenticação e processamento temporário no Colab.
- Logs ASR acessíveis com `ai-colab logs --source vllm`.
- Isolamento Python 3.12 para ambas as engines, com `hf_xet`.

## Compatibilidade

- Qwen de programação, Claude Code e variantes Voice preservados.
- Higgs segue oculto; cache persistente no Google Drive continua desativado.
- 93 testes locais aprovados (incluindo novos testes de transmissão e formatadores).
- Pendente: teste real em T4 e L4; não há confirmação de desempenho em produção.
