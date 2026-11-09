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
