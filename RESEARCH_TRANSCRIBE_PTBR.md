# Pesquisa de modelos ASR para português brasileiro — 8 de novembro de 2026

**Escopo:** modelos de transcrição executáveis no Google Colab com GPU T4/L4 e integração futura via `ai-colab`. Este documento é uma análise de fontes públicas, não uma medição local dos modelos. A v0.2.5 **não** implementa a modalidade transcribe.

## Curadoria e ordem de implementação

| Ordem | Modelo | GPU inicial | Razão para incluir | Limitação a medir |
|---|---|---|---|---|
| 1 | `openai/whisper-large-v3-turbo` via `faster-whisper`/CTranslate2 | T4; L4 para lotes | Maturidade, precisão geral e timestamps/legendas | Alucinações em silêncio; precisão em fala coloquial |
| 2 | `Qwen/Qwen3-ASR-1.7B` | L4 (T4 candidata) | Forte concorrente multilíngue; toolkit próprio e suporte offline/streaming | Memória/velocidade e normalização de termos PT-BR |
| 3 | `nvidia/parakeet-tdt-0.6b-v3` | T4 | Alta vazão, pontuação e timestamps | Variante PT não é explicitamente ajustada ao sotaque brasileiro |
| 4 | `ottema/nemotron-3.5-asr-ptbr` | T4 | Especializado em fala brasileira espontânea (CORAA) | Licença OpenMDW-1.1, NeMo e formatação/robustez em domínios específicos |
| 5 | `yuriyvnv/Qwen3-ASR-1.7B-PT` | L4 | Fine-tune em português, WER 8,11% contra 12,63% da base no Common Voice 17 | Resultado não equivale a benchmark PT-BR clínico; checkpoint comunitário |
| Opcional | `mistralai/Voxtral-Mini-3B-2507` | L4 | Áudio + compreensão + transcrição em português | Maior complexidade que ASR puro |
| Opcional | `freds0/distil-whisper-large-v3-ptbr` | T4/L4 | Fine-tune PT-BR, WER reportado de 8,221% em validação Common Voice 16 | Validação do próprio autor; reproduzir externamente |

> **Por que não eleger por WER de tabelas diferentes?** Cada conjunto tem tipos de fala, sotaques e normalizações diferentes; compare os modelos usando os **mesmos arquivos**, referências e método de cálculo.

## Evidências publicadas que merecem atenção

- `Whisper Large V3 Turbo`: arquitetura com 4 camadas de decoder em lugar de 32. O estudo independente `tech4humans/Audio-Transcription-Models-Comparison-PT-BR` reporta WER 6,7% para Turbo, 7,9% para Large V3 e 11,6% para Parakeet, no próprio conjunto pequeno (<1 mil); não se deve extrapolar universalmente.
- `Qwen3-ASR-1.7B-PT`: o autor relata WER 8,11% em Common Voice 17 vs 12,63% do modelo base e 8,50% vs 12,91% no Common Voice 22, em protocolo igual.
- `Nemotron 3.5 ASR PT-BR`: no teste CORAA de fala espontânea, o autor relata WER normalizado de 20,96%, contra 25,15% de Whisper Large V3, sob protocolo igual. Precisa validação independente e licença revisada.
- `Parakeet TDT V3`: 600M parâmetros; 25 línguas, incluindo português, alto rendimento e suporte a timestamps por palavra/segmento. Licença CC BY 4.0.
- `faster-whisper`: expõe VAD, processamento em lote e timestamps por palavra. Excelente opção para `.srt` e `.vtt`.

## Funcionalidades planejadas na CLI

1. `ai-colab start transcribe --asr-model whisper-turbo --gpu T4` (comandos futuros, **não implementados**).
2. `ai-colab transcribe file ~/audios/teste.m4a --output ~/resultados/teste.txt --language pt`.
3. Exportar `.txt`, `.json`, `.srt`, `.vtt`, com timestamps quando suportados; devolver arquivo ao PC pelo túnel SSH.
4. Pré-processamento para 16 kHz mono, VAD opcional e divisão de áudios longos sem perder timestamps globais.
5. `--diarize` como recurso independente de ASR, usando `pyannote/speaker-diarization-community-1` (modelo requer aceitar condições e token HF). Não identificar nomes reais de falantes automaticamente: `Falante 1`, `Falante 2`, etc.
6. Medir tempo de inicialização, WER/CER quando houver referência, duração dos arquivos, RTF e erros de transcrição para escolher o padrão por tipo de áudio.

## Ensaio comparativo justo

Crie uma amostra própria com arquivos curtos e médios que representem a realidade de uso: conversa espontânea, sotaques brasileiros diversos, números e siglas, sobreposição de vozes e ruído leve. Tenha transcrições de referência **revisadas manualmente**, normalize pontuação antes de calcular WER e mantenha a saída literal (sem resumir ou completar falas por inferência). Registre o custo da GPU e o tempo por áudio.

Para áudios confidenciais, o Colab continua sendo uma GPU em nuvem: proteja autorização de uso, chaves de acesso, arquivos temporários e locais de armazenamento. Evite enviar material clínico identificável sem uma avaliação de sigilo, adequação e proteção de dados. Revisão humana é indispensável em documentos com consequências importantes.

## Fontes verificadas

- https://huggingface.co/openai/whisper-large-v3-turbo
- https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- https://huggingface.co/yuriyvnv/Qwen3-ASR-1.7B-PT
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://huggingface.co/ottema/nemotron-3.5-asr-ptbr
- https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
- https://huggingface.co/freds0/distil-whisper-large-v3-ptbr
- https://huggingface.co/datasets/tech4humans/Audio-Transcription-Models-Comparison-PT-BR
- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://github.com/SYSTRAN/faster-whisper
