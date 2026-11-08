# Histórico de alterações

## 0.2.1 — compatibilidade real das engines Voice (08/11/2026)

Correções baseadas nos logs de execução enviados pelo usuário:

- **VoxCPM2:** ambiente virtual Python 3.11, sem dependências do Python 3.13 do Colab; opção `optimize=False` para evitar problemas associados a `torch.compile` na primeira inferência. Em caso de novo `TypeError`, a API informa tipo e localização da falha, sem incluir os dados enviados.
- **Chatterbox PT-BR:** Python 3.11 isolado; `chatterbox-tts==0.1.7` com `torchvision==0.21.0`, compatível com `torch==2.6.0` do pacote. Mantém ajuste de pesos brasileiros sobre Chatterbox Multilingual V3.
- **Kokoro:** Python 3.11 isolado, pacotes `kokoro==0.9.4` e `misaki[en]==0.9.4`, eSpeak NG para PT-BR.
- **Higgs 3:** retirado o uso incorreto de `transformers.pipeline`; adicionada execução experimental por `vLLM-Omni==0.32.0` em Python 3.12/CUDA 13, com servidor privado na porta 8002. G4 como padrão e requisito prudencial de ~70 GiB; novo `ai-colab logs --source omni`.
- **Qwen Voice:** preservado no Python original do runtime, já validado pelo usuário.
- **Colab CLI:** tratamento de `colab exec` sem resposta, verificando PID/log remoto antes de reportar falha. Estado da sessão é preservado; nenhuma nova GPU é criada automaticamente durante a tentativa de recuperação.
- **Code/Claude:** fluxo e contexto mantidos, sem alterações no vLLM de programação.
- **Downloads:** permanece `hf_xet`, sem cache no Google Drive.
- **Validação:** 71 testes locais sem GPU passaram. Os testes reais com esses quatro perfis corrigidos ainda não foram executados na conta Colab do usuário.

## 0.2.0 — perfis Voice alternativos

- Qwen3-TTS, VoxCPM2, Chatterbox PT-BR, Kokoro e Higgs TTS 3.
- Primeira implementação experimental dos perfis; apresentou incompatibilidades reais na instalação e execução que motivaram a v0.2.1.

## 0.1.9

- Qwen3-TTS 1.7B, hf_xet e retirada do cache persistente no Drive.
