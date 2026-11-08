# Changelog

## 0.2.0 — perfis Voice alternativos

- Mantidos Qwen3-TTS `custom`, `design`, `clone` e Qwen3.8 para programação.
- Adicionados VoxCPM2 (geração, design e clonagem), Chatterbox Multilingual PT-BR (checkpoint regional T3 sobre base V3), Kokoro 82M (três vozes brasileiras) e Higgs TTS 3 (geração experimental).
- GPUs padrão por perfil: L4 para Qwen/VoxCPM2/Chatterbox; T4 para Kokoro; A100 para Higgs experimental.
- Bootstrap remoto instala dependências somente da engine selecionada e usa hf_xet, sem Drive.
- Controles por engine: estilo, velocidade, temperatura, exagero expressivo, CFG e passos de inferência.
- Garantias: consentimento obrigatório para referências de voz, sem clonagem inventada em Kokoro/Higgs, uma engine por VM, WAV atômico, túnel privado.
- Novos testes simulados de roteamento e quatro engines; nenhum teste real em GPU foi executado.

## 0.1.9

- Qwen3-TTS 1.7B, hf_xet e retirada do cache persistente no Drive.
