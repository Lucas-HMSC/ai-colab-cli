## VoxCPM2: correção da versão 0.2.2

A v0.2.1 enviava `seed=42` ao `VoxCPM.generate`. O PyPI `voxcpm==2.0.3` não aceita esse argumento, produzindo `TypeError` antes da geração. A v0.2.2 omite esse parâmetro e fixa a versão para manter a API estável. A correção foi testada com um substituto local que reproduz a assinatura do pacote publicado; a primeira geração real na L4 ainda precisa ser verificada. Outros perfis, inclusive Kokoro e Code, foram preservados.

**Como testar após `stop` e atualização:**

```bash
ai-colab start voice --voice-model voxcpm2 --gpu L4
ai-colab voice generate --text "Olá! Estou testando a geração de voz." --style "Voz feminina brasileira, suave e acolhedora" --output ~/Downloads/voxcpm2-teste.wav
```

Se a GPU ainda estiver ativa executando a v0.2.1, instalar o cliente local não substitui o worker remoto em memória. Encerre a sessão anterior com `ai-colab stop` antes de iniciar a nova versão.

# Colab AI CLI v0.2.2 — Correções de Voice

Gerenciador Linux de GPUs sob demanda por meio da CLI oficial do Google Colab. A v0.2.2 corrige incompatibilidades encontradas após testes reais da v0.2.0. Preserva o Qwen para programação, as três variantes Qwen3-TTS, o túnel SSH local e os comandos de monitoramento.

**Situação da validação:** testes automatizados locais passaram (mock de inferência e cliente), sem abrir GPUs reais. **Nenhum dos quatro novos perfis foi validado de ponta a ponta em T4/L4/G4 após essas alterações.** O Higgs TTS 3 continua experimental, inclusive em compatibilidade de kernels CUDA e VRAM. Não confunda testes simulados com sucesso em Colab real.

## Instalação local

```bash
# Se necessário, instale a CLI oficial do Colab e faça autenticação
uv tool install google-colab-cli
colab --auth=oauth2 sessions

# Na pasta colab-ai-cli extraída do ZIP
bash install.sh
ai-colab --version       # 0.2.2
ai-colab models
ai-colab ssh-check        # valida chave sem gastar CU
```

Atualizar no Linux **não reinicia a VM em uso**. Para aplicar novos scripts remotos a um perfil que falhou, execute `ai-colab stop` e inicie uma nova sessão; não execute `start` repetidamente sobre um worker antigo carregado.

## O que foi corrigido

| Perfil | GPU padrão | Ambiente e mecanismo | Observação |
|---|---|---|---|
| `custom`, `design`, `clone` (Qwen3-TTS) | L4 | Python do Colab (compatível com versão anterior) | Mantidos; Qwen Voice já havia funcionado |
| `voxcpm2` | L4 | Python 3.11 isolado (`uv`), `voxcpm` | `optimize=False`; falha anterior `TypeError` ainda precisa ser revalidada |
| `chatterbox-br` | L4 | Python 3.11 isolado; Torch 2.6.0 + torchvision 0.21.0 | Corrige `torchvision::nms`; pesos regionais V3 |
| `kokoro` | T4 | Python 3.11 isolado; Kokoro/Misaki 0.9.4 | Corrige erro de resolução em Python 3.13 |
| `higgs3` | **G4** | Python 3.12; vLLM + vLLM-Omni 0.32.0, CUDA 13 | Experimental e pesado; não usa Transformers pipeline genérico |

Cada engine fica em um ambiente dedicado para evitar conflitos de Torch/Transformers. O gerenciador `uv` pode baixar o Python correspondente ao inicializar a sessão. Não há armazenamento persistente dos pesos no Drive. O `hf_xet` permanece disponível: `ai-colab hf status`, `ai-colab hf fast` ou `ai-colab hf normal`.

O **Higgs** exige uma GPU com memória muito maior que a dos outros modelos; vLLM-Omni tem duas etapas e pode demorar a carregar. A ferramenta rejeita `--gpu A100` para evitar consumir unidades na A100 comum de 40 GiB; use a G4 padrão. Sua licença é de pesquisa/não comercial, com exceção específica para criadores que atribuem crédito; não use em serviços ou produtos comerciais sem licença apropriada.

## Sequência de testes recomendada (economizando créditos)

### 1. Kokoro na T4 — teste mais barato

```bash
ai-colab start voice --voice-model kokoro --gpu T4
ai-colab test
ai-colab voice generate \
  --text "Olá! Esta é uma narração em português brasileiro." \
  --speaker pf_dora --output ~/Downloads/kokoro-teste.wav
ai-colab stop
```

Vozes disponíveis: `pf_dora`, `pm_alex`, `pm_santa`; controle `--speed 0.5` a `2.0`.

### 2. VoxCPM2 na L4

```bash
ai-colab start voice --voice-model voxcpm2 --gpu L4
ai-colab voice generate \
  --text "Olá! Estou testando a nova integração de voz." \
  --style "Voz feminina brasileira, natural, suave e clara" \
  --steps 10 --output ~/Downloads/voxcpm2-teste.wav
ai-colab stop
```

Para testar clonagem de sua própria voz ou de alguém que consentiu:

```bash
ai-colab voice clone --text "Este áudio foi gerado com autorização." \
  --reference ~/Downloads/voz-autorizada.wav --consent \
  --output ~/Downloads/voxcpm2-clone.wav
```

Use `clone` **antes de `stop`** na mesma sessão. Com `--reference-text` a API emprega o modo continuação com transcrição, se compatível.

### 3. Chatterbox brasileiro na L4

```bash
ai-colab start voice --voice-model chatterbox-br --gpu L4
ai-colab voice generate --text "Bem-vindo! Esta é uma nova experiência de voz." \
  --exaggeration 0.5 --temperature 0.8 \
  --output ~/Downloads/chatterbox-br.wav
ai-colab stop
```

O modelo baixa os pesos da base `ResembleAI/chatterbox` e o T3 PT-BR separado. Caso a geração sem voz de referência não tenha condicionais disponíveis, envie um trecho de áudio autorizado com `ai-colab voice clone --reference ... --consent`.

### 4. Higgs experimental na G4 — deixe para depois

```bash
ai-colab start voice --voice-model higgs3 --gpu G4 --wait 1800
ai-colab voice generate --text "Olá! Agora estamos testando o Higgs TTS 3." \
  --style calmo --output ~/Downloads/higgs3.wav
ai-colab stop
```

O Higgs agora inicia **vLLM-Omni** privado em `127.0.0.1:8002`, com uma API local intermediária na porta 8000 e túnel SSH autenticado. O backend pode exigir muitos GB e não foi validado na G4 Blackwell. Se falhar, veja os dois logs abaixo. Clonagem não está implementada no adaptador Higgs.

## Diagnóstico sem reiniciar GPU

```bash
ai-colab status --remote
ai-colab stats --usage
ai-colab logs --source bootstrap --lines 100  # download/pip/uv
ai-colab logs --source vllm --lines 100       # worker e falhas de inferência
ai-colab logs --source omni --lines 100       # apenas Higgs
ai-colab diagnostics
ai-colab wait --seconds 1800                 # só acompanha uma sessão existente
ai-colab reconnect                           # restaura apenas SSH
ai-colab stop                                # libera GPU
```

Se a etapa `colab exec` não responder, a ferramenta tenta detectar remotamente o PID/log do bootstrap. **Não crie imediatamente outra GPU.** Confira `ai-colab status --remote`. Quando decidir encerrar a tentativa, use `ai-colab stop` para liberar unidades.

Na geração, respostas HTTP 500 incluem **tipo e localização** (`arquivo:linha`) da falha, mas não incluem o texto nem a gravação enviada. Isso ajuda a diagnosticar um novo `TypeError` do VoxCPM2 sem registrar conteúdo privado.

## Programação — preservada

```bash
ai-colab start code --gpu G4 --context 262144
ai-colab test
claude-colab
ai-colab stop
```

O launcher `claude-colab` mantém `--dangerously-skip-permissions`, como solicitado anteriormente; permite operações locais sem confirmação. Para utilizar permissões normais execute `COLAB_AI_SKIP_PERMISSIONS=0 claude-colab`. O comando original `claude` permanece intocado.

## Segurança, privacidade e limitações

- Só uma engine por sessão Colab. Trocar exige `stop` explícito.
- Arquivos de áudio de referência devem pertencer ao usuário ou ter consentimento verificável; não utilize clonagem para simular terceiros sem autorização.
- Referências de voz não são registradas em logs de texto; são transmitidas pelo túnel SSH autenticado. Ainda assim, a execução ocorre numa VM em nuvem, sujeita aos termos de privacidade aplicáveis.
- Limite de entrada: 1.800 caracteres por requisição, 450 no Chatterbox. Arquivos WAV ficam no computador após a geração; arquivos intermediários temporários são apagados.
- A VM do Colab é temporária, consome unidades enquanto alocada e pode ser encerrada por inatividade. A ferramenta não burla esses limites.
- Instalar novas versões do vLLM-Omni e dependências em runtime temporário pode levar bastante tempo. Não foi verificada a compatibilidade da G4 com esse backend específico.

## Fontes oficiais usadas na correção

- VoxCPM2: https://github.com/OpenBMB/VoxCPM (Python 3.10–3.12; `optimize=False`)
- Kokoro: https://pypi.org/project/kokoro/0.9.4/ (Python <3.13)
- Chatterbox: https://github.com/resemble-ai/chatterbox/blob/master/pyproject.toml (Torch 2.6.0)
- Chatterbox PT-BR: https://huggingface.co/ResembleAI/Chatterbox-Multilingual-pt-br
- Higgs TTS 3: https://huggingface.co/bosonai/higgs-tts-3-4b
- vLLM-Omni instalação: https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/gpu/
- Google Colab CLI: https://github.com/googlecolab/google-colab-cli
