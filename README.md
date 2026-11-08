# Colab AI CLI v0.2.0 — Code + Voice (5 famílias de TTS)

Gerenciador Linux para GPUs temporárias do Google Colab Pro por meio da CLI oficial. `ai-colab` controla a VM; `claude-colab` continua acessando o Qwen de programação pelo túnel SSH privado. **A versão 0.2.0 preserva a v0.1.9 e acrescenta quatro perfis de voz**: VoxCPM2, Chatterbox PT-BR, Kokoro e Higgs TTS 3.

> **Validação:** 61 testes de integração simulada e regressão passaram no Linux, sem criar GPUs. **Ainda não há testes de inferência real desses quatro backends em Colab.** A primeira execução de cada um pode revelar incompatibilidades de dependências, checkpoints ou VRAM. O Higgs é particularmente experimental.

## Instalação

```bash
# Primeiro uso: instale e autentique a CLI oficial do Colab
uv tool install google-colab-cli
colab --auth=oauth2 sessions

# Na pasta extraída deste ZIP
bash install.sh
ai-colab --version         # 0.2.0
ai-colab models
ai-colab ssh-check         # não inicia GPU
```

Não há cache persistente no Google Drive; `hf_xet` permanece ativado no modo rápido por padrão. `ai-colab hf normal` permite desativar o alto paralelismo nos próximos starts.

## Escolha de modelos de voz

| `--voice-model` | Modelo HF | GPU padrão | Geração | Design | Clonagem autorizada |
|---|---|---|---|---|---|
| `custom` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | L4 | Sim | — | — |
| `design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | L4 | — | Sim | — |
| `clone` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | L4 | — | — | Sim |
| `voxcpm2` | `openbmb/VoxCPM2` | L4 | Sim | Sim | Sim |
| `chatterbox-br` | `ResembleAI/Chatterbox-Multilingual-pt-br` | L4 | Sim | — | Sim |
| `kokoro` | `hexgrad/Kokoro-82M` | **T4** | Sim | — | — |
| `higgs3` | `bosonai/higgs-tts-3-4b` | **A100** | Sim (experimental) | — | — |

GPU é recomendação inicial, não teste de VRAM realizado. A GPU pode ser alterada com `--gpu L4` / `--gpu T4` / `--gpu G4`. A100 para Higgs foi escolhida como opção conservadora; L4 é um experimento possível, sem garantia.

**Somente um modelo por sessão.** Se a VM estiver com Qwen Code ou outra variante Voice ativa, execute `ai-colab stop` antes de trocar. Isso evita descarregar/duplicar modelos sem controle.

### 1) VoxCPM2 — primeiro teste recomendado

```bash
ai-colab start voice --voice-model voxcpm2 --gpu L4
ai-colab test
ai-colab voice generate --text "Olá! Esta narração foi criada com o VoxCPM dois." \
  --style "Voz brasileira suave, clara e natural" \
  --steps 10 --output ~/Downloads/voxcpm2.wav
ai-colab voice design --text "Bem-vindo ao nosso canal." \
  --description "Voz feminina brasileira, serena, dicção clara" \
  --output ~/Downloads/voxcpm2-design.wav
```

Para clonar uma voz **com autorização**:

```bash
ai-colab voice clone --text "Esta é uma gravação gerada com autorização." \
  --reference ~/Downloads/minha-voz.wav --consent \
  --output ~/Downloads/voxcpm2-clone.wav
```

No VoxCPM2, `--reference-text` é opcional; se fornecido, é empregado no modo de continuação da referência, como previsto no projeto oficial. Use uma gravação sua ou devidamente autorizada.

### 2) Chatterbox PT-BR — ajuste regional

```bash
ai-colab stop
ai-colab start voice --voice-model chatterbox-br --gpu L4
ai-colab voice generate --text "Olá, tudo bem? Agora estamos testando a voz brasileira." \
  --exaggeration 0.5 --temperature 0.8 --cfg-weight 0.5 \
  --output ~/Downloads/chatterbox-br.wav
```

O checkpoint `ResembleAI/Chatterbox-Multilingual-pt-br` **não é autossuficiente**: o worker carrega os componentes compartilhados da versão multilíngue V3 (`ResembleAI/chatterbox`) e aplica estritamente o T3 regional `t3_pt_br.safetensors`. O código não ignora silenciosamente uma incompatibilidade. Ainda não foi comprovado que esse conjunto gera áudio no runtime real.

Caso o modelo reclame de ausência de condicionais padrão, tente uma referência autorizada:

```bash
ai-colab voice clone --text "A voz brasileira também pode ser personalizada." \
  --reference ~/Downloads/minha-voz.wav --consent \
  --output ~/Downloads/chatterbox-ref.wav
```

Em caso de erro consulte `ai-colab logs --source vllm --lines 100` e interrompa a GPU se desistir.

### 3) Kokoro — rápido e econômico

```bash
ai-colab stop
ai-colab start voice --voice-model kokoro --gpu T4
ai-colab voice generate --text "Bom dia, pessoal!" \
  --speaker pf_dora --speed 1.0 --output ~/Downloads/kokoro-dora.wav
ai-colab voice generate --text "Esse teste utiliza outra voz." \
  --speaker pm_alex --output ~/Downloads/kokoro-alex.wav
```

Vozes brasileiras: `pf_dora`, `pm_alex`, `pm_santa`. A engine usa `KPipeline(lang_code='p')` e o `espeak-ng` instalado na VM. Não oferece clonagem ou voice design nesta implementação.

### 4) Higgs TTS 3 — experimental e com licença restritiva

```bash
ai-colab stop
ai-colab start voice --voice-model higgs3 --gpu A100
ai-colab voice generate --text "Olá! Esse é o teste do Higgs TTS três." \
  --style calmo --output ~/Downloads/higgs3.wav
```

O adaptador inicial usa `transformers.pipeline('text-to-speech', model=...)`, conforme cartão do modelo. **Ainda não há evidência de funcionamento nessa configuração; o servidor pode falhar ao carregar.** A implementação recomendada para produção do fabricante usa SGLang-Omni ou vLLM-Omni, que exigem uma integração própria. Nesta versão, Higgs suporta somente texto→voz e três estilos básicos (`calmo`, `alegre`, `sussurrar`). Não implementei clonagem nem um serviço comercial para ele.

**Licença:** Boson Research and Non-Commercial. Há uma exceção de uso para criadores de conteúdo com atribuição, mas hospedar serviço/API comercial, incorporar em produto ou revender exige licença separada. Consulte o documento original antes de usar: https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/LICENSE

## Qwen Voice — continua disponível

```bash
ai-colab stop
ai-colab start voice --voice-model custom --gpu L4
ai-colab voice generate --text "O Qwen continua disponível." \
  --speaker Ryan --output ~/Downloads/qwen.wav

ai-colab stop
ai-colab start voice --voice-model design --gpu L4
ai-colab voice design --text "Olá!" --description "Voz brasileira natural." \
  --output ~/Downloads/qwen-design.wav

ai-colab stop
ai-colab start voice --voice-model clone --gpu L4
ai-colab voice clone --text "Teste autorizado." --reference ~/Downloads/minha-voz.wav \
  --reference-text "Transcrição literal do áudio enviado." --consent \
  --output ~/Downloads/qwen-clone.wav
```

## Programação — mantida

```bash
ai-colab stop
ai-colab start code --gpu G4 --context 262144
ai-colab test
claude-colab
ai-colab stop
```

`claude-colab` preserva a flag `--dangerously-skip-permissions` solicitada anteriormente (cuidado com alterações locais sem confirmação). `COLAB_AI_SKIP_PERMISSIONS=0 claude-colab` desativa o bypass. O `claude` normal continua usando a configuração Anthropic original.

## Diagnóstico, monitoramento e custos

```bash
ai-colab status --remote
ai-colab logs --source bootstrap --lines 100   # instalação e download
ai-colab logs --source vllm --lines 100       # inferência, inclusive Voice
ai-colab wait --seconds 1800
ai-colab reconnect                            # apenas túnel SSH
ai-colab stats --usage
ai-colab monitor --interval 10
ai-colab stop                                 # libere a GPU
```

- Cada VM é temporária: a primeira execução pode baixar pacotes e pesos novamente.
- O modo `hf_xet` rápido ajuda em algumas redes, mas não garante downloads mais rápidos.
- A conexão privada usa SSH + chave local para o proxy remoto em `127.0.0.1`.
- Áudios de referência (até 4 MB, 0,5–60 s) não são gravados em logs; são mantidos apenas durante a chamada e enviados somente pelo túnel autenticado. Revise as obrigações de privacidade e consentimento antes de usar voz de terceiros.
- WAVs são gravados atomica e localmente; textos têm limite de 1.800 caracteres por solicitação; o Chatterbox limita 450.
- O sistema **não** cria GPUs novas automaticamente durante retentativas nem garante proteção contra inatividade do Colab.

## Referências verificadas

- VoxCPM2: https://github.com/OpenBMB/VoxCPM e https://huggingface.co/openbmb/VoxCPM2
- Chatterbox V3/regionais: https://github.com/resemble-ai/chatterbox e https://huggingface.co/ResembleAI/Chatterbox-Multilingual-pt-br
- Kokoro: https://github.com/hexgrad/kokoro e https://huggingface.co/hexgrad/Kokoro-82M
- Higgs TTS 3: https://huggingface.co/bosonai/higgs-tts-3-4b
- Hugging Face Hub: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
- Colab CLI: https://github.com/googlecolab/google-colab-cli
