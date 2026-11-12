# Pesquisa técnica — geração de vídeo no Google Colab Pro (novembro de 2026)

**Estágio na v0.2.9:** catálogo de candidatos e desenho do próximo módulo. **Não há backend de geração de vídeo instalado, nem comandos de inferência funcionando nesta versão.** `ai-colab start video` continua bloqueado sem provisionar GPU. Consultar `ai-colab video models` (ou `--json`) não usa GPU.

## Requisitos para a Colab AI CLI

- Usar a infraestrutura que já funciona: sessão Colab, túnel **SSH privado**, API autenticada e cliente Linux sem obrigatoriedade de armazenamento no Google Drive.
- Suportar **texto → vídeo (T2V)** e **imagem → vídeo (I2V)** como objetivos iniciais. Áudio sincronizado é opcional e deve ser distinguido de vídeo silencioso.
- A saída deverá ser um **MP4 local** no Linux, preferencialmente com imagem de referência recebida por upload privado; não transmitir arquivo por API pública.
- Priorizar download rápido dos pesos via Hugging Face/hf_xet e limpar resultados grandes do disco efêmero depois do download.
- Início, consulta de progresso, cancelamento e download devem ser operações distintas: vídeo pode demorar muito mais que a janela normal de uma requisição HTTP.
- A interface de vídeo não deve prometer disponibilidade das GPUs. A `google-colab-cli` pode falhar na alocação mesmo quando o Colab no navegador funciona.

## Candidatos prioritários

| Identificador | Modelo | Recursos | Requisito publicado | GPU inicial proposta | Licença / atenção |
|---|---|---|---|---|---|
| `hunyuan15` | HunyuanVideo-1.5, 8.3B | T2V + I2V; versões 480p/720p, I2V 480p destilado 8–12 passos | 14 GB VRAM **com offload** | L4; T4 experimental (limite estreito de VRAM); G4 | Tencent Hunyuan Community |
| `wan22-5b` | Wan2.2 TI2V-5B | T2V + I2V, vídeo 720p a 24 fps | 24 GB com `--offload_model True --convert_model_dtype --t5_cpu` | G4 para desempenho; L4 com offload | Apache-2.0 |
| `wan22-14b` | Wan2.2 A14B (T2V/I2V, checkpoints distintos) | T2V + I2V 480p/720p, MoE maior | **≥80 GB VRAM** no fluxo oficial de GPU única | G4, se suficiente, com cuidado com disco e RAM | Apache-2.0 |
| `ltx25` | LTX-2.5 Fast | T2V, I2V e **áudio-vídeo sincronizados** | aplicação Desktop Linux recomenda ≥24 GB VRAM, ≥32 GB RAM e ≥100 GB disco; exemplo oficial de pesos bf16 totaliza **~66 GiB**; offload/FP8 disponíveis | G4; L4 só após estudo de compatibilidade e memória | LTX-2.x Community; pesos gated, aceitar termos no HF |

**Importante:** VRAM mínima oficial não garante viabilidade em qualquer instância do Colab: RAM do host, espaço em disco, driver CUDA, bibliotecas de atenção, tamanho do clipe e offload interferem no resultado. GPU indicada é recomendação para _o primeiro teste_, não uma validação neste projeto. As duas famílias Wan e Hunyuan acima produzem vídeo silencioso no fluxo básico; LTX 2.5 pode produzir áudio alinhado ao vídeo.

### 1. HunyuanVideo-1.5 — melhor candidato para primeiro protótipo econômico

- Pesos 8,3B e instrução oficial de 14 GB VRAM com offload. Isso o torna o candidato mais realista para T4 de 16 GB, mas é uma margem pequena; a **L4 (24 GB) é preferível** para depuração.
- Presets oficiais de 480p I2V destilado fazem 8/12 steps; testar primeiro com imagem estática de referência e clipe curto.
- A licença do modelo não é a mesma licença Apache do código; verificar condições para usos comerciais.
- Requer ainda implementação de bootstrap, instalação de dependências e engine; **não** está funcional na CLI.

Fontes: [repositório oficial](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5); [modelo no Hugging Face](https://huggingface.co/tencent/HunyuanVideo-1.5).

### 2. Wan2.2 TI2V-5B — melhor primeiro modelo na G4

- Modelo unificado texto/imagem → vídeo, suporte oficial a 720p/24fps.
- O fabricante exige ao menos 24 GB para single-GPU com offload. Com ≥80 GB é possível remover as flags de offload e acelerar o processamento; isso favorece a G4, se alocada.
- Sem geração nativa de áudio nessa pipeline; MP4 com frames apenas.
- Para L4, planejar memória do host e gerenciamento de pesos.

Fontes: [Wan2.2 oficial](https://github.com/Wan-Video/Wan2.2), [TI2V 5B no Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

### 3. Wan2.2 A14B — opção avançada

- Fluxo oficial para ~80 GB de VRAM; mais pesado para provisionamento e download.
- Não é recomendação para primeiro teste de integração, mesmo na G4, porque hardware, espaço e latência ainda não foram medidos.

Fonte: [Wan2.2 oficial](https://github.com/Wan-Video/Wan2.2).

### 4. LTX-2.5 — áudio + vídeo, com ressalvas

- É a geração atual recomendada pela própria Lightricks em 2026, substituindo o LTX-Video legado.
- Pipeline `distilled` da Lightricks: transformer 22B, Gemma4 12B text encoder, VAE de vídeo e áudio, upscaler; download de exemplo **~66 GiB**. `--quantization fp8-cast --offload cpu` ajuda em GPUs limitadas, mas não é garantia de caber com RAM/disco do Colab.
- Requer aceitar termos dos pesos e autenticação Hugging Face em repositório gated; nunca solicitar/armazenar chave do usuário sem uma necessidade clara.
- Licença Community LTX-2.x: uso comercial sujeito às condições específicas, inclusive limite de receita/licença empresarial.

Fontes: [código e instruções oficiais](https://github.com/Lightricks/LTX-2); [model card e licença](https://huggingface.co/Lightricks/LTX-2.5); [requisitos Desktop](https://github.com/Lightricks/LTX-Desktop).

## Proposta de sequência de implementação

1. **MVP L4 — HunyuanVideo-1.5 480p:** endpoint privado com job assíncrono, geração de 1 clipe curto, download MP4 e logs; modalidade `text-to-video` primeiro, depois `image-to-video` com upload local.
2. **Perfil G4 — Wan2.2 TI2V-5B:** mesmo contrato API, presets de 720p e I2V; teste com e sem offload apenas se memória disponível.
3. **Perfil avançado — LTX-2.5 Fast:** vídeo com áudio, somente depois de avaliar políticas de licença, HF token, disco disponível, requisitos de CUDA e custo de créditos.
4. **Otimização:** histórico local de jobs, limpeza de arquivos, cancelamento, seed reprodutível, fps/duração/predefinições, indicador de progresso e retomada após perder túnel.

### Comandos da v0.3.0 (implementados, aguardando teste de GPU real)

```bash
ai-colab video models
ai-colab start video --video-model hunyuan15 --gpu L4
ai-colab video generate --prompt "A ship sailing calmly" --frames 49 --output barco.mp4
ai-colab video generate --image foto.png --prompt "Slow camera push-in" --output animacao.mp4
ai-colab video jobs
ai-colab video status JOB_ID
ai-colab video fetch JOB_ID --output resultado.mp4
ai-colab video cancel JOB_ID
```

**Observação pós-implementação:** o backend Hunyuan da v0.3.0 usa Diffusers e baixa dois checkpoints separados (T2V e I2V); o checkpoint T2V 480p ocupa aproximadamente 53 GiB. Offload habilitado em L4 ainda pode exigir RAM/disco altos. O Wan usa o gerador oficial por subprocesso e offload conforme VRAM. A execução em GPUs reais é pendente. A alocação da GPU continua dependendo da Colab CLI (que pode retornar HTTP 503 mesmo quando o navegador funciona).

## Critérios de aprovação antes de declarar funcional

- No boot, detecta CUDA, VRAM real e disco livre antes de baixar checkpoints extensos.
- Gera MP4 válido (`ffprobe` verifica codec, fps, duração e dimensões) em GPU real.
- Testa I2V visualmente usando referência controlada, sem inventar vídeos bem-sucedidos.
- Valida cancelamento, perda/reconexão do túnel, timeout e limpeza remota.
- Preserva as modalidades Code, Voice e Transcribe (regressão do cliente), com testes de integração na GPU relevante.
