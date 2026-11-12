"""Gera MP4 por subprocesso, usando as rotas oficiais Hunyuan/Diffusers e Wan."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch

HUNYUAN_T2V = 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v'
HUNYUAN_I2V = 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled'
WAN_REPO = 'Wan-AI/Wan2.2-TI2V-5B'
WAN_GIT = 'https://github.com/Wan-Video/Wan2.2.git'



def offload_choice(setting: str, total_gib: float) -> bool:
    if setting == 'on':
        return True
    if setting == 'off':
        if total_gib < 80:
            raise RuntimeError('Sem offload requer pelo menos 80 GiB de VRAM para esta configuração; use --offload auto')
        return False
    return total_gib < 80


def image_prepare(path: Path, width: int, height: int, fit_mode: str, output: Path):
    from PIL import Image, ImageOps
    with Image.open(path) as opened:
        im = ImageOps.exif_transpose(opened).convert('RGB')
        if fit_mode == 'cover':
            im = ImageOps.fit(im, (width, height), method=Image.Resampling.LANCZOS)
        else:
            im = ImageOps.pad(im, (width, height), method=Image.Resampling.LANCZOS, color=(0,0,0))
        im.save(output, format='PNG')
    print(f'Imagem ajustada para {width}x{height} ({fit_mode}); original preservada.', flush=True)
    return output

def build_wan_command(python, repo_dir, weights, args):
    size = f'{args.width}*{args.height}'
    cmd = [str(python), str(repo_dir / 'generate.py'), '--task', 'ti2v-5B',
           '--size', size, '--ckpt_dir', str(weights),
           '--prompt', args.prompt, '--frame_num', str(args.frames),
           '--sample_steps', str(args.steps), '--base_seed', str(args.seed),
           '--save_file', str(args.output)]
    # CPU offload para placas de 24GB, desligado em >=80GB
    if offload_choice(args.offload, torch.cuda.get_device_properties(0).total_memory / 1024**3):
        cmd += ['--offload_model', 'True', '--convert_model_dtype', '--t5_cpu']
    else:
        cmd += ['--offload_model', 'False']
    if args.image:
        cmd += ['--image', str(args.image)]
    return cmd


def generate_hunyuan(args):
    from diffusers import HunyuanVideo15Pipeline, HunyuanVideo15ImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image
    repo = HUNYUAN_I2V if args.image else HUNYUAN_T2V
    print('Modelo Diffusers:', repo, flush=True)
    cls = HunyuanVideo15ImageToVideoPipeline if args.image else HunyuanVideo15Pipeline
    pipe = cls.from_pretrained(repo, torch_dtype=torch.bfloat16)
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    use_offload = offload_choice(args.offload, memory_gib)
    print(f'Estratégia Hunyuan: {"model_cpu_offload" if use_offload else "GPU direta"} | VRAM {memory_gib:.1f} GiB', flush=True)
    if use_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to('cuda')
    pipe.vae.enable_tiling()
    gen = torch.Generator(device='cuda').manual_seed(args.seed)
    kwargs = {'prompt': args.prompt, 'generator': gen,
              'num_frames': args.frames, 'num_inference_steps': args.steps}
    if args.image:
        kwargs['image'] = load_image(str(args.image)).convert('RGB')
    else:
        kwargs['width'], kwargs['height'] = args.width, args.height
    print(f'Iniciando inferência Hunyuan ({args.width}x{args.height}, 480p)', flush=True)
    frames = pipe(**kwargs).frames[0]
    export_to_video(frames, str(args.output), fps=24)


def generate_wan(args):
    from huggingface_hub import snapshot_download
    root = Path('/content/colab-ai-video-engine')
    repo_dir = root / 'Wan2.2'
    if not (repo_dir / 'generate.py').is_file():
        root.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(['git', 'clone', '--depth', '1', WAN_GIT, str(repo_dir)])
    weights = root / 'weights' / 'wan22-5b'
    weights.mkdir(parents=True, exist_ok=True)
    print('Verificando/baixando pesos Wan TI2V-5B em armazenamento efêmero...', flush=True)
    snapshot_download(WAN_REPO, local_dir=str(weights))
    g = torch.cuda.get_device_properties(0)
    print(f'Estratégia Wan: {"offload CPU" if offload_choice(args.offload, g.total_memory/1024**3) else "GPU direta"} | VRAM {g.total_memory/1024**3:.1f} GiB', flush=True)
    cmd = build_wan_command(sys.executable, repo_dir, weights, args)
    print('Iniciando inferência Wan 2.2 TI2V-5B (720p)', flush=True)
    subprocess.check_call(cmd, cwd=str(repo_dir))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['hunyuan15', 'wan22-5b'], required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--image', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--aspect', choices=['16:9', '9:16'], required=True)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--height', type=int, required=True)
    p.add_argument('--offload', choices=['auto','on','off'], default='auto')
    p.add_argument('--image-fit', choices=['cover','contain'], default='cover')
    p.add_argument('--seed', type=int, required=True)
    args = p.parse_args()
    if not 17 <= args.frames <= 121 or args.frames % 4 != 1:
        p.error('frames requer 4n+1, de 17 a 121')
    if not 1 <= args.steps <= 60:
        p.error('steps de 1 a 60')
    if not 0 <= args.seed < 2**32:
        p.error('seed fora do intervalo')
    if args.image and not args.image.is_file():
        p.error('imagem não encontrada')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA do PyTorch indisponível no worker Video; veja ai-colab gpu status --deep')
    device = torch.cuda.get_device_properties(0)
    print(f'GPU efetiva: {device.name} | VRAM {device.total_memory/1024**3:.1f} GiB '
          f'| PyTorch {torch.__version__} | CUDA {torch.version.cuda}', flush=True)
    use_offload = offload_choice(args.offload, device.total_memory/1024**3)
    print(f'Offload solicitado={args.offload}; escolhido={"CPU" if use_offload else "GPU"}', flush=True)
    if args.image:
        corrected = args.output.with_suffix('.reference.png')
        args.image = image_prepare(args.image, args.width, args.height, args.image_fit, corrected)
    if args.variant == 'hunyuan15':
        generate_hunyuan(args)
    else:
        generate_wan(args)
    if not args.output.is_file() or args.output.stat().st_size < 128:
        raise RuntimeError('Geração não produziu MP4 válido')
    with args.output.open('rb') as video:
        header = video.read(32)
    if b'ftyp' not in header[4:16]:
        raise RuntimeError('Arquivo gerado não parece ser MP4')
    print(f'Vídeo concluído: {args.output} ({args.output.stat().st_size} bytes)', flush=True)


if __name__ == '__main__':
    main()
