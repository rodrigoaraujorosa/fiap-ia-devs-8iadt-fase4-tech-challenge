# ⚙️ Setup do OpenPose (Windows)

O OpenPose **não** se instala via `pip` — é um binário externo. Usamos a
**Portable Demo** (pré-compilada, sem precisar buildar em C++/CUDA).

## 1. Baixar

OpenPose v1.7.0 — [releases da CMU](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/tag/v1.7.0):

| Build | Arquivo | Quando usar |
|-------|---------|-------------|
| **GPU** | `openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended.zip` (438 MB) | tem GPU NVIDIA |
| CPU | `openpose-1.7.0-binaries-win64-cpu-python3.7-flir-3d.zip` (154 MB) | sem GPU / fallback |

Extraia em `tools/openpose/` (essa pasta é ignorada pelo git).

## 2. Baixar o modelo BODY_25

O script oficial é `models\getBaseModels.bat`, mas ele aponta para o servidor da CMU
(`posefs1.perception.cs.cmu.edu`) que está **fora do ar** (confirmado — DNS não resolve
mais). Para a análise postural só precisamos do modelo **BODY_25** (~100 MB); baixe de um
mirror direto para a pasta certa:

```bash
# mirror HuggingFace (commit fixado, ~100 MB) — testado e funcionando
curl -L -o tools/openpose/models/pose/body_25/pose_iter_584000.caffemodel \
  "https://huggingface.co/camenduru/openpose/resolve/f4a22b0e6fa2a4a2b1e2d50bd589e8bb11ebea7c/pose_iter_584000.caffemodel"
```

Os `.prototxt` já vêm no zip; só falta esse `.caffemodel`. (Face/hand são dispensáveis
para esta entrega.) Mirror alternativo: `http://vcl.snu.ac.kr/OpenPose/models/pose/body_25/`.

## 3. Testar

```bat
cd tools\openpose
bin\OpenPoseDemo.exe --video examples\media\video.avi --write_json out_json ^
  --model_pose BODY_25 --net_resolution 320x176 --display 0 --render_pose 0
```

Se gerar JSONs em `out_json\`, está funcionando. ✅ (Testado nesta máquina: GPU MX330
2 GB, ~1,2 s/frame com `320x176`, sem estouro de VRAM.)

## 4. GPU fraca (pouca VRAM)

A resolução padrão (`656x368`) estoura a memória de GPUs com ~2 GB (ex.: MX330). Reduza
com `--net_resolution 320x176` — o nosso `run_openpose.py` já usa esse valor por padrão e
foi **confirmado funcionando** na MX330. Se ainda faltar memória ou o CUDA reclamar, use o
build **CPU** (lento) ou o **Colab** (`notebooks/openpose_rehab24-6_colab.ipynb`).

> ⏱️ **Desempenho:** ~1,2 s/frame na MX330. Um vídeo do REHAB24-6 (ex.: `PM_008`, ~5200
> frames a 30 fps) levaria ~1h45 — por isso subamostramos (1 a cada 3 frames) para ~35 min.
> Para processar vários vídeos, o **Colab** (GPU T4) compensa muito.

## 5. Rodar pelo nosso pipeline

```bash
python -m src.video.cli --video data/video/rehab24-6/PM_008-Camera17-30fps.mp4 --openpose-root tools/openpose
```
