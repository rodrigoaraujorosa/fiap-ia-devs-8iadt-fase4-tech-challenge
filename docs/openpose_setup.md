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

## 2. Baixar os modelos

Dentro da pasta extraída:

```bat
cd tools\openpose\models
getModels.bat
```

> ⚠️ Os links da CMU (`posefs1.perception.cs.cmu.edu`) **caem com frequência**. Se
> falhar, baixe o `pose_iter_584000.caffemodel` (BODY_25) de um mirror e coloque em
> `models\pose\body_25\`.

## 3. Testar

```bat
cd tools\openpose
bin\OpenPoseDemo.exe --video examples\media\video.avi --write_json out_json --display 0 --render_pose 0
```

Se gerar JSONs em `out_json\`, está funcionando.

## 4. GPU fraca (pouca VRAM)

A resolução padrão (`656x368`) pode estourar a memória de GPUs com ~2 GB (ex.: MX330).
Reduza com `--net_resolution 320x176` (o nosso `run_openpose.py` já usa esse valor por
padrão). Se ainda faltar memória ou o CUDA reclamar, use o build **CPU** (lento, mas
funciona) ou o **Colab** (`notebooks/openpose_kimore_colab.ipynb`).

## 5. Rodar pelo nosso pipeline

```bash
python -m src.video.cli --video data/video/kimore_ex1.mp4 --openpose-root tools/openpose
```
