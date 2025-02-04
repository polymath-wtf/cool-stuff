# Cool scripts and links

## Ez install Runpod with 2.5 clicks

- Ez install Runpod ComfyUI
```
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
python -m venv venv
cd venv
source bin/activate
pip3 install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
cd ..
pip install -r requirements.txt
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
cd ..
```
- Models for vid2vid SD 1.5: ControlNet, AnimateDiff, IpAdapter, vae, upscale
```
cd /workspace/ComfyUI/models
cd checkpoints
wget --content-disposition https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1.safetensors
wget --content-disposition https://civitai.com/api/download/models/304415
cd ..
mkdir animatediff_models
cd animatediff_models
wget --content-disposition https://huggingface.co/guoyww/animatediff/resolve/cd71ae134a27ec6008b968d6419952b0c0494cf2/v3_sd15_mm.ckpt
wget --content-disposition https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/main/animatediff_lightning_8step_comfyui.safetensors
wget --content-disposition https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt
cd ..
cd clip_vision
wget --content-disposition https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors
mv model.safetensors CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
wget --content-disposition https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors
mv model.safetensors CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors
cd ..
cd controlnet
wget --content-disposition https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth
wget --content-disposition https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth
wget --content-disposition https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth
wget --content-disposition https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth
wget --content-disposition https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth
wget --content-disposition https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/resolve/main/v2/control_v1p_sd15_qrcode_monster_v2.safetensors
cd ..
mkdir ipadapter
cd ipadapter
wget --content-disposition https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light_v11.bin
wget --content-disposition https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors
cd ..
cd loras
wget --content-disposition https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors
cd ..
cd vae
wget --content-disposition https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors
cd ..
cd upscale_models
wget --content-disposition https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth
wget --content-disposition https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/8x_NMKD-Superscale_150000_G.pth
cd ..
```
- New terminal and Run (check port)
```
apt update
apt install psmisc
fuser -k 8188/tcp
cd /workspace/ComfyUI/venv
source bin/activate
cd /workspace/ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```
- RunPod <=> Google Disc download notebook https://colab.research.google.com/drive/13UMW1lbuxVRZOzZeHQfQPmZ05KHWNw7f
- Paste in new terminal
```
wget --quiet --show-progress https://github.com/Run-Pod/runpodctl/releases/download/v1.6.1/runpodctl-linux-amd -O runpodctl && chmod +x runpodctl
```
Open terminal in folder with file and paste
```
runpodctl send <filename.blablabla>
or
runpodctl receive <link>
```
- thx https://youtu.be/D2uQufx3W94?si=ahwy1MprdZvauTfb&t=407 for guide
