# Cool scripts and links

## Ez install Runpod with 2.5 clicks

- Ez install Runpod ComfyUI
```
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
python -m venv venv
cd venv
source bin/activate
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
cd ..
pip install -r requirements.txt
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git
git clone https://github.com/kijai/ComfyUI-KJNodes.git
git clone https://github.com/jags111/efficiency-nodes-comfyui.git
git clone https://github.com/evanspearman/ComfyMath.git
git clone https://github.com/crystian/comfyui-crystools.git
git clone https://github.com/rgthree/rgthree-comfy.git
git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack.git
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
git clone https://github.com/calcuis/gguf.git
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
git clone https://github.com/pollockjj/ComfyUI-MultiGPU.git
git clone https://github.com/Flow-two/ComfyUI-WanStartEndFramesNative.git
git clone https://github.com/orssorbit/ComfyUI-wanBlockswap.git

```
Wan insta build
```
cd /workspace/ComfyUI/models
cd diffusion_models
wget --content-disposition https://huggingface.co/QuantStack/Wan2.1_I2V_14B_FusionX-GGUF/resolve/main/Wan2.1_I2V_14B_FusionX-Q4_0.gguf
wget --content-disposition https://huggingface.co/QuantStack/Wan2.1_T2V_14B_FusionX_VACE-GGUF/resolve/main/Wan2.1_T2V_14B_FusionX_VACE-Q4_0.gguf
wget --content-disposition https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_0.gguf
wget --content-disposition https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_0.gguf
cd ..
cd vae
wget --content-disposition https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/main/VAE/Wan2.1_VAE.safetensors
wget --content-disposition https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/VAE/Wan2.2_VAE.safetensors
cd ..
cd lora
mkdir wan
cd wan
mkdir quality
cd quality
wget --content-disposition https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/resolve/main/loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors
wget --content-disposition  https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v/resolve/main/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors
cd ..
cd ..
cd ..
cd clip
wget --content-disposition  https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q4_K_M.gguf
wget --content-disposition  https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
cd ..
cd clip_vision
wget --content-disposition  https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors
cd ..

```
## AnimateDiff pack for vid2vid SD 1.5: ControlNet, AnimateDiff, IpAdapter, vae, upscale
```
cd /workspace/ComfyUI/models
cd checkpoints
wget --content-disposition https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1.safetensors
wget --content-disposition https://civitai.com/api/download/models/304415
wget --content-disposition https://civitai.com/api/download/models/128713
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

## Wan 2.1 pack
Main
480 gguf https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/tree/main  
720 gguf https://huggingface.co/city96/Wan2.1-I2V-14B-720P-gguf/tree/main

Clip

Vae

Lora
Reward (hd boost) https://huggingface.co/alibaba-pai/Wan2.1-Fun-Reward-LoRAs/tree/main


Controlnet
Control 1.3B https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-Control/tree/main  
Inpaint 1.3B https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP  
  
Control 14B https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-Control  
Control 14B GGUF https://huggingface.co/city96/Wan2.1-Fun-14B-Control-gguf/tree/main  
Inpaint 14B https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-InP  
Inpaint 14B GGUF https://huggingface.co/city96/Wan2.1-Fun-14B-InP-gguf/tree/main  


Optimization
Sage Attention guide https://civitai.com/articles/12848
```
моделей слишком дохуя пока хз
```
