# Cool scripts and links

## Ez install Runpod with 2.5 clicks in jupiter
```
git clone https://github.com/comfyanonymous/ComfyUI
python -m venv /workspace/ComfyUI/venv
source /workspace/ComfyUI/venv/bin/activate
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r /workspace/ComfyUI/requirements.txt
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git /workspace/ComfyUI/custom_nodes/ComfyUI-Manager
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt
git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/requirements.txt
git clone https://github.com/evanspearman/ComfyMath.git /workspace/ComfyUI/custom_nodes/ComfyMath
pip install -r /workspace/ComfyUI/custom_nodes/ComfyMath/requirements.txt
git clone https://github.com/rgthree/rgthree-comfy.git /workspace/ComfyUI/custom_nodes/rgthree-comfy
pip install -r /workspace/ComfyUI/custom_nodes/rgthree-comfy/requirements.txt
git clone https://github.com/city96/ComfyUI-GGUF.git /workspace/ComfyUI/custom_nodes/ComfyUI-GGUF
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git /workspace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt
git clone https://github.com/kijai/ComfyUI-KJNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt
git clone https://github.com/pollockjj/ComfyUI-MultiGPU.git /workspace/ComfyUI/custom_nodes/ComfyUI-MultiGPU
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-MultiGPU/requirements.txt
git clone https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use/requirements.txt
git clone https://github.com/WASasquatch/was-node-suite-comfyui.git /workspace/ComfyUI/custom_nodes/ComfyUI-was-node-suite-comfyui
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-was-node-suite-comfyui/requirements.txt
git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /workspace/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts/requirements.txt
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux
pip install -r /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/requirements.txt
git clone https://github.com/chflame163/ComfyUI_LayerStyle.git /workspace/ComfyUI/custom_nodes/ComfyUI_LayerStyle
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI_LayerStyle/requirements.txt
git clone https://github.com/cubiq/ComfyUI_essentials.git /workspace/ComfyUI/custom_nodes/ComfyUI_essentials
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI_essentials/requirements.txt
git clone https://github.com/chrisgoringe/cg-use-everywhere.git /workspace/ComfyUI/custom_nodes/cg-use-everywhere
pip install -r /workspace/ComfyUI/custom_nodes/cg-use-everywhere/requirements.txt
git clone https://github.com/VraethrDalkr/ComfyUI-TripleKSampler.git /workspace/ComfyUI/custom_nodes/ComfyUI-TripleKSampler
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-TripleKSampler/requirements.txt
git clone https://github.com/DoctorDiffusion/ComfyUI-MediaMixer.git /workspace/ComfyUI/custom_nodes/ComfyUI-MediaMixer
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-MediaMixer/requirements.txt
git clone https://github.com/stduhpf/ComfyUI-WanMoeKSampler.git /workspace/ComfyUI/custom_nodes/ComfyUI-WanMoeKSampler
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI-WanMoeKSampler/requirements.txt
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git /workspace/ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale/requirements.txt
git clone https://github.com/filliptm/ComfyUI_Fill-Nodes.git /workspace/ComfyUI/custom_nodes/ComfyUI_Fill-Nodes
pip install -r /workspace/ComfyUI/custom_nodes/ComfyUI_Fill-Nodes/requirements.txt
cd ..
```
New terminal and Run (check port)
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
- Open terminal in folder with file and paste
```
runpodctl send <filename.blablabla>
or
runpodctl receive <link>
```
- thx https://youtu.be/D2uQufx3W94?si=ahwy1MprdZvauTfb&t=407 for guide

### Models
# Wan 2.2 i2v + vace = 74gb
```
cd /workspace/ComfyUI/models
# diffusion_models
# i2v fp8 29gb
mkdir -p /workspace/ComfyUI/models/diffusion_models/wan
wget --content-disposition -P /workspace/ComfyUI/models/diffusion_models/wan https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
wget --content-disposition -P /workspace/ComfyUI/models/diffusion_models/wan https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
# Vace 35 gb
wget --content-disposition -P /workspace/ComfyUI/models/diffusion_models/wan https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors
wget --content-disposition -P /workspace/ComfyUI/models/diffusion_models/wan https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors
# Lightx2v lora 1gb
mkdir -p /workspace/ComfyUI/models/loras/wan/quality
mkdir -p /workspace/ComfyUI/models/loras/wan/quality/22
mkdir -p /workspace/ComfyUI/models/loras/wan/quality/22/1022
wget --content-disposition -P /workspace/ComfyUI/models/loras/wan/quality/22/1022 https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors
wget --content-disposition -P /workspace/ComfyUI/models/loras/wan/quality/22/1022 https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors
# clip fp8 7gb
wget --content-disposition -P /workspace/ComfyUI/models/clip https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
wget --content-disposition -P /workspace/ComfyUI/models/clip_vision https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors
# vae 2 gb
wget --content-disposition -P /workspace/ComfyUI/models/vae https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors
wget --content-disposition -P /workspace/ComfyUI/models/vae https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors
```
# Flux 18 gb
```
mkdir -p /workspace/ComfyUI/models/diffusion_models/flux
wget --content-disposition -P /workspace/ComfyUI/models/diffusion_models/flux https://huggingface.co/Comfy-Org/FLUX.1-Krea-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-krea-dev_fp8_scaled.safetensors
wget --content-disposition -P /workspace/ComfyUI/models/clip https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
wget --content-disposition -P /workspace/ComfyUI/models/clip https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors
wget --content-disposition -P /workspace/ComfyUI/models/vae https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors
```
# Qwen-image 20 + edit 2509 20 + clip vae 17 + lora ~4 = 61
```
mkdir -p /workspace/ComfyUI/models/diffusion_models/qwen
wget --content-disposition -P /workspace/ComfyUI/models/diffusion_models/qwen https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors
wget --content-disposition -P /workspace/ComfyUI/models/diffusion_models/qwen https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors
#stuff
wget --content-disposition -P /workspace/ComfyUI/models/clip https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors
wget --content-disposition -P /workspace/ComfyUI/models/vae https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors
# lora
wget --content-disposition -P /workspace/ComfyUI/models/loras/qwen/quality https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32.safetensors
wget --content-disposition -P /workspace/ComfyUI/models/loras/qwen/quality https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V2.0.safetensors
wget --content-disposition -P /workspace/ComfyUI/models/loras/qwen/edit/ https://huggingface.co/2600th/Qwen-Edit-2509-Multiple-angles-LORA/blob/main/Qwen-Edit-2509-Multiple-angles.safetensors
mv "/workspace/ComfyUI/models/loras/qwen/edit/镜头转换.safetensors" "/workspace/ComfyUI/models/loras/qwen/edit/qwen_edit_multi_angles.safetensors"
wget --content-disposition -P /workspace/ComfyUI/models/loras/qwen/edit https://huggingface.co/dx8152/Qwen-Image-Edit-2509-Fusion/resolve/main/溶图.safetensors
mv "/workspace/ComfyUI/models/loras/qwen/edit/溶图.safetensors" "/workspace/ComfyUI/models/loras/qwen/edit/qwen_edit_fusion.safetensors"
wget --content-disposition -P /workspace/ComfyUI/models/loras/qwen/edit https://huggingface.co/lovis93/next-scene-qwen-image-lora-2509/resolve/main/next-scene_lora-v2-3000.safetensors
```
# AnimateDiff pack for vid2vid SD 1.5: ControlNet, AnimateDiff, IpAdapter, vae, upscale
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

Optimization
Sage Attention guide https://civitai.com/articles/12848
```
моделей слишком дохуя пока хз
```
