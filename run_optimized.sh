#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512

cd /vol2/1000/projects/MuseTalk

echo '[Launch] Starting MuseTalk with VRAM optimization...'
echo '[VRAM] Batch size limited to 4 for 12GB GPU'

python3 app.py --port 7865 --batch_size 4
