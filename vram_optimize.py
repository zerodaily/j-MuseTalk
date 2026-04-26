import torch
import gc

def optimize_vram():
    """VRAM optimization for RTX 2060 12GB"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f'[VRAM] Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
        print(f'[VRAM] Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB')
    gc.collect()

def get_safe_batch_size():
    """Get batch size safe for 12GB VRAM"""
    return 4

print('[VRAM] Optimization module loaded')
