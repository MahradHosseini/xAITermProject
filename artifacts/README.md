# Local Artifacts

Large local assets are stored here for offline work and are intentionally not tracked by Git.

- `data/`: datasets such as FloodNet.
- `models/`: model checkpoints copied from framework caches.
- `downloads/`: raw archives before extraction.

## Cached Assets

FloodNet:

- Archive: `artifacts/downloads/FloodNet.zip`
- Extracted dataset: `artifacts/data/FloodNet-Supervised_v1.0/`
- Extracted color masks: `artifacts/data/ColorMasks-FloodNetv1.0/`

TorchVision checkpoints:

- `artifacts/models/resnet18-f37072fd.pth`
- `artifacts/models/resnet50-11ad3fa6.pth`

Official Facebook DeiT checkpoints for PyTorch/timm:

- `artifacts/models/deit_tiny_patch16_224-a1311bcf.pth`
- `artifacts/models/deit_small_patch16_224-cd65a155.pth`

## Loading DeiT Offline

```python
import torch
import timm

model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=1000)
checkpoint = torch.load(
    "artifacts/models/deit_tiny_patch16_224-a1311bcf.pth",
    map_location="cpu",
)
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()
```

For DeiT-S, use `deit_small_patch16_224` and `artifacts/models/deit_small_patch16_224-cd65a155.pth`.
