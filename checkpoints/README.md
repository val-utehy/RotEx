# Pretrained Checkpoints

Weights: [Hugging Face](https://huggingface.co/val-utehy/RotEx/tree/main).

| File | Download | Save to |
| --- | --- | --- |
| Full model, best | [best.pt](https://huggingface.co/val-utehy/RotEx/resolve/main/checkpoints/full/best.pt?download=true) | `checkpoints/full/best.pt` |
| Full model, final | [last.pt](https://huggingface.co/val-utehy/RotEx/resolve/main/checkpoints/full/last.pt?download=true) | `checkpoints/full/last.pt` |

Run from the project root:

```shell
hf download val-utehy/RotEx checkpoints/full/best.pt --local-dir .
hf download black-forest-labs/FLUX.1-dev ae.safetensors --local-dir checkpoints
```

One-step weights: [distill_checkpoints](../distill_checkpoints/README.md).

Usage: [main README](../README.md).
