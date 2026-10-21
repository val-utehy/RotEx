# Multi-Step Weights

| File | Download |
| --- | --- |
| Best checkpoint | [best.pt](https://huggingface.co/val-utehy/RotEx/resolve/main/checkpoints/full/best.pt?download=true) |
| Final checkpoint | [last.pt](https://huggingface.co/val-utehy/RotEx/resolve/main/checkpoints/full/last.pt?download=true) |

Run from the project root:

```shell
hf download val-utehy/RotEx checkpoints/full/best.pt --local-dir .
```

Optional final checkpoint:

```shell
hf download val-utehy/RotEx checkpoints/full/last.pt --local-dir .
```

Use with `infer_multistep.py` or `infer_multistep_vid.py`.

Usage: [main README](../../README.md).
