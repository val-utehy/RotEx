# One-Step Weights

Download: [best_distill.pt](https://huggingface.co/val-utehy/RotEx/resolve/main/distill_checkpoints/best_distill.pt?download=true).

Run from the project root:

```shell
hf download val-utehy/RotEx distill_checkpoints/best_distill.pt --local-dir .
```

In `infer_1step_style.py`:

```python
CHECKPOINT = "distill_checkpoints/best_distill.pt"
```

Usage: [main README](../README.md).
