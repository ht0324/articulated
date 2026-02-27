# SO2 Scaling Sweep Infrastructure

This folder defines the sweep specification used by `scripts/run_scaling_sweep.py`.

## Files

- `base.yaml`: sweep grid + global defaults
- `architectures/*.yaml`: architecture-specific model overrides

## Launch

```bash
python scripts/run_scaling_sweep.py \
  --sweep-config articulated/configs/estimation/sweeps/so2_scaling/base.yaml
```

Smoke test (plan only):

```bash
python scripts/run_scaling_sweep.py \
  --sweep-config articulated/configs/estimation/sweeps/so2_scaling/base.yaml \
  --dry-run \
  --max-runs 2
```

## Summarize

```bash
python scripts/summarize_scaling.py \
  --runs-root /mldata/huntae/runs/so2_repr_scaling_template
```

## Plugging in a new RNN architecture

1. Create a new file under `architectures/` (or edit `custom_rnn_placeholder.yaml`).
2. Set a unique `name` and add model overrides under `overrides.model`.
3. Add that file path to `architectures:` in `base.yaml`.
4. Re-run `run_scaling_sweep.py`.

No runner changes are required if the training config schema stays compatible.
