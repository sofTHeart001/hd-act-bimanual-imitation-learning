# T1 Record

This repository records a local TronCamp Mani T1 ACT baseline run.

## Task

- Track: T1
- Task: adjust_bottle
- Dataset: 200 local demonstrations
- Policy: ACT
- Training seed: 0
- Best validation loss: 0.028757 at epoch 5125

## Local Evaluation

Evaluation used the official local evaluation entrypoint on the public 100-seed table.

```json
{
  "sr": 0.52,
  "n_repeats": 1,
  "n_episodes": 100,
  "per_repeat": [0.52],
  "track": "T1"
}
```

## Submission

- Track: T1
- Checkpoint submitted: `policy_best.ckpt`
- Official queue id: `#70`
- Submission time: 2026-07-08

Large generated artifacts are intentionally not included in this repository:

- collected demonstrations
- processed training data
- ACT checkpoints
- local evaluation logs
- submission tokens or credentials
