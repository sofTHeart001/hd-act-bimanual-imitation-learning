# LIMX VLA TronCamp Record

This repository is a public record of a TronCamp Mani T1 ACT baseline run.

It keeps the project scripts, configuration notes, and evaluation records, while excluding
large generated artifacts such as collected demonstrations, processed training data,
checkpoints, local logs, and credentials.

## T1 Result

- Track: T1
- Task: `adjust_bottle`
- Policy: ACT
- Demonstrations: 200 local episodes
- Public-seed local evaluation: `sr = 0.52`
- Episodes: 100
- Repeats: 1
- Official submission queue id: `#70`

See [records/t1_record.md](records/t1_record.md) for the detailed record.

## What Is Not Included

The following files are intentionally excluded:

- `external/robotwin_local/`
- collected `.hdf5` demonstrations
- `processed_data/`
- ACT `.ckpt` checkpoints
- local evaluation logs
- submission tokens or other credentials

## Notes

This is a record repository, not a full clone-and-run release. The local training run used
the TronCamp Mani starter package with RoboTwin installed under `external/robotwin_local`.
