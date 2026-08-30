# Third-Party Notices

This repository vendors the following projects for reproducible MotionInsight inference and training. Their own license files control use and redistribution of those components.

| Component | Upstream | Revision | License |
|---|---|---|---|
| CoTracker | https://github.com/JohnZhan2023/co-tracker | `ca0a716` (based on upstream `82e02e8`) | CC BY-NC 4.0 |
| SAM3 | https://github.com/JohnZhan2023/sam3 | `828a87a` (based on upstream `86ed770`) | SAM License |
| VIPE | https://github.com/JohnZhan2023/vipe | `7e66b5a` | Apache-2.0 |

The vendored CoTracker and SAM3 trees include local compatibility changes used by the MotionInsight feature pipeline. Generated caches, package metadata, compiled VIPE extensions, and example VIPE outputs are excluded.

The files under `training/` adapt Apache-2.0-licensed Hugging Face TRL and R1-V
training code. Their retained source headers identify the applicable copyright and
license terms.

The components are tracked as git submodules. License texts are available after
running `git submodule update --init --recursive`:

- `thirdparty/co-tracker/LICENSE.md`
- `thirdparty/sam3/LICENSE`
- `thirdparty/vipe/LICENSE`
- `thirdparty/vipe/THIRD_PARTY_LICENSES.md`
