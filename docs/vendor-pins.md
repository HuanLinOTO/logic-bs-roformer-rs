# vendor 钉扎记录

## vendor/cuda-rust

| 日期 | commit | 来源 | 说明 |
|---|---|---|---|
| 2026-10-02 | `a0cc6cc0` | NVIDIA/cuda-rust upstream | 初次 vendor，nightly-2026-08-28 工具链 |
| 2026-10-06 | `f3738d88` | [ansidium/cuda-rust-windows](https://github.com/ansidium/cuda-rust-windows) fork/main | Windows 原生移植（+101 commits 于 a0cc6cc 之上）；stable 1.99 工具链 |

- 切换原因：Windows 原生构建/运行（见 `docs/plans/2026-10-07-windows-native-lbrr.md`）
- 基线验证：`git merge-base --is-ancestor a0cc6cc f3738d88` 通过（线性包含，无丢失）
- local branch: `fork-main`，remote: `fork` → https://github.com/ansidium/cuda-rust-windows
- 上游同步入口：`git fetch origin && git fetch fork`
