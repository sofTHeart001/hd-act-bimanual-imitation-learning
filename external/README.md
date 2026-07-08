# external/ — 内嵌的第三方依赖 / Bundled third-party dependencies

本目录内嵌了选手包运行所依赖的第三方组件,**clone 即用,无需手动获取任何额外资源**。
参赛文档「§安装」的分步命令会基于这里的内容安装运行环境。

This directory bundles the third-party components the kit runs on. Everything needed
is already included — **no manual fetching is required**. The step-by-step setup in
the docs builds the environment from what is here.

## 内容 / Contents

- `robotwin_local/` — **RoboTwin 2.0** 双臂操作仿真 benchmark(含 Tron2 接入、四个任务的
  task config、`policy/ACT` 训练栈,以及 `envs/curobo` 运动规划)。
  RoboTwin 2.0 dual-arm manipulation simulation benchmark (with Tron2 integration,
  the four tasks' configs, the `policy/ACT` training stack, and `envs/curobo` motion
  planning).

## 许可 / Licensing

各第三方组件**各自遵循其上游许可证**,其原始 LICENSE 文件已原样保留在对应目录内。完整清单、
版权与上游地址见仓库根目录的 [`NOTICE`](../NOTICE)。

Each bundled component keeps its **own upstream license**; the original LICENSE files
are preserved in place. See the top-level [`NOTICE`](../NOTICE) for the full inventory,
copyrights, and upstream URLs.

> ℹ️ 内嵌的 NVIDIA cuRobo 以 **Apache-2.0**(v0.8.0)再分发;各第三方组件各自遵循其上游
> 许可证。完整清单与 Tron2 模型条款见 [`NOTICE`](../NOTICE)。
>
> ℹ️ The bundled NVIDIA cuRobo is redistributed under the **Apache License 2.0**
> (v0.8.0); each third-party component keeps its own upstream license. See
> [`NOTICE`](../NOTICE) for the full inventory and the Tron2 model terms.
