# Navigation DDP / NCCL 黑皮书

这份文档记录 `qwen-vl-finetune/scripts/sft_nav_3b.sh` 在 Blackwell 双卡机上的多卡调试结论。

## 结论

- `base` 环境可直接看到两张 `NVIDIA RTX PRO 6000 Blackwell Server Edition`。
- `torch 2.7.0+cu128` + `NCCL 2.26.2` 本机基础通信没问题：2 卡 `all_reduce` / `all_gather` / `reduce_scatter` 都通过。
- 纯 PyTorch `DDP` 的小模型训练也能跑通。
- 这条 Qwen2.5-VL 训练链在两卡上仍会在 `destroyEvent` / `illegal memory access` 位置退出，单卡可正常跑完 1 step。
- 所以当前根因更像是 Qwen2.5-VL 多卡训练路径上的 CUDA/kernel 兼容问题，而不是启动脚本或 NCCL 版本本身。

## 已修改内容

- `scripts/sft_nav_3b.sh`
  - 去掉了 `CUDA_VISIBLE_DEVICES=0`
  - 默认 `NPROC_PER_NODE=$(torch.cuda.device_count())`
  - 增加了 `NNODES` / `NODE_RANK`
  - 增加了 `NCCL_DEBUG` / `NCCL_ASYNC_ERROR_HANDLING`
  - `USE_DEEPSPEED=1` 时才启用 DeepSpeed
  - 支持 `MAX_STEPS` 做快速 smoke test

- `qwenvl/train/train_qwen.py`
  - 显式从 `LOCAL_RANK` 读取本地 rank
  - 调用 `torch.cuda.set_device(local_rank)`，确保每个进程绑定到自己的 GPU
  - 加了 `DEBUG_TRAIN=1` 调试打印，方便确认是否进入 nav graph

## 推荐启动

```bash
cd /root/Qwen3-VL/qwen-vl-finetune
conda activate base
LLM_MODEL=/root/autodl-fs/Qwen2.5-VL-3B-Instruct \
DATASETS=llava_nav_v8 \
bash scripts/sft_nav_3b.sh
```

单机两卡时，默认会自动起 2 个进程。

## 快速 smoke test

先跑 1 步确认多卡、数据和通信都通：

```bash
cd /root/Qwen3-VL/qwen-vl-finetune
LLM_MODEL=/root/autodl-fs/Qwen2.5-VL-3B-Instruct \
DATASETS=llava_nav_v8 \
MAX_STEPS=1 \
NCCL_DEBUG=INFO \
bash scripts/sft_nav_3b.sh
```

## 如果还报 NCCL 错

按这个顺序排：

1. 先确认 `python -c "import torch; print(torch.cuda.device_count())"` 是否能看到所有 GPU。
2. 跑最小通信测试：
   - `torchrun --nproc_per_node=2 ... all_reduce`
3. 看是否是网络选择问题：
   - 只做单机训练时，通常不需要改 IB
   - 如果出现 socket / timeout，再显式设置 `NCCL_SOCKET_IFNAME`
4. 看是否是 DDP 无用参数同步问题：
   - 当前脚本已设置 `--ddp_find_unused_parameters False`
5. 再看是否是模型 / DeepSpeed 组合问题：
   - 先 `MAX_STEPS=1`，再放大到正式训练

## 本机验证记录

- `torchrun --nproc_per_node=2` 的最小 all-reduce 通过。
- `torchrun --nproc_per_node=2` 的 `all_gather_into_tensor` / `reduce_scatter_tensor` 也通过。

这说明当前问题优先级更高的是 rank-to-GPU 绑定，而不是 NCCL 版本本身。
