# iDASH Track 3 GPU-MPC DeepDTAGen 运行指南

## 环境要求

### 硬件
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥ 7.5
  - 测试配置: RTX 4060 (8GB VRAM, SM 8.9)
  - 最低要求: ≥4GB VRAM
- **CPU**: x86_64, ≥4 cores recommended
- **内存**: ≥16GB RAM
- **存储**: ≥10GB 可用空间

### 软件依赖

#### 系统软件
- **操作系统**: Linux (测试于 Ubuntu 20.04 / WSL2)
- **CUDA Toolkit**: 12.1+
  - `nvcc` (CUDA 编译器)
  - `cudart`, `cublas`, `curand` 运行时库
- **GCC**: 9.x - 11.x (CUDA 12.1 兼容)
- **GNU Make**: 4.x
- **Git**: 用于克隆 EzPC 仓库

#### Python 环境 (≥3.8)
```bash
# 核心依赖
pip install numpy scipy pandas
pip install torch==1.12.1+cu102 torchvision torchaudio  # PyTorch (CUDA 10.2+)
pip install torch-geometric torch-scatter torch-sparse   # PyG
pip install rdkit-pypi==2022.9.5                         # RDKit (化学信息学)
pip install networkx                                      # 图算法
```

---

## 安装步骤

### 1. 克隆 EzPC/GPU-MPC 框架
```bash
cd ~
git clone https://github.com/mpc-msri/EzPC.git
cd EzPC/GPU-MPC

# 检查提交哈希 (应为 66d9cddc 或更新)
git log --oneline -1
```

### 2. 设置 CUDA 路径
```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# WSL2 特定: 添加 Windows CUDA 库路径
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# 重新加载 shell 配置
source ~/.bashrc
```

### 3. 验证 CUDA 安装
```bash
nvcc --version
# 应输出: Cuda compilation tools, release 12.1, ...

nvidia-smi
# 应显示 GPU 信息和驱动版本
```

### 4. 构建 GPU-MPC DeepDTAGen
```bash
cd /path/to/idash/mpc/gpu_mpc

# 设置 EzPC 根目录
export GPU_MPC_ROOT=$HOME/EzPC/GPU-MPC

# 构建 (BW=32 for Q20.12 fixed-point, GPU_ARCH=89 for RTX 4060)
make GPU_MPC_ROOT=$GPU_MPC_ROOT BW=32 GPU_ARCH=89 deepdtagen_inference

# 验证构建成功
ls -lh deepdtagen_inference
# 应输出: -rwxr-xr-x ... 2.1M ... deepdtagen_inference
```

**架构代码查询**:
```bash
# 查找你的 GPU 架构代码
nvidia-smi --query-gpu=name,compute_cap --format=csv
# 例如: RTX 4060, 8.9 → GPU_ARCH=89
#       RTX 3090, 8.6 → GPU_ARCH=86
```

---

## 快速开始

### 单样本 2PC 测试

```bash
cd /path/to/idash/mpc/gpu_mpc

# 运行本地 2PC (dealer + party 0 + party 1)
./run_local_2pc.sh \
    test_sample/sample_0 \
    /tmp/keys_test \
    test_sample/weights.bin

# 预期输出:
# [run_local_2pc.sh] dealer keygen for party 0 ...
# [dealer] keys written to /tmp/keys_test/DeepDTAGen_32_12
# [run_local_2pc.sh] dealer keygen for party 1 ...
# [dealer] keys written to /tmp/keys_test/DeepDTAGen_32_12
# [run_local_2pc.sh] starting online parties ...
# Average time taken (microseconds)=...
# Comm (B)=...
# AFFINITY=39.773438    ← 预测的亲和力值
```

**注意事项**:
- 第一次运行可能需要 2-3 分钟 (GPU JIT 编译 + 预热)
- 后续运行约 30-60 秒 / 样本
- 密钥目录必须不存在或为空 (否则会覆盖)

---

## 完整测试套件

### 1. 真实 2PC 基准测试
```bash
cd /path/to/idash/mpc
python3 real_gpu_2pc_benchmark.py

# 预期输出:
# === Davis Dataset ===
# Processing 10 samples...
# Pearson r=0.XXX, Spearman ρ=0.XXX
# MAE=X.XX, RMSE=X.XX
# Avg inference time: XX.Xs
#
# === Kiba Dataset ===
# ...
```

### 2. Davis 批量 MPC 评估
```bash
cd /path/to/idash/mpc
./run_davis_multibatch.sh 5 4  # 5批 × 4样本 = 20样本

# 聚合验证:
python3 scripts/dev_tools/aggregate_davis_validation.py 5
# 输出累积 MAE/RMSE
#   Times : drug=XX.Xs  prot=XX.Xs  fusion=X.XXs
```

### 3. 回归指标对比
```bash
cd /path/to/idash/mpc
python3 benchmark_regression.py

# 生成详细的回归对比表 (MSE, RMSE, Pearson, Spearman, CI, rm2)
# 输出到 stdout 和 benchmark_results.log
```

---

## 准备自定义样本

### 从 SMILES + 序列生成 2PC 输入

```bash
cd /path/to/idash/mpc

# 例子: Erlotinib + EGFR 激酶域
SMILES="n1cnc(c2cc(c(cc12)OCCOC)OCCOC)Nc1cc(ccc1)C#C"
SEQUENCE="MAAVILESIFLKRSQQKKKTSPLNFKKRLFLLTVHKLSYYEYDFERGRRGSKKGSIDVEKITCVETVVPEKNPPPERQIPRRGEESSEMEQISIIERFPYPFQVVYDEGPLYVFSPTEELRKRWIHQLKNVIRYNSDLVQKYHPCFWIDGQYLCCSQTAKNAMGCQILENRNGSLKPGSSHRKTKKPLPPTPEEDQILKKPLPPEPAAAPVSTSELKKVVALYDYMPMNANDLQLRKGDEYFILEESNLPWWRARDKNGQEGYIPSNYVTEAEDSIEMYEWYSKHMTRSQAEQLLKQEGKEGGFIVRDSSKAGKYTVSVFAKSTGDPQGVIRHYVVCSTPQSQYYLAARNCLVNDQGVVKVSDFGLSRYVLDDEYTSSVGSKFPVRWSPPEVLMYSKFSSKSDIWAFGVLMWEIYSLGKMPYERFTNSETAEHIAQGLRLYRPHLASEKVYTIMYSCWHEKADERPTFKILLSNILDVMDEES"

python3 prepare_sample.py /tmp/my_sample "$SMILES" "$SEQUENCE"

# 生成的文件:
# /tmp/my_sample/
#   ├── x_share0.dat         # 节点特征 (party 0)
#   ├── x_share1.dat         # 节点特征 (party 1)
#   ├── adj_share0.dat       # 邻接矩阵 (party 0)
#   ├── adj_share1.dat       # 邻接矩阵 (party 1)
#   ├── mask_share0.dat      # 掩码 (party 0)
#   ├── mask_share1.dat      # 掩码 (party 1)
#   └── protein_emb.dat      # 蛋白质嵌入 (party 1 only)
```

### 运行自定义样本的 2PC

```bash
cd /path/to/idash/mpc/gpu_mpc

./run_local_2pc.sh \
    /tmp/my_sample \
    /tmp/keys_my_sample \
    test_sample/weights.bin
```

---

## 性能调优

### GPU 内存池配置

编辑 `gpu_mpc/pool_override.c`:
```c
// 默认 defrag threshold = 0 (适用于 <40GB GPU)
// 对于大 GPU (≥40GB), 可以设置更高的阈值
threshold_bytes = 0;  // 改为 10ULL * 1024 * 1024 * 1024 (10GB)
```

重新构建:
```bash
make clean
make GPU_MPC_ROOT=$GPU_MPC_ROOT BW=32 GPU_ARCH=89 deepdtagen_inference
```

### 批量推理优化

当前实现是单样本的。对于批量推理:
1. **离线准备**: 预计算所有唯一 SMILES 的药物路径 → 缓存
2. **蛋白质缓存**: 预计算所有唯一序列的蛋白质嵌入
3. **融合批处理**: 仅运行便宜的融合 FC 层

参考 `real_gpu_2pc_benchmark.py` 中的缓存策略（按唯一 SMILES / 序列缓存）。

---

## 故障排除

### 问题: `nvcc: command not found`
**解决方案**: 确保 CUDA bin 目录在 PATH 中
```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
```

### 问题: `error while loading shared libraries: libcudart.so.12`
**解决方案**: 设置 LD_LIBRARY_PATH
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

### 问题: `CUDA out of memory`
**解决方案**:
1. 减小批量大小 (如果使用批处理)
2. 使用更大的 GPU (≥8GB VRAM)
3. 关闭其他 GPU 应用程序

### 问题: `Assertion failed: fopen`
**解决方案**: 确保密钥目录存在且可写
```bash
mkdir -p /tmp/keys_test
chmod 755 /tmp/keys_test
```

### 问题: 2PC 挂起 (无输出)
**解决方案**:
1. 检查 party 1 是否在监听 (它先启动)
2. 检查防火墙 (本地主机通常不是问题)
3. 增加超时: `timeout 300 ./run_local_2pc.sh ...`

### 问题: 非确定性输出 (不同运行产生不同结果)
**解决方案**: 这在旧版本中是一个 bug (已修复)。确保:
1. 使用最新的 `ddg_orca.h` (Sigma 原生实现)
2. 重新构建: `make clean && make ...`
3. 删除旧的密钥目录

---

## 高级用法

### 更改定点配置

编辑 `Makefile`:
```makefile
BW ?= 64          # 环宽度 (32 或 64 位)
SCALE ?= 24       # 小数位数 (Q40.24 vs Q20.12)
```

重新导出权重:
```bash
cd /path/to/idash/mpc
python3 reference/export_weights.py \
    model/deepdtagen_model_davis.pth \
    gpu_mpc/test_sample/weights_bw64_scale24.bin \
    --scale 24 --bw 64
```

### 网络 2PC (非本地主机)

在 Party 1 机器上:
```bash
./deepdtagen_inference 32 12 1 1 /keys/dir/ /sample/dir/ 0.0.0.0 > p1.log 2>&1
```

在 Party 0 机器上:
```bash
./deepdtagen_inference 32 12 1 0 /keys/dir/ /sample/dir/ <party1_ip> > p0.log 2>&1
```

**注意**: 确保端口 8000 (默认) 在两台机器之间开放。

### 使用不同模型 (KIBA / BindingDB)

```bash
# 导出 KIBA 权重
python3 reference/export_weights.py \
    model/deepdtagen_model_kiba.pth \
    gpu_mpc/weights_kiba.bin \
    --scale 12 --bw 32

# 准备 KIBA 样本
python3 prepare_sample.py /tmp/kiba_sample \
    "<kiba_smiles>" \
    "<kiba_sequence>"

# 运行 2PC
./run_local_2pc.sh /tmp/kiba_sample /tmp/keys_kiba gpu_mpc/weights_kiba.bin
```

---

## 性能基准

### 单样本 2PC (RTX 4060, localhost)

| 阶段           | 时间       | 通信量    |
|----------------|-----------|-----------|
| Dealer (P0)    | ~8s       | 0         |
| Dealer (P1)    | ~8s       | 0         |
| Online (2PC)   | ~30-60s   | ~500 MB   |

### 全数据集定点仿真 (CPU, Intel i7)

| 数据集 | 样本数   | 时间    | 吞吐量        |
|--------|---------|---------|--------------|
| Davis  | 5,010   | ~45s    | ~111 样本/秒 |
| KIBA   | 19,653  | ~180s   | ~109 样本/秒 |

**注意**: 真实 2PC 比定点仿真慢 ~1000× (由于加密通信)。

---

## 参考资料

- **EzPC/GPU-MPC**: https://github.com/mpc-msri/EzPC
- **DeepDTAGen 论文**: Lee et al., 2021 (原始模型)
- **iDASH Track 3 规范**: (官方挑战文档)
- **CUDA 编程指南**: https://docs.nvidia.com/cuda/
- **Sytorch 框架**: EzPC/GPU-MPC/ext/sytorch

---

## 许可与引用

本实现基于:
- **EzPC/GPU-MPC**: Microsoft Research (MIT License)
- **DeepDTAGen**: 原始作者 (学术用途)

如果在研究中使用此代码，请引用:
```bibtex
@inproceedings{ezpc-gpu-mpc-2024,
  title={EzPC: Programmable and Efficient Secure Two-Party Computation for Machine Learning},
  author={Jawalkar, Neha and others},
  booktitle={IEEE S\&P},
  year={2024}
}
```
