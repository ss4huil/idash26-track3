# iDASH Track 3 GPU-MPC 框架完整指南

## 框架概述

这是一个基于GPU加速的多方安全计算(MPC)框架，用于在保护隐私的前提下进行药物-靶点亲和力预测。该框架实现了DeepDTAGen模型的安全两方计算版本。

## 目录结构与文件职能

### 核心目录

```
idash/mpc/
├── gpu_mpc/                    # GPU-MPC 主实现目录
│   ├── deepdtagen_inference.cu # 2PC推理主程序（dealer + online评估）
│   ├── deepdtagen.h            # DeepDTAGen亲和力模型定义
│   ├── ddg_orca.h              # Orca后端扩展（支持mul/scalarmul）
│   ├── ddg_orca_base.h         # Orca基类（matmul, conv2d等）
│   ├── ddg_orca_opt.h          # 计算图优化（融合ReLU-MaxPool等）
│   ├── dpf_dcf_adapter.h       # **新增**：DPF-DCF适配器（Algorithm 1）
│   ├── gcn_layer.h             # 图卷积层（GCN）实现
│   ├── masked_maxpool.h        # 带掩码的全局最大池化
│   ├── pool_override.c         # CuBLAS内存池拦截器
│   ├── Makefile                # 编译配置
│   ├── run_local_2pc.sh        # 本地2PC测试脚本
│   ├── keys_tmp/               # FSS密钥存储目录
│   └── shares_tmp/             # 秘密分享数据存储目录
│
├── reference/                  # Python参考实现
│   ├── offline_prepare.py      # 离线准备：生成秘密分享和密钥
│   ├── share_data.py           # 秘密分享数据生成
│   ├── export_weights.py       # 导出模型权重为二进制格式
│   ├── fixed_forward.py        # 定点前向传播参考
│   ├── fixedpoint.py           # 定点数运算库
│   ├── affinity_model.py       # DeepDTAGen模型定义
│   ├── dense_graph.py          # SMILES到图转换
│   ├── protein_plaintext.py    # 蛋白质序列处理
│   └── mpc_config.py           # MPC配置（位宽、缩放因子）
│
├── model/                      # 训练好的模型权重
│   ├── deepdtagen_model_davis.pth
│   ├── deepdtagen_model_kiba.pth
│   └── deepdtagen_model_bindingdb.pth
│
├── baseline/                   # 明文基线结果（用于验证）
│   ├── official_baseline_davis.json
│   └── official_baseline_kiba.json
│
├── tests/                      # Davis/Kiba MPC 准确性门测试
│   ├── test_mpc_online_gate.py # 2PC在线准确性门测试
│   ├── test_official_baseline.py # 官方明文基线验证
│   └── conftest.py, test_data_paths.py # 配置文件
│
├── offline/                    # 离线数据准备测试
│   ├── test_offline_prepare.py # 离线准备驱动
│   ├── test_share_data.py      # 秘密分享测试
│   └── ...                     # 权重/NPZ导出
│
└── microbench/                 # 组件单元测试
    ├── test_dense_gcn.py       # GCN层测试
    ├── test_fixedpoint.py      # 定点算术测试
    └── ...                     # 其他底层组件
    └── ...其他单元测试

```

## 关键文件详解

### 1. GPU-MPC核心实现

#### `deepdtagen_inference.cu`
- **职能**: 主程序入口，实现dealer和online两个角色
- **功能**:
  - **Role 0 (Dealer)**: FSS密钥生成（离线阶段）
  - **Role 1 (Online)**: 2PC在线评估
  - 加载秘密分享的输入数据
  - 加载模型权重
  - 执行安全计算并揭示最终亲和力值

#### `deepdtagen.h`
- **职能**: DeepDTAGen亲和力预测路径的模型定义
- **架构**:
  ```
  Drug分支 (secret, MPC):
    GCN1: ReLU(A_hat @ (X @ W1 + b1))    94 → 188
    GCN2: ReLU(A_hat @ (H1 @ W2 + b2))   188 → 282
    GCN3: ReLU(A_hat @ (H2 @ W3 + b3))   282 → 376
    Pool: masked global max              (138×376) → (1×376)
    DFC1: ReLU(pool @ Wd1 + bd1)         376 → 1024
    DFC2: d1 @ Wd2 + bd2                 1024 → 128
  
  Protein分支 (public):
    GatedCNN → 128维embedding (P2计算)
  
  Fusion分支 (secret, MPC):
    Concat[drug_emb, protein_emb]        → 256
    FFC1: ReLU(. @ Wf1 + bf1)            256 → 1024
    FFC2: ReLU(. @ Wf2 + bf2)            1024 → 512
    FFC3: ReLU(. @ Wf3 + bf3)            512 → 256
    Out:  . @ Wo + bo                    256 → 1 (亲和力)
  ```

#### `ddg_orca.h`
- **职能**: Orca MPC后端的扩展，添加图神经网络所需操作
- **新增功能**:
  - `mul()`: 秘密×秘密逐元素乘法（用于GCN和masked maxpool）
  - `scalarmul()`: 公开标量乘法（线性组合）
  - **DPF-DCF替换**: 所有比较操作现在使用Algorithm 1（FSS-DT论文）

#### `dpf_dcf_adapter.h` ⭐**新增文件**
- **职能**: DPF-based DCF适配器，替换旧DCF实现
- **核心功能**:
  - 包装`fss/gpu_dpf.cu`中的DPF-based DCF（Algorithm 1）
  - 提供与旧`fss/dcf/*`兼容的API
  - 处理布尔分享→加性分享的转换
  - 实现ReLU、truncation、maxpool的密钥生成和评估

#### `gcn_layer.h`
- **职能**: 图卷积层实现
- **优化**: aggregate-first策略（先聚合邻居特征，再应用权重）
- **公式**: `H' = ReLU(A_hat @ (H @ W + b))`

#### `masked_maxpool.h`
- **职能**: 带掩码的全局最大池化（排除padding节点）
- **实现**: 使用`max(a,b) = a + ReLU(b-a)`公式迭代计算
- **关键**: 确保padding节点不影响最大值计算

### 2. Python参考实现

#### `offline_prepare.py`
- **职能**: 离线准备阶段的主脚本
- **流程**:
  1. 从CSV加载药物SMILES和蛋白质序列
  2. 将SMILES转换为图结构（邻接矩阵、节点特征）
  3. 计算蛋白质embedding（GatedCNN）
  4. 生成秘密分享（P0和P1）
  5. 导出模型权重为二进制格式
- **输出**:
  - `x_share{0,1}.dat`: 节点特征分享
  - `adj_share{0,1}.dat`: 邻接矩阵分享
  - `mask_share{0,1}.dat`: 节点掩码分享
  - `protein_emb.dat`: 蛋白质embedding
  - `weights.bin`: 模型权重二进制

#### `share_data.py`
- **职能**: 秘密分享数据生成
- **方法**: 加性秘密分享 `x = x0 + x1 (mod 2^bw)`
- **定点数**: Q20.12格式（bw=32，scale=12）

#### `export_weights.py`
- **职能**: 导出PyTorch模型权重为GPU-MPC可读的二进制格式
- **格式**: 小端序int64数组（与模型前向传播顺序对应）

### 3. 编译和配置

#### `Makefile`
- **职能**: 编译配置
- **重要参数**:
  - `BW`: 环位宽（32或64）
  - `GPU_ARCH`: GPU架构（89 for RTX 4060, 90a for H100）
  - `GPU_MPC_ROOT`: EzPC/GPU-MPC路径
- **目标**:
  - `deepdtagen_inference`: 主二进制
  - `clean`: 清理编译产物

## 完整运行流程

### 前提条件

#### 1. 硬件要求
- NVIDIA GPU（支持CUDA 12.1+）
- 至少8GB GPU内存
- 至少16GB系统内存

#### 2. 软件依赖

**CUDA工具链**:
```bash
# 安装CUDA 12.1+
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

**Python依赖**:
```bash
# Python 3.8+
pip install torch pandas numpy "numpy<2"  # numpy<2 for rdkit compatibility
pip install rdkit-pypi
```

**EzPC/GPU-MPC**:
```bash
# 克隆并编译EzPC/GPU-MPC
cd ~
git clone https://github.com/mpc-msri/EzPC.git
cd EzPC/GPU-MPC/ext/sytorch
mkdir build && cd build
cmake .. && make -j
```

### 步骤1: 编译GPU-MPC推理程序

```bash
cd idash/mpc/gpu_mpc

# 设置环境变量
export GPU_MPC_ROOT=$HOME/EzPC/GPU-MPC
export PATH=/usr/local/cuda-12.1/bin:$PATH

# 编译（RTX 4060示例）
make GPU_MPC_ROOT=$GPU_MPC_ROOT BW=32 GPU_ARCH=89 deepdtagen_inference

# 验证编译
ls -lh deepdtagen_inference  # 应该看到~2MB的二进制文件
```

### 步骤2: 准备测试数据（离线阶段）

```bash
cd idash/mpc

# 方式1: 准备单个样本
python3 << 'EOF'
from reference import offline_prepare as op
import tempfile

tmp_dir = tempfile.mkdtemp(prefix="mpc_sample_")
result = op.prepare_sample(
    dataset="davis",  # 或 "kiba"
    csv_path="/path/to/davis_test.csv",
    row_idx=0,  # 测试集第0行
    out_dir=tmp_dir,
    scale=12,   # Q20.12定点格式
    bw=32       # 32位环
)

print(f"Sample prepared at: {result['sample_dir']}")
print(f"Weights at: {result['weights_path']}")
EOF

# 方式2: 批量准备（测试脚本会自动调用）
```

**生成的文件**:
```
<sample_dir>/
├── x_share0.dat           # P0的节点特征分享
├── x_share1.dat           # P1的节点特征分享
├── adj_share0.dat         # P0的邻接矩阵分享
├── adj_share1.dat         # P1的邻接矩阵分享
├── mask_share0.dat        # P0的掩码分享
├── mask_share1.dat        # P1的掩码分享
├── protein_emb.dat        # 蛋白质embedding（公开）
└── weights.bin            # 模型权重（公开）
```

### 步骤3: 运行2PC推理

#### 方式A: 使用测试脚本（推荐）

```bash
cd idash/mpc/gpu_mpc

# 运行本地2PC测试
./run_local_2pc.sh <sample_dir> <key_dir> <weights.bin>

# 示例
./run_local_2pc.sh /tmp/mpc_sample_xyz/row_0 ./keys_tmp /tmp/mpc_sample_xyz/weights.bin
```

**脚本执行流程**:
1. **Dealer阶段** (role=0): 为P0和P1生成FSS密钥
   ```bash
   ./deepdtagen_inference 32 12 0 0 keys_tmp/ sample_dir/
   ./deepdtagen_inference 32 12 0 1 keys_tmp/ sample_dir/
   ```

2. **Online阶段** (role=1): P0和P1并行执行2PC
   ```bash
   # P1先启动（监听）
   ./deepdtagen_inference 32 12 1 1 keys_tmp/ sample_dir/ 127.0.0.1 &
   
   # P0连接
   ./deepdtagen_inference 32 12 1 0 keys_tmp/ sample_dir/ 127.0.0.1
   ```

3. **输出结果**: `AFFINITY=<float>` （揭示的亲和力值）

#### 方式B: 手动执行

```bash
cd idash/mpc/gpu_mpc

# 1. 生成密钥（dealer）
./deepdtagen_inference 32 12 0 0 ./keys_tmp/ /path/to/sample_dir/
./deepdtagen_inference 32 12 0 1 ./keys_tmp/ /path/to/sample_dir/

# 2. 运行在线2PC
# 在终端1（P1）
./deepdtagen_inference 32 12 1 1 ./keys_tmp/ /path/to/sample_dir/ 127.0.0.1

# 在终端2（P0）
./deepdtagen_inference 32 12 1 0 ./keys_tmp/ /path/to/sample_dir/ 127.0.0.1
```

**命令参数说明**:
```
deepdtagen_inference <bw> <scale> <role> <party> <key_dir> <sample_dir> [ip]

<bw>        : 环位宽（32或64）
<scale>     : 定点缩放因子（12 → Q20.12）
<role>      : 0=dealer（密钥生成）, 1=online（2PC评估）
<party>     : 0=P0, 1=P1
<key_dir>   : FSS密钥目录（必须以/结尾）
<sample_dir>: 样本数据目录
[ip]        : 对方IP地址（online阶段需要，默认127.0.0.1）
```

### 步骤4: 验证准确性

```bash
cd idash/mpc

# 运行准确性门测试
python3 -m pytest tests/test_mpc_online_gate.py -v

# 或手动比较
python3 << 'EOF'
import json

# 加载基线
with open('baseline/official_baseline_davis.json') as f:
    baseline = json.load(f)

# 从2PC输出中提取AFFINITY值
mpc_affinity = 5.0653  # 从run_local_2pc.sh输出获取

baseline_affinity = baseline['predictions'][0]  # 第0个样本
diff = abs(mpc_affinity - baseline_affinity)

print(f"MPC affinity:      {mpc_affinity:.6f}")
print(f"Baseline affinity: {baseline_affinity:.6f}")
print(f"Difference:        {diff:.6f}")
print(f"Status:            {'✓ PASS' if diff < 0.1 else '✗ FAIL'}")
EOF
```

**准确性要求**:
- 绝对误差 < 0.1
- 这验证了定点数运算和MPC协议的正确性

### 步骤5: 性能评估

```bash
cd idash/mpc

# 运行性能测试（多个样本）
python3 << 'EOF'
import time
import subprocess
import tempfile
from reference import offline_prepare as op

num_samples = 10
times = []

for i in range(num_samples):
    # 准备样本
    tmp_dir = tempfile.mkdtemp()
    result = op.prepare_sample("davis", "/path/to/davis_test.csv", 
                               row_idx=i, out_dir=tmp_dir, scale=12, bw=32)
    
    # 运行2PC
    start = time.time()
    subprocess.run([
        "bash", "gpu_mpc/run_local_2pc.sh",
        result['sample_dir'], f"{tmp_dir}/keys", result['weights_path']
    ], capture_output=True, check=True)
    elapsed = time.time() - start
    
    times.append(elapsed)
    print(f"Sample {i}: {elapsed:.2f}s")

print(f"\nAverage: {sum(times)/len(times):.2f}s per sample")
print(f"Total: {sum(times):.2f}s for {num_samples} samples")
EOF
```

## DPF-DCF替换说明

### 更改内容

我们将旧的DCF（Distributed Comparison Function）实现替换为基于DPF的DCF实现（FSS-DT论文Algorithm 1）:

**替换前**:
- 使用 `fss/dcf/*` 中的旧DCF实现
- 单独的GPU_relu、truncate、maxpool实现

**替换后**:
- 使用 `fss/gpu_dpf.cu` 中的DPF-based DCF
- 实现Algorithm 1（Lightweight Distributed Comparison Function）
- 通过`dpf_dcf_adapter.h`提供兼容API

### 技术细节

**Algorithm 1核心**:
- 使用DPF树遍历进行比较操作
- 输出布尔分享（通过epilogue函数转换为加性分享）
- 更高效的密钥生成和评估

**影响的操作**:
- ReLU: `ReLU(x) = x * (x > 0)` - 需要比较
- MaxPool: `max(a,b) = a + ReLU(b-a)` - 需要ReLU
- Truncation: 概率截断（本模型未使用）

### 验证方法

比较操作的正确性通过以下方式验证:
1. 编译成功（API兼容性）
2. 2PC准确性测试（功能正确性）
3. 与明文基线对比（数值准确性）

## 常见问题

### Q1: 编译错误 "nvcc: command not found"
```bash
# 确保CUDA在PATH中
export PATH=/usr/local/cuda-12.1/bin:$PATH
nvcc --version  # 验证
```

### Q2: 运行时错误 "cudaErrorNoKernelImageForDevice"
```bash
# GPU架构不匹配，重新编译
# 查看GPU架构
nvidia-smi --query-gpu=compute_cap --format=csv
# 重新编译，例如RTX 4060 (8.9)
make GPU_ARCH=89 clean deepdtagen_inference
```

### Q3: Python导入错误
```bash
# 确保在正确目录
cd idash/mpc
python3 -c "import sys; sys.path.insert(0, '.'); from reference import offline_prepare"
```

### Q4: 准确性测试失败（误差>0.1）
- 检查定点格式是否正确（bw=32, scale=12）
- 检查权重是否正确导出
- 检查秘密分享生成是否正确
- **新增**：验证DPF-DCF比较操作是否正确

### Q5: OOM错误
```bash
# 减少批量大小或使用更大内存的GPU
# 或者调整keyBufSize（在ddg_orca_base.h中）
```

## 性能指标

**预期性能**（RTX 4060，BW=32）:
- 单样本推理时间: ~10-30秒
- 密钥生成: ~5-10秒
- 在线阶段: ~5-20秒
- GPU内存占用: ~4-6GB

**与明文比较**:
- 明文推理: ~50ms
- MPC推理: ~10-30s
- 开销: ~200-600倍（这是MPC的典型开销）

## 调试技巧

### 启用调试输出
```bash
# 环境变量控制调试输出
export DDG_DEBUG_STAGE=h3    # 在GCN3后提前返回
export DDG_DEBUG_POOLED=1    # 输出pool后的向量
export DDG_NO_OUTPUT_TR=1    # 跳过最终截断（用于中间层对比）
```

### 查看中间层输出
```cpp
// 在deepdtagen.h中的_forward()函数
// 已有调试点：h1, h2, h3, masked, pooled
```

### 对比官方基线
```bash
cd idash/mpc
pytest tests/test_mpc_online_gate.py -v
# 这会运行真实2PC并对比官方baseline预测
```

## 文件大小参考

> **注意**: `weights.bin` 和 `.dat` 文件都是离线准备生成的产物（被 gitignore），
> 不提交到代码仓库。使用前需要先运行 `reference.offline_prepare.prepare_sample()`
> 或相关测试脚本生成。参见主 README "Generate offline artifacts" 步骤。

```
deepdtagen_inference    : ~2 MB     (编译后二进制)
weights.bin            : ~530 MB    (模型权重，离线生成)
x_share0.dat           : ~52 KB     (节点特征，138×94×4字节)
adj_share0.dat         : ~76 KB     (邻接矩阵，138×138×4字节)
mask_share0.dat        : ~52 KB     (掩码，138×376×4字节)
protein_emb.dat        : ~512 B     (128×4字节)
*_inference_key0.dat   : ~数MB到数GB（取决于模型复杂度）
```

## 总结

本框架实现了DeepDTAGen模型的GPU加速安全两方计算版本，支持在保护隐私的前提下进行药物-靶点亲和力预测。通过DPF-based DCF替换，比较操作现在使用更高效的Algorithm 1实现。完整的工作流程包括：

1. **编译**: 构建GPU-MPC推理程序
2. **离线准备**: 生成秘密分享和FSS密钥
3. **在线评估**: 执行2PC安全计算
4. **验证**: 与明文基线对比确认准确性

关键创新点：
- GPU加速的MPC原语（matmul, ReLU, maxpool等）
- 图神经网络的MPC支持（GCN层）
- **DPF-based比较操作**（Algorithm 1，更高效）
- 定点数运算确保数值稳定性
