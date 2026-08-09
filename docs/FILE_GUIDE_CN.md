# iDASH Track 3 GPU-MPC DeepDTAGen 项目文件说明

## 目录结构

```
idash/mpc/
├── gpu_mpc/                    # GPU-MPC 2PC 实现核心代码
│   ├── deepdtagen_inference.cu # 主推理二进制（dealer + online 2PC）
│   ├── deepdtagen.h            # DeepDTAGen 模型图定义
│   ├── ddg_orca.h              # ORCA 后端（已替换为 Sigma 原生实现）
│   ├── ddg_orca_base.h         # ORCA 基类（内存池、优化等）
│   ├── ddg_orca_opt.h          # 模式传播（mode 0/1/2/3 bitwidth 规则）
│   ├── gcn_layer.h             # GCN 层（聚合优先 + FC + ReLU）
│   ├── masked_maxpool.h        # Masked 全局 MaxPool
│   ├── dpf_dcf_adapter.h       # [已弃用] 自定义 DPF 适配器（被 Sigma 原生替代）
│   ├── pool_override.c         # GPU 内存池钩子（defrag 设置）
│   ├── Makefile                # 构建系统
│   ├── run_local_2pc.sh        # 单机 2PC 测试脚本
│   ├── sytorch/                # Sytorch 框架副本（Concat 修复）
│   └── test_sample/            # 测试样本（sample_0 + weights.bin）
│
├── reference/                  # Python 参考实现（明文 + 定点仿真）
│   ├── affinity_model.py       # 明文 PyTorch 模型加载器
│   ├── fixed_forward.py        # 定点前向传播（MPC 数值等价）
│   ├── dense_graph.py          # SMILES → 稠密图（Nmax=138）
│   ├── dense_gcn.py            # 稠密 GCN 层
│   ├── masked_maxpool.py       # Masked MaxPool（NumPy）
│   ├── protein_plaintext.py    # 蛋白质序列 → GatedCNN 嵌入
│   ├── fixedpoint.py           # 定点算术工具
│   ├── metrics.py              # 回归 / 分类指标
│   ├── export_weights.py       # PyTorch → 定点权重二进制
│   ├── offline_prepare.py      # 生成秘密共享（dealer 输入）
│   └── share_data.py           # 共享序列化工具
│
├── model/                      # 预训练权重（PyTorch .pth 格式）
│   ├── deepdtagen_model_davis.pth
│   ├── deepdtagen_model_kiba.pth
│   └── deepdtagen_model_bindingdb.pth
│
├── baseline/                   # 官方明文基线结果
│   ├── official_baseline_davis.json   # Davis 测试集预测 + 指标
│   ├── official_baseline_kiba.json    # KIBA 测试集预测 + 指标
│   └── official_baseline_data.py      # 基线生成脚本
│
├── real_gpu_2pc_benchmark.py   # 真实 2PC 基准测试（计时 + 回归统计）
└── run_davis_multibatch.sh     # Davis 批量 MPC 评估
```

---

## 核心文件详解

### 1. GPU-MPC 2PC 实现 (`gpu_mpc/`)

#### `deepdtagen_inference.cu` — 主推理驱动程序
- **作用**：单一二进制，支持两种角色：
  - **Role 0 (Dealer)**：FSS 密钥生成（离线阶段）
  - **Role 1 (Evaluator)**：在线 2PC 协议（Party 0 和 Party 1）
- **输入**：
  - 秘密共享：`x_share{0,1}.dat`, `adj_share{0,1}.dat`, `mask_share{0,1}.dat`
  - 蛋白质嵌入：`protein_emb.dat`（Party 1 加载，Party 0 为零）
  - 权重：`weights.bin`（导出的 int64 定点权重）
- **输出**：`AFFINITY=<float>` 到 stdout

#### `deepdtagen.h` — 模型图定义
- **类**：`DeepDTAGenAffinity<T>` 继承自 `SytorchModule<T>`
- **架构**：
  ```
  DrugPath:  GCN(X, A_hat) → GCN → GCN → MaskedMaxPool → FC(1024) → ReLU → FC(128)
  ProteinPath: proteinEmb (预计算，128-d)
  Fusion:    Concat(drug_emb, protein_emb) → FC(1024) → ReLU → FC(512) → ReLU → FC(256) → ReLU → FC(1)
  ```
- **侧输入**：`A_hat`（图邻接矩阵）、`maskTiled`（节点掩码）、`proteinEmb`（蛋白质嵌入）

#### `ddg_orca.h` — ORCA 后端（Sigma 原生实现）
- **最新修改**：替换了自定义随机截断 + 符号扩展为 **Sigma 原生确定性实现**：
  - **ReLU**：`gpuGenReluKey` / `gpuRelu`（DPF 两轮 ReLU）
  - **截断**：`genGPUTruncateKey` / `gpuTruncate`（`TruncateType::TrFloor`，确定性 ARS + MSB 校正）
  - **符号扩展**：`genGPUSignExtendKey` / `gpuSignExtend`（DPF + 校正表）
- **保留**：
  - `mul`：秘密×秘密乘法（Beaver 三元组 + 截断）
  - `scalarmul`：秘密×公开标量乘法
  - `maxPool2D`：Masked 最大池化（DCF 比较）
- **关键修复**：FC 层非确定性 bug —— 原因是手写的 `dpf_dcf_adapter.h` 中的 `signext()` 掩码不匹配；现已替换为 Sigma 原生版本，结果确定性且正确。

#### `ddg_orca_base.h` — 基类 + 优化
- **类**：`DDGOrcaBase<T>`（评估器）、`DDGOrcaBaseKeygen<T>`（密钥生成）
- **功能**：
  - 内存池初始化（GPU defragmentation threshold）
  - 图优化（钉住 CPU 内存、模式传播）
  - 密钥 I/O（`readKey` / `writeKey`）

#### `ddg_orca_opt.h` — 模式传播规则
- **模式系统**（4 种）：
  - **Mode 0**：ℓ → ℓ 位（全精度）
  - **Mode 1**：ℓ → ℓ-scale 位（截断后）
  - **Mode 3**：ℓ-scale → ℓ-scale 位（融合截断 + ReLU）
  - **Mode 2**：ℓ-scale → ℓ 位（ReluExtend，重新扩展到全精度）
- **触发逻辑**：
  - `doPreSignExtension`：当父节点输出为缩减比特宽度（mode 1/3）时触发
  - `doTruncationForward`：所有 matmul / mul 后执行
  - `doPostSignExtension`：特殊情况（一般不用）

#### `gcn_layer.h` — GCN 层
- **实现**：聚合优先（`matmul(A_hat, X)` → `lin.forward(AX)` → `relu(Z)`）
- **偏置融合**：GCN 偏置折叠到 P2 的 FC 权重中（简化秘密共享）

#### `masked_maxpool.h` — Masked 全局最大池化
- **功能**：`MaskedGlobalMaxPool<T>` 在节点维度（Nmax）上执行掩码池化
- **实现**：逐元素乘以 mask，然后 `maxPool2D` 跨节点维度

#### `sytorch/` — Sytorch 修改副本
- **修复**：`Concat` 层的 GPU 内存上传（`moveToGPU` 在 `_forward` 中）
- **原因**：原始 Sytorch 不将拼接结果上传到 GPU → 融合路径崩溃

#### `Makefile`
- **变量**：
  - `GPU_MPC_ROOT`：指向 EzPC/GPU-MPC 的路径
  - `BW`：环宽度（32 或 64）
  - `GPU_ARCH`：CUDA 架构（89 for RTX 4060）
- **目标**：`deepdtagen_inference`（nvcc 编译 CUDA + 链接 GPU-MPC）

#### `run_local_2pc.sh` — 单机 2PC 测试
- **用法**：`./run_local_2pc.sh <sample_dir> <key_dir> <weights_bin>`
- **步骤**：
  1. Dealer 为 Party 0 生成密钥
  2. Dealer 为 Party 1 生成密钥
  3. 并行启动 Party 1（监听）和 Party 0（连接）
  4. 从 Party 0 输出中提取 `AFFINITY=` 行

---

### 2. Python 参考实现 (`reference/`)

#### `affinity_model.py` — 明文 PyTorch 模型
- **类**：`AffinityModel`
- **方法**：
  - `from_pth(pth_path)`：从 `.pth` 文件加载权重
  - `drug_path(X, A_hat, mask)`：药物路径前向传播 → 128-d 嵌入
  - `protein_path(protein_seq)`：蛋白质路径 → 128-d 嵌入
  - `predict(X, A_hat, mask, protein_seq)`：完整亲和力预测

#### `fixed_forward.py` — 定点前向传播
- **类**：`FixedAffinity`
- **目的**：数值上等价于 MPC 在秘密共享下的计算（无加密开销）
- **方法**：
  - `_drug_path_fx(X, A_hat, mask)`：定点药物路径（Q20.12）
  - `_fusion_fx(pmvo_fx, pvec_fx)`：定点融合层
  - `predict(X, A_hat, mask, protein_seq)`：端到端定点预测

#### `dense_graph.py` — SMILES → 图
- **函数**：`smile_to_dense_graph(smiles, nmax)`
- **输出**：
  - `X`：节点特征矩阵（Nmax × 94）
  - `A_hat`：归一化邻接矩阵（Nmax × Nmax）
  - `mask`：节点掩码（Nmax,）

#### `export_weights.py` — 权重导出
- **功能**：将 PyTorch `.pth` 权重转换为定点 int64 二进制 `weights.bin`
- **格式**：无头部，小端序，按拓扑顺序序列化层权重

#### `offline_prepare.py` — 秘密共享生成
- **功能**：将 SMILES + 序列对转换为 2PC 输入（药物图 + 蛋白质嵌入的共享）
- **输出**：`{x,adj,mask}_share{0,1}.dat`, `protein_emb.dat`

#### `metrics.py` — 评估指标
- **函数**：
  - `sensitivity`, `specificity`, `sens_spec_accuracy`（分类指标）
  - `is_qualified(original_acc, mpc_acc)`：精度门（<2pp 下降）

---

### 3. 评估脚本

#### `real_gpu_2pc_benchmark.py` — 真实 2PC 基准测试
- **作用**：在 davis 和 kiba 测试集样本上运行真实加密 2PC
- **指标**：Pearson、Spearman 相关系数 + MAE、RMSE + 推理计时
- **样本数**：默认每数据集 10 个样本
- **输出**：回归统计摘要 + 每样本计时

#### `run_davis_multibatch.sh` — Davis 批量 MPC 评估
- **作用**：多批次运行 MPC 推理（B=4 样本/批），累积 MAE/RMSE
- **用法**：`./run_davis_multibatch.sh <批数> <批大小>`
- **示例**：`./run_davis_multibatch.sh 5 4` → 5 批 × 4 样本 = 20 样本
- **聚合**：`python3 scripts/dev_tools/aggregate_davis_validation.py 5`


---

### 4. 数据文件

#### `model/` — 预训练权重
- **来源**：官方 DeepDTAGen 发布（iDASH Track 3）
- **格式**：PyTorch `.pth`（OrderedDict，float32）

#### `baseline/` — 明文基线
- **文件**：
  - `official_baseline_davis.json`：5010 样本，Pearson=0.8697
  - `official_baseline_kiba.json`：19653 样本，Pearson=0.8867
- **内容**：
  - `ground_truth`：真实亲和力值
  - `predictions`：模型预测（float32）
  - `mse`, `rmse`, `pearson`, `spearman`, `cindex`, `rm2`, `aupr`：指标

#### `gpu_mpc/test_sample/` — 单样本测试数据
- **sample_0/**：第一个 davis 测试样本的共享
- **weights.bin**：定点权重二进制（Q20.12，int64）

---

## 关键技术点

### 1. 定点算术
- **格式**：Q20.12（32 位环，12 位小数）
- **截断**：ARS（算术右移）+ MSB 校正（TrFloor）
- **符号扩展**：DPF + 校正表（处理 ℓ-scale → ℓ 位扩展）

### 2. 2PC 协议
- **密钥生成**：Dealer（Role 0）为两方生成 FSS 密钥
- **在线阶段**：Party 0 和 Party 1 在秘密共享上执行协议
- **通信**：DReLU（两轮）、Select、DCF/DPF、截断

### 3. 模式系统
- **目的**：跟踪整个图中的比特宽度，最小化截断 / 符号扩展
- **规则**：由 `ddg_orca_opt.h` 中的父 / 子模式逻辑传播

### 4. Sigma 原生实现（最新修复）
- **替换内容**：自定义随机截断 + 符号扩展 → Sigma 的 `TrFloor` + `gpuSignExtend`
- **结果**：确定性输出，FC 层 bug 已修复
- **效率**：单轮通信（vs 双轮随机），更好的代码重用

---

## 使用流程

### 构建
```bash
cd idash/mpc/gpu_mpc
export PATH=/usr/local/cuda-12.1/bin:$PATH
make GPU_MPC_ROOT=$HOME/EzPC/GPU-MPC BW=32 GPU_ARCH=89 deepdtagen_inference
```

### 单样本 2PC 测试
```bash
./run_local_2pc.sh test_sample/sample_0 /tmp/keys test_sample/weights.bin
# 输出: AFFINITY=<value>
```

### 全数据集回归对比
```bash
cd idash/mpc
python3 benchmark_regression.py
# 生成对比表：明文 vs MPC 定点
```

### 集成测试（5 样本）
```bash
python3 simple_2pc_test.py
# 验证 2PC 输出 vs 定点仿真
```

---

## 文件依赖关系

```
deepdtagen_inference.cu
  ├─ deepdtagen.h           (模型图)
  ├─ ddg_orca.h             (后端: Sigma 原生 ReLU/截断/符号扩展)
  │   ├─ ddg_orca_base.h    (基类: 内存池, 密钥 I/O)
  │   ├─ fss/gpu_relu.h     (Sigma DPF ReLU)
  │   ├─ fss/gpu_truncate.h (Sigma TrFloor 截断)
  │   └─ dpf_dcf_adapter.h  (ReluExtend 仅保留, 其他已废弃)
  ├─ gcn_layer.h            (GCN 层)
  ├─ masked_maxpool.h       (Masked MaxPool)
  └─ sytorch/layers/layers.h (Concat 修复)
```

---

## 已知限制

1. **MaxPool**：当前使用 DCF 实现（非 DPF），未优化
2. **ReluExtend**：Mode 2 仍使用旧 DCF（Sigma 无 DPF 版本）
3. **全数据集 2PC**：由于计算开销，未在完整测试集上运行真实 2PC
4. **通信优化**：未实现批处理压缩（每个门独立通信）

---

## 未来工作

1. 用 DPF 替换 MaxPool DCF
2. 实现批量样本 2PC（摊销通信成本）
3. 与 SPU BumbleBee 基线（官方安全参考）对比
4. 在真实网络延迟下测试 2PC（目前仅 localhost）
