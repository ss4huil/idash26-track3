#!/usr/bin/env python3
"""测量明文DeepDTAGen单样本推理时间"""
import sys, time
sys.path.insert(0, '/home/jiang/master/idash/mpc')

import torch
import pandas as pd
from baseline.official_baseline_data import load_model, _build_from_df, load_tokenizer

def time_single_inference(dataset, csv_path, row_idx):
    """测量单样本推理的实际时间（不含模型加载）"""
    # 加载模型和数据（不计入推理时间）
    tokenizer = load_tokenizer(dataset)
    model, device = load_model(dataset)

    df = pd.read_csv(csv_path)
    sample_df = df.iloc[[row_idx]]
    data_list = _build_from_df(sample_df, tokenizer)
    data = data_list[0].to(device)

    # 预热GPU（如果使用）
    with torch.no_grad():
        for _ in range(3):
            _, _, _, _ = model(data)

    # 实际计时
    times = []
    n_runs = 20
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            prediction, _, _, _ = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    pred_pkd = prediction.item()
    true_pkd = data.y.item()

    return {
        'device': str(device),
        'times_ms': times,
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'pred_pkd': pred_pkd,
        'true_pkd': true_pkd,
    }

if __name__ == '__main__':
    dataset = 'davis'
    csv_path = '/home/jiang/master/idash/project/DeepDTAGen/data/davis_test.csv'

    samples = [0, 3, 5, 9]  # 对应MPC测试的样本

    print(f"明文DeepDTAGen推理时间基准测试 (dataset={dataset})")
    print("=" * 70)

    for idx in samples:
        print(f"\n测试 sample_{idx}...")
        result = time_single_inference(dataset, csv_path, idx)
        print(f"  设备: {result['device']}")
        print(f"  预测值: {result['pred_pkd']:.4f} pKd (true={result['true_pkd']:.4f})")
        print(f"  推理时间: {result['mean_ms']:.2f} ± {(result['max_ms']-result['min_ms'])/2:.2f} ms")
        print(f"             (最小={result['min_ms']:.2f}, 最大={result['max_ms']:.2f}, n={len(result['times_ms'])})")
