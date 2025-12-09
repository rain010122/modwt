import os
import numpy as np
from modwt import MODWTPeriodExtractor
from data_loader import load_data

def main():
    dataset_dir = './datasets/one_year'
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 【核心配置】：针对 ADF-Net 的输入窗口约束
    # 假设你的 Look-back Window 最大是 720 (30天/1个月)
    # 对于小时数据，周期不能超过 512 (2^9)
    # 对于分钟数据，周期不能超过 1024 (2^10) 或 2048
    
    datasets_config = {
        # Hour level (24 points/day)
        # Target: Daily(24 -> 16/32), Weekly(168 -> 128/256)
        'ETTh1.csv':       {'max_p': 720}, 
        'ETTh2.csv':       {'max_p': 720},
        'Traffic.csv':     {'max_p': 720},
        'Electricity.csv': {'max_p': 720},
        
        # 15-min level (96 points/day)
        # Target: Daily(96 -> 64/128), Weekly(672 -> 512/1024)
        'ETTm1.csv':       {'max_p': 720}, 
        'ETTm2.csv':       {'max_p': 720},
        
        # 10-min level (144 points/day)
        # Target: Daily(144 -> 128/256)
        'Weather.csv':     {'max_p': 720}
    }

    # 提取 Top-3 即可，给模型 3 个专家视角
    extractor = MODWTPeriodExtractor(wavelet_name='db4', top_k=3)

    final_periods_dict = {}

    print("=" * 60)
    print("Start Wavelet-based Local Periodicity Extraction")
    print("Strategy: Masking global trends during extraction")
    print("=" * 60)

    for ds_name, config in datasets_config.items():
        file_path = os.path.join(dataset_dir, ds_name)
        if not os.path.exists(file_path):
            continue
            
        print(f"\nProcessing {ds_name}...")
        
        # 1. 加载数据
        series = load_data(file_path, scale=True)
        if series is None: continue
            
        # 2. 提取周期 (传入约束参数)
        max_p = config['max_p']
        results = extractor.extract(series, input_name=ds_name, max_allowed_period=max_p)
        
        # 3. 获取结果
        top_k_periods = [p for p, e in results['top_k']]
        
        # 4. 兜底策略：如果提取出来的全是 2, 4 这种极小噪声
        # 或者不足 3 个，我们可以手动补全一些更有意义的
        # 但通常经过上述修改，应该能提取到 16, 32, 64 等中频信号
        
        final_periods_dict[ds_name] = top_k_periods
        
        # 绘图
        plot_name = ds_name.replace('.csv', '_spectrum.png')
        extractor.plot_spectrum(results, save_path=os.path.join(output_dir, plot_name))

    print("\n" + "=" * 60)
    print("Final Periods Configuration for ADF-Net:")
    print("=" * 60)
    print("periods_map = {")
    for name, periods in final_periods_dict.items():
        # 格式化输出，方便直接复制到论文代码
        print(f"    '{name.split('.')[0]}': {periods},")
    print("}")

if __name__ == '__main__':
    main()