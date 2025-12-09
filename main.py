import os
import numpy as np
from modwt import MODWTPeriodExtractor
from data_loader import load_data

def main():
    # 1. 配置
    # 请确保将你的数据集 (ETTh1.csv 等) 放入 datasets 文件夹
    dataset_dir = './datasets'
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义要处理的数据集文件名
    datasets = [
        'ETTh1.csv',
        'ETTh2.csv', 
        'Traffic.csv',
        'Weather.csv',
        'Electricity.csv'
    ]

    # 初始化提取器 (使用 db4 小波, 提取 Top-3)
    extractor = MODWTPeriodExtractor(wavelet_name='db4', top_k=3)

    final_periods_dict = {}

    print("=" * 50)
    print("Start Wavelet-based Periodicity Extraction")
    print("=" * 50)

    for ds_name in datasets:
        file_path = os.path.join(dataset_dir, ds_name)
        
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}, skipping...")
            continue
            
        print(f"\nProcessing {ds_name}...")
        
        # 2. 加载数据
        # 这里默认取最后一列(OT)进行分析，代表该数据集的主节奏
        series = load_data(file_path, scale=True)
        
        if series is None:
            continue
            
        # 3. 提取周期
        results = extractor.extract(series, input_name=ds_name)
        
        # 4. 保存结果
        top_periods = [p for p, e in results['top_k']]
        final_periods_dict[ds_name] = top_periods
        
        # 5. 绘制并保存能量谱图 (论文素材!)
        plot_name = ds_name.replace('.csv', '_spectrum.png')
        extractor.plot_spectrum(results, save_path=os.path.join(output_dir, plot_name))

    print("\n" + "=" * 50)
    print("Final Extracted Periods (Ready for ADF-Net):")
    print("=" * 50)
    for name, periods in final_periods_dict.items():
        print(f"{name}: {periods}")
        # 这里输出的格式，可以直接复制到你论文代码的配置字典里

if __name__ == '__main__':
    main()