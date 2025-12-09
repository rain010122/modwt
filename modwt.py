'''
为了实现 MODWT（不降采样），我们使用 pywt.swt (Stationary Wavelet Transform)，它在数学上等价于 MODWT。
'''
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os

class MODWTPeriodExtractor:
    def __init__(self, wavelet_name='db4', top_k=3):
        """
        初始化 MODWT 周期提取器
        :param wavelet_name: 母小波名称，论文中选用 'db4'
        :param top_k: 提取前 k 个主导周期
        """
        self.wavelet_name = wavelet_name
        self.top_k = top_k
        # 验证小波名称
        assert wavelet_name in pywt.wavelist(), f"Wavelet {wavelet_name} not found in PyWavelets."

    def _pad_sequence(self, data, max_level):
        """
        为了进行 J 层分解，序列长度必须能被 2^J 整除。
        这里使用反射填充 (Reflection Padding) 以减少边界效应。
        """
        n = len(data)
        target_len = int(np.ceil(n / (2**max_level))) * (2**max_level)
        if target_len > n:
            pad_width = target_len - n
            # 使用 'reflect' 模式填充，保持边界连续性
            data = np.pad(data, (0, pad_width), 'reflect')
        return data, n  # 返回填充后的数据和原始长度

    def extract(self, data_series, input_name="Series"):
        """
        执行 MODWT 并提取主导周期
        :param data_series: 一维 numpy 数组 (T,)
        :param input_name: 用于绘图的标题
        :return: 包含 (周期, 能量) 的字典列表
        """
        # 1. 确定最大分解层数 J <= log2(L)
        L = len(data_series)
        max_level = int(np.floor(np.log2(L)))
        # 为了避免边界严重失真，通常保留一些余量，比如取 min(8, log2(L))
        # 这里的 8 意味着最大能检测 2^8 = 256 的周期，对于 Traffic/ETT 足够
        decompose_level = min(8, max_level) 

        # 2. 序列补齐 (Padding) - SWT/MODWT 需要特定长度
        padded_data, original_len = self._pad_sequence(data_series, decompose_level)

        # 3. 执行 MODWT (利用 PyWavelets 的 swt)
        # coeffs 结构: [(cA_n, cD_n), ..., (cA_1, cD_1)]
        coeffs = pywt.swt(padded_data, self.wavelet_name, level=decompose_level, trim_approx=True)
        
        # 4. 计算小波能量谱 (Wavelet Variance)
        variances = []
        periods = []
        
        # coeffs 列表是从深层(低频)到浅层(高频)排列的
        # level j 对应 coeffs[-(j)] 的细节系数 cD
        # 我们按照 level 1, 2, ..., J 的顺序遍历
        for j in range(1, decompose_level + 1):
            # 获取第 j 层的细节系数 (Detail Coefficients)
            # swt 返回的是 [(cA_J, cD_J), (cA_J-1, cD_J-1), ..., (cA_1, cD_1)]
            # 所以 level 1 对应索引 -1
            cD_j = coeffs[-j][1] 
            
            # 截断回原始长度，去除 Padding 部分
            cD_j = cD_j[:original_len]
            
            # 计算无偏估计的方差 (能量)
            energy = np.var(cD_j)
            variances.append(energy)
            
            # 映射为周期 P = 2^j
            periods.append(2 ** j)

        # 5. Top-k 筛选
        # 将周期和能量组合，按能量降序排列
        period_energy_pairs = list(zip(periods, variances))
        sorted_pairs = sorted(period_energy_pairs, key=lambda x: x[1], reverse=True)
        
        top_k_periods = sorted_pairs[:self.top_k]
        
        # 简单打印日志
        print(f"[{input_name}] Extracted Top-{self.top_k} Periods: {[p for p, e in top_k_periods]}")
        
        return {
            "all_energies": variances,
            "all_periods": periods,
            "top_k": top_k_periods
        }

    def plot_spectrum(self, results, save_path=None):
        """
        可视化：绘制小波能量谱
        """
        periods = results['all_periods']
        energies = results['all_energies']
        top_k = [p for p, e in results['top_k']]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(periods)), energies, color='skyblue', edgecolor='navy')
        
        # 高亮 Top-k
        sorted_indices = np.argsort(energies)[::-1][:self.top_k]
        for idx in sorted_indices:
            bars[idx].set_color('orange')
            bars[idx].set_edgecolor('red')

        plt.xticks(range(len(periods)), [str(p) for p in periods])
        plt.xlabel('Period Length ($2^j$)')
        plt.ylabel('Wavelet Variance (Energy)')
        plt.title(f'Wavelet Energy Spectrum (Top Periods: {top_k})')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        plt.close()