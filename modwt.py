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

    def extract(self, data_series, input_name="Series", max_allowed_period=None):
            """
            执行 MODWT 并提取主导周期
            :param data_series: 一维 numpy 数组
            :param input_name: 数据集名称
            :param max_allowed_period: 【新增】最大允许的周期长度 (对应模型 Look-back Window)
            :return: 结果字典
            """
            L = len(data_series)
            max_level = int(np.floor(np.log2(L)))
            # 保证分解层数足够深，能覆盖到 max_allowed_period
            # 但也不要太深，通常 12 层足够覆盖 4096 (15分钟数据的月周期)
            decompose_level = min(12, max_level)

            # 序列补齐
            padded_data, original_len = self._pad_sequence(data_series, decompose_level)

            # MODWT 分解
            coeffs = pywt.swt(padded_data, self.wavelet_name, level=decompose_level)
            
            candidates = [] # 存储 (period, energy)
            
            # 遍历每一层
            for j in range(1, decompose_level + 1):
                period = 2 ** j
                
                # 【核心修改】：如果在计算能量前发现周期已经超标，直接跳过
                # 这样趋势项的能量再大，也不会进入 candidates 列表
                if max_allowed_period is not None and period > max_allowed_period:
                    continue

                # 获取细节系数 cD
                cD_j = coeffs[-j][1] 
                cD_j = cD_j[:original_len]
                energy = np.var(cD_j)
                
                candidates.append((period, energy))
                
            # Top-k 筛选
            # 现在 candidates 里全是合规的周期，按能量排序即可
            sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            top_k_periods = sorted_candidates[:self.top_k]
            
            # 提取用于绘图的全谱数据 (为了画图好看，我们还是把所有的存下来，但在 top_k 里不显示)
            all_periods = [2**j for j in range(1, decompose_level+1)]
            # 这里为了简单，只存合规的能量，不合规的补0或者重算一遍
            # 简单起见，我们重新遍历一遍只为画图 (画图可以画出趋势，但选只能选局部)
            # 但为了逻辑简单，这里只返回 candidates 里的数据用于画图也行
            # 或者为了严谨，我们仅返回合规的用于 Top-K，画图数据可以在外部不管
            
            print(f"[{input_name}] Constraints(<= {max_allowed_period}): Top-{self.top_k} -> {[p for p, e in top_k_periods]}")
            
            return {
                # 画图用的数据，我们可以简单造一下，只包含合规的，或者你保留上面的逻辑
                "all_periods": [p for p,e in candidates], 
                "all_energies": [e for p,e in candidates],
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