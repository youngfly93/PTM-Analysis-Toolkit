#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肽段分组分析模块 - 修改版

根据修饰检测情况将肽段分为三组进行比较分析：
- PTM-only: 只检测到特定修饰类型的肽段（保持原逻辑）
- PTM + WT: 同序列同时检测到任意修饰类型与未修饰肽（新逻辑）
- WT-only: 只检测到未修饰肽，从未检测到任何修饰（新逻辑）

作者: Tumor Proteomics Analysis Team
修改版本: 基于新的分组逻辑
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal
import warnings
warnings.filterwarnings('ignore')

# 尝试导入统计检验包
try:
    from scipy.stats import dunn
    DUNN_AVAILABLE = True
except ImportError:
    try:
        import scikit_posthocs as sp
        DUNN_AVAILABLE = True
        USE_SCIKIT_POSTHOCS = True
    except ImportError:
        DUNN_AVAILABLE = False
        USE_SCIKIT_POSTHOCS = False

# 尝试导入进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 导入核心分析器
from tumor_analysis_core import TumorAnalysisCore

class PeptideGroupAnalyzer:
    """肽段分组分析器 - 修改版"""
    
    def __init__(self, enable_resume=True):
        self.core_analyzer = TumorAnalysisCore()

        # 目标修饰类型（排除氧化修饰）
        self.target_modifications = [
            'Phospho[S]', 'Phospho[T]', 'Phospho[Y]',
            'Acetyl[K]', 'Methyl[K]', 'Dimethyl[K]', 'Trimethyl[K]',
            'Deamidated[N]', 'Deamidated[Q]'
        ]

        # 理化性质列名
        self.property_columns = [
            'corrected_pi', 'corrected_charge_at_ph7',
            'corrected_hydrophobicity_kd', 'corrected_molecular_weight'
        ]

        self.property_names = ['pI', 'Net_Charge', 'Hydrophobicity_KD', 'Molecular_Weight']

        # 输出目录
        self.output_dir = 'peptide_group_analysis_results_modified'
        self.fig_dir = os.path.join(self.output_dir, 'fig')
        self.cache_dir = os.path.join(self.output_dir, 'cache')

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # 断点续传设置
        self.enable_resume = enable_resume
        self.progress_file = os.path.join(self.cache_dir, 'analysis_progress.json')
        self.completed_datasets = self.load_progress() if enable_resume else set()

    def load_progress(self) -> set:
        """加载分析进度"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    import json
                    progress_data = json.load(f)
                    completed = set(progress_data.get('completed_datasets', []))
                    print(f"[LIST] 加载断点续传进度: 已完成 {len(completed)} 个数据集")
                    if completed:
                        print(f"  已完成数据集: {', '.join(sorted(completed))}")
                    return completed
        except Exception as e:
            print(f"[WARNING]  加载进度文件失败: {e}")
        return set()

    def save_progress(self, completed_dataset: str = None):
        """保存分析进度"""
        try:
            if completed_dataset:
                self.completed_datasets.add(completed_dataset)

            import json
            progress_data = {
                'completed_datasets': list(self.completed_datasets),
                'last_updated': pd.Timestamp.now().isoformat(),
                'total_completed': len(self.completed_datasets)
            }

            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)

            if completed_dataset:
                print(f"[SAVE] 保存进度: {completed_dataset} 已完成 ({len(self.completed_datasets)} 个数据集)")

        except Exception as e:
            print(f"[WARNING]  保存进度失败: {e}")

    def is_dataset_completed(self, dataset_id: str) -> bool:
        """检查数据集是否已完成分析"""
        return self.enable_resume and dataset_id in self.completed_datasets

    def load_cached_results(self, dataset_id: str) -> tuple:
        """加载缓存的分析结果"""
        try:
            cache_file = os.path.join(self.cache_dir, f'{dataset_id}_results.pkl')
            if os.path.exists(cache_file):
                import pickle
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
                    print(f"[LOAD] 加载缓存结果: {dataset_id}")
                    return cached_results.get('group_stats'), cached_results.get('stat_tests')
        except Exception as e:
            print(f"[WARNING]  加载缓存失败 {dataset_id}: {e}")
        return None, None

    def save_cached_results(self, dataset_id: str, group_stats: pd.DataFrame, stat_tests: pd.DataFrame):
        """保存分析结果到缓存"""
        try:
            cache_file = os.path.join(self.cache_dir, f'{dataset_id}_results.pkl')
            import pickle
            cached_results = {
                'group_stats': group_stats,
                'stat_tests': stat_tests,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_results, f)
            print(f"[SAVE] 缓存结果: {dataset_id}")
        except Exception as e:
            print(f"[WARNING]  缓存保存失败 {dataset_id}: {e}")

    def clear_cache(self):
        """清除所有缓存"""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                self.completed_datasets = set()
                print("[CLEAR]  已清除所有缓存")
        except Exception as e:
            print(f"[WARNING]  清除缓存失败: {e}")

    def classify_peptides_by_detection_per_modification(self, dataset_data: pd.DataFrame,
                                                       modification_type: str) -> pd.DataFrame:
        """
        针对特定修饰类型，根据检测情况对肽段进行分组
        
        新的分组逻辑：
        - PTM-only: 只检测到特定修饰类型的肽段（保持原逻辑）
        - PTM + WT: 同序列同时检测到任意修饰类型与未修饰肽（新逻辑）
        - WT-only: 只检测到未修饰肽，从未检测到任何修饰（新逻辑）

        Args:
            dataset_data: 数据集数据
            modification_type: 特定修饰类型，如'Acetyl[K]'

        Returns:
            DataFrame with additional 'peptide_group' column
        """
        print(f"[SEARCH] 分析 {modification_type} 的肽段检测情况（使用新的分组逻辑）...")

        # 先获取整个数据集中每个序列的完整检测情况
        all_sequence_stats = dataset_data.groupby('sequence').agg({
            'num_modifications': ['min', 'max', 'count'],
            'modification_type': lambda x: list(x.unique())
        }).reset_index()

        # 扁平化列名
        all_sequence_stats.columns = ['sequence', 'min_mods', 'max_mods', 'total_count', 'all_mod_types']

        # 新的分类逻辑
        def classify_sequence_new_logic(row):
            min_mods = row['min_mods']
            max_mods = row['max_mods']
            all_mod_types = row['all_mod_types']

            # 移除NaN值
            all_mod_types = [x for x in all_mod_types if pd.notna(x) and x != '']

            # 检查是否包含目标修饰
            has_target_mod = modification_type in all_mod_types
            # 检查是否有未修饰形态
            has_unmodified = None in all_mod_types or '' in all_mod_types or min_mods == 0
            # 检查是否有任何修饰（不仅仅是目标修饰）
            has_any_modification = any(mod_type in self.target_modifications for mod_type in all_mod_types)

            if has_target_mod and not has_unmodified:
                return 'PTM-only'  # 只检测到特定修饰类型
            elif has_any_modification and has_unmodified:
                return 'PTM + WT'  # 既有修饰（任意类型）又有未修饰
            elif not has_any_modification and has_unmodified:
                return 'WT-only'   # 只有未修饰肽
            else:
                return 'Unknown'

        all_sequence_stats['peptide_group'] = all_sequence_stats.apply(classify_sequence_new_logic, axis=1)

        # 只保留该特定修饰和无修饰肽段，以及PTM + WT组的所有肽段
        target_data = dataset_data.copy()
        
        # 合并分组信息
        target_data = target_data.merge(
            all_sequence_stats[['sequence', 'peptide_group']],
            on='sequence',
            how='left'
        )

        # 根据新逻辑筛选数据
        # PTM-only: 只保留特定修饰类型
        # PTM + WT: 保留所有肽段（修饰和未修饰）
        # WT-only: 只保留未修饰肽段
        filtered_data = []
        
        for group in ['PTM-only', 'PTM + WT', 'WT-only']:
            group_sequences = all_sequence_stats[all_sequence_stats['peptide_group'] == group]['sequence'].tolist()
            
            if group == 'PTM-only':
                # 只保留特定修饰类型的肽段
                group_data = target_data[
                    (target_data['sequence'].isin(group_sequences)) &
                    (target_data['modification_type'] == modification_type)
                ]
            elif group == 'PTM + WT':
                # 保留所有肽段（修饰和未修饰）
                group_data = target_data[target_data['sequence'].isin(group_sequences)]
            elif group == 'WT-only':
                # 只保留未修饰肽段
                group_data = target_data[
                    (target_data['sequence'].isin(group_sequences)) &
                    (target_data['num_modifications'] == 0)
                ]
            
            if len(group_data) > 0:
                filtered_data.append(group_data)

        if len(filtered_data) > 0:
            target_data = pd.concat(filtered_data, ignore_index=True)
        else:
            print(f"[WARNING]  没有找到 {modification_type} 的有效数据")
            return pd.DataFrame()

        # 统计各组数量
        group_counts = target_data['peptide_group'].value_counts()
        print(f"[STATS] {modification_type} 肽段分组统计（新逻辑）:")
        for group, count in group_counts.items():
            print(f"  {group}: {count:,} 条记录")
            
        # 打印详细的分组信息
        print(f"\n[LIST] 分组详情:")
        print(f"  - PTM-only: 只有{modification_type}修饰的序列")
        print(f"  - PTM + WT: 同时有任意修饰和未修饰形态的序列（包含所有肽段）")
        print(f"  - WT-only: 只有未修饰形态的序列")

        return target_data

    def calculate_group_statistics_per_modification(self, grouped_data: pd.DataFrame,
                                                   dataset_id: str, modification_type: str) -> pd.DataFrame:
        """计算特定修饰类型各组的统计量"""

        stats_list = []

        for group in ['PTM-only', 'PTM + WT', 'WT-only']:
            group_data = grouped_data[grouped_data['peptide_group'] == group]

            if len(group_data) == 0:
                continue

            for i, prop_col in enumerate(self.property_columns):
                prop_name = self.property_names[i]
                values = group_data[prop_col].dropna()

                if len(values) > 0:
                    stats_list.append({
                        'dataset_id': dataset_id,
                        'modification_type': modification_type,
                        'peptide_group': group,
                        'property': prop_name,
                        'count': len(values),
                        'median': values.median(),
                        'mean': values.mean(),
                        'std': values.std(),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75),
                        'min': values.min(),
                        'max': values.max()
                    })

        return pd.DataFrame(stats_list)

    def calculate_group_statistics(self, grouped_data: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
        """计算各组的统计量"""
        
        stats_list = []
        
        for group in ['PTM-only', 'PTM + WT', 'WT-only']:
            group_data = grouped_data[grouped_data['peptide_group'] == group]
            
            if len(group_data) == 0:
                continue
            
            for i, prop_col in enumerate(self.property_columns):
                prop_name = self.property_names[i]
                values = group_data[prop_col].dropna()
                
                if len(values) > 0:
                    stats_list.append({
                        'dataset_id': dataset_id,
                        'peptide_group': group,
                        'property': prop_name,
                        'count': len(values),
                        'median': values.median(),
                        'mean': values.mean(),
                        'std': values.std(),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75),
                        'min': values.min(),
                        'max': values.max()
                    })
        
        return pd.DataFrame(stats_list)

    def perform_statistical_tests_per_modification(self, grouped_data: pd.DataFrame,
                                                  dataset_id: str, modification_type: str) -> pd.DataFrame:
        """执行特定修饰类型的统计检验"""

        test_results = []

        for i, prop_col in enumerate(self.property_columns):
            prop_name = self.property_names[i]

            # 准备三组数据
            groups = {}
            for group_name in ['PTM-only', 'PTM + WT', 'WT-only']:
                group_data = grouped_data[grouped_data['peptide_group'] == group_name]
                values = group_data[prop_col].dropna()
                if len(values) >= 3:  # 至少需要3个值
                    groups[group_name] = values

            if len(groups) < 2:
                continue

            # Kruskal-Wallis检验
            group_values = list(groups.values())
            if len(group_values) >= 2:
                try:
                    h_stat, p_value = kruskal(*group_values)

                    test_results.append({
                        'dataset_id': dataset_id,
                        'modification_type': modification_type,
                        'property': prop_name,
                        'test_type': 'Kruskal-Wallis',
                        'comparison': 'Overall',
                        'statistic': h_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': np.nan,
                        'group1': 'All',
                        'group2': 'All'
                    })

                    # 如果总体检验显著，进行事后检验
                    if p_value < 0.05 and len(groups) >= 2:
                        group_names = list(groups.keys())

                        # 两两比较
                        for j in range(len(group_names)):
                            for k in range(j+1, len(group_names)):
                                group1_name = group_names[j]
                                group2_name = group_names[k]
                                group1_values = groups[group1_name]
                                group2_values = groups[group2_name]

                                # Mann-Whitney U检验
                                try:
                                    u_stat, u_p = stats.mannwhitneyu(
                                        group1_values, group2_values,
                                        alternative='two-sided'
                                    )

                                    # 计算效应量 (Cliff's delta approximation)
                                    n1, n2 = len(group1_values), len(group2_values)
                                    effect_size = (u_stat / (n1 * n2)) * 2 - 1

                                    test_results.append({
                                        'dataset_id': dataset_id,
                                        'modification_type': modification_type,
                                        'property': prop_name,
                                        'test_type': 'Mann-Whitney U',
                                        'comparison': f'{group1_name} vs {group2_name}',
                                        'statistic': u_stat,
                                        'p_value': u_p,
                                        'significant': u_p < 0.05,
                                        'effect_size': effect_size,
                                        'group1': group1_name,
                                        'group2': group2_name
                                    })
                                except Exception as e:
                                    print(f"[WARNING]  两两比较失败 {group1_name} vs {group2_name}: {e}")
                                    continue

                except Exception as e:
                    print(f"[WARNING]  Kruskal-Wallis检验失败 {prop_name}: {e}")
                    continue

        return pd.DataFrame(test_results)

    def perform_statistical_tests(self, grouped_data: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
        """执行统计检验"""
        
        test_results = []
        
        for i, prop_col in enumerate(self.property_columns):
            prop_name = self.property_names[i]
            
            # 准备三组数据
            groups = {}
            for group_name in ['PTM-only', 'PTM + WT', 'WT-only']:
                group_data = grouped_data[grouped_data['peptide_group'] == group_name]
                values = group_data[prop_col].dropna()
                if len(values) >= 3:  # 至少需要3个值
                    groups[group_name] = values
            
            if len(groups) < 2:
                continue
            
            # Kruskal-Wallis检验
            group_values = list(groups.values())
            if len(group_values) >= 2:
                try:
                    h_stat, p_value = kruskal(*group_values)
                    
                    test_results.append({
                        'dataset_id': dataset_id,
                        'property': prop_name,
                        'test_type': 'Kruskal-Wallis',
                        'comparison': 'Overall',
                        'statistic': h_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': np.nan,
                        'group1': 'All',
                        'group2': 'All'
                    })
                    
                    # 如果总体检验显著，进行事后检验
                    if p_value < 0.05 and len(groups) >= 2:
                        group_names = list(groups.keys())
                        
                        # 两两比较
                        for j in range(len(group_names)):
                            for k in range(j+1, len(group_names)):
                                group1_name = group_names[j]
                                group2_name = group_names[k]
                                group1_values = groups[group1_name]
                                group2_values = groups[group2_name]
                                
                                # Mann-Whitney U检验
                                try:
                                    u_stat, u_p = stats.mannwhitneyu(
                                        group1_values, group2_values, 
                                        alternative='two-sided'
                                    )
                                    
                                    # 计算效应量 (Cliff's delta approximation)
                                    n1, n2 = len(group1_values), len(group2_values)
                                    effect_size = (u_stat / (n1 * n2)) * 2 - 1
                                    
                                    test_results.append({
                                        'dataset_id': dataset_id,
                                        'property': prop_name,
                                        'test_type': 'Mann-Whitney U',
                                        'comparison': f'{group1_name} vs {group2_name}',
                                        'statistic': u_stat,
                                        'p_value': u_p,
                                        'significant': u_p < 0.05,
                                        'effect_size': effect_size,
                                        'group1': group1_name,
                                        'group2': group2_name
                                    })
                                except Exception as e:
                                    print(f"[WARNING]  两两比较失败 {group1_name} vs {group2_name}: {e}")
                                    continue
                
                except Exception as e:
                    print(f"[WARNING]  Kruskal-Wallis检验失败 {prop_name}: {e}")
                    continue
        
        return pd.DataFrame(test_results)
    
    def create_box_violin_plots(self, grouped_data: pd.DataFrame, dataset_id: str):
        """创建合并的箱线图/小提琴图（4个理化性质在一张图上）"""

        print(f"[STATS] 绘制 {dataset_id} 的合并可视化图表...")

        # 设置图形样式
        plt.style.use('default')
        sns.set_palette("Set2")

        # 创建2x2子图布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{dataset_id} - All Modifications - Physicochemical Properties by Peptide Group (Modified Logic)',
                    fontsize=16, fontweight='bold')

        axes = axes.flatten()

        # 为每个理化性质创建子图
        for i, prop_col in enumerate(self.property_columns):
            prop_name = self.property_names[i]
            ax = axes[i]

            # 准备数据
            plot_data = []
            for group in ['PTM-only', 'PTM + WT', 'WT-only']:
                group_data = grouped_data[grouped_data['peptide_group'] == group]
                values = group_data[prop_col].dropna()

                if len(values) > 0:
                    for value in values:
                        plot_data.append({
                            'Group': group,
                            'Value': value,
                            'Property': prop_name
                        })

            if len(plot_data) == 0:
                ax.text(0.5, 0.5, f'No data for {prop_name}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{prop_name}')
                continue

            plot_df = pd.DataFrame(plot_data)

            # 准备分组数据用于统计检验
            groups_data = {}
            for group in ['PTM-only', 'PTM + WT', 'WT-only']:
                group_data = grouped_data[grouped_data['peptide_group'] == group]
                values = group_data[prop_col].dropna()
                if len(values) > 0:
                    groups_data[group] = values

            # 绘制更窄的小提琴图 + 箱线图
            sns.violinplot(data=plot_df, x='Group', y='Value', ax=ax,
                          alpha=0.6, width=0.6, inner=None)  # 更窄的小提琴图
            sns.boxplot(data=plot_df, x='Group', y='Value', ax=ax,
                       width=0.2, boxprops=dict(alpha=0.9),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))  # 更窄的箱线图

            # 设置标题和标签
            ax.set_title(f'{prop_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Peptide Group', fontsize=12)
            ax.set_ylabel(f'{prop_name}', fontsize=12)

            # 执行统计检验并添加显著性标注
            self.add_significance_annotations(ax, groups_data, plot_df)

            # 添加统计信息
            group_stats = plot_df.groupby('Group')['Value'].agg(['count', 'median']).round(3)
            stats_text = '\n'.join([f'{group}: n={row["count"]}, median={row["median"]}'
                                   for group, row in group_stats.iterrows()])
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                   verticalalignment='bottom', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # 旋转x轴标签
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        filename = f'combined_properties_{dataset_id}_all_modifications_modified.png'
        filepath = os.path.join(self.fig_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] 保存: {filename}")

    def create_box_violin_plots_per_modification(self, grouped_data: pd.DataFrame,
                                                dataset_id: str, modification_type: str):
        """为特定修饰类型创建合并的箱线图/小提琴图（4个理化性质在一张图上）"""

        print(f"[STATS] 绘制 {dataset_id} - {modification_type} 的合并可视化图表...")

        # 设置图形样式
        plt.style.use('default')
        sns.set_palette("Set2")

        # 创建2x2子图布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{dataset_id} - {modification_type} - Physicochemical Properties by Peptide Group (Modified Logic)',
                    fontsize=16, fontweight='bold')

        axes = axes.flatten()

        # 为每个理化性质创建子图
        for i, prop_col in enumerate(self.property_columns):
            prop_name = self.property_names[i]
            ax = axes[i]

            # 准备数据
            plot_data = []
            groups_data = {}
            for group in ['PTM-only', 'PTM + WT', 'WT-only']:
                group_data = grouped_data[grouped_data['peptide_group'] == group]
                values = group_data[prop_col].dropna()

                if len(values) > 0:
                    groups_data[group] = values
                    for value in values:
                        plot_data.append({
                            'Group': group,
                            'Value': value,
                            'Property': prop_name
                        })

            if len(plot_data) == 0:
                ax.text(0.5, 0.5, f'No data for {prop_name}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{prop_name}')
                continue

            plot_df = pd.DataFrame(plot_data)

            # 绘制更窄的小提琴图 + 箱线图
            sns.violinplot(data=plot_df, x='Group', y='Value', ax=ax,
                          alpha=0.6, width=0.6, inner=None)  # 更窄的小提琴图
            sns.boxplot(data=plot_df, x='Group', y='Value', ax=ax,
                       width=0.2, boxprops=dict(alpha=0.9),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))  # 更窄的箱线图

            # 设置标题和标签
            ax.set_title(f'{prop_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Peptide Group', fontsize=12)
            ax.set_ylabel(f'{prop_name}', fontsize=12)

            # 执行统计检验并添加显著性标注
            self.add_significance_annotations(ax, groups_data, plot_df)

            # 添加统计信息
            group_stats = plot_df.groupby('Group')['Value'].agg(['count', 'median']).round(3)
            stats_text = '\n'.join([f'{group}: n={row["count"]}, median={row["median"]}'
                                   for group, row in group_stats.iterrows()])
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                   verticalalignment='bottom', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # 旋转x轴标签
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        # 清理修饰类型名称用于文件名
        clean_mod_name = modification_type.replace('[', '_').replace(']', '_').replace('/', '_')
        filename = f'combined_properties_{dataset_id}_{clean_mod_name}_modified.png'
        filepath = os.path.join(self.fig_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [OK] 保存: {filename}")

    def add_significance_annotations(self, ax, groups_data: dict, plot_df: pd.DataFrame):
        """添加显著性检验的p值标注"""

        try:
            from scipy import stats

            # 获取组名和对应的x轴位置
            group_names = ['PTM-only', 'PTM + WT', 'WT-only']
            available_groups = [g for g in group_names if g in groups_data and len(groups_data[g]) >= 3]

            if len(available_groups) < 2:
                return

            # 执行Kruskal-Wallis检验
            group_values = [groups_data[g] for g in available_groups]
            try:
                h_stat, kw_p = stats.kruskal(*group_values)

                # 在图表顶部添加总体检验结果
                y_max = plot_df['Value'].max()
                y_range = plot_df['Value'].max() - plot_df['Value'].min()

                # Kruskal-Wallis结果
                kw_text = f'Kruskal-Wallis: p={kw_p:.2e}' if kw_p < 0.001 else f'Kruskal-Wallis: p={kw_p:.3f}'
                if kw_p < 0.001:
                    kw_text += ' ***'
                elif kw_p < 0.01:
                    kw_text += ' **'
                elif kw_p < 0.05:
                    kw_text += ' *'
                else:
                    kw_text += ' ns'

                ax.text(0.5, 0.98, kw_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

                # 如果总体检验显著，进行两两比较
                if kw_p < 0.05 and len(available_groups) >= 2:
                    y_offset = y_max + y_range * 0.05

                    # 两两比较
                    comparisons = []
                    for j in range(len(available_groups)):
                        for k in range(j+1, len(available_groups)):
                            group1 = available_groups[j]
                            group2 = available_groups[k]

                            try:
                                u_stat, u_p = stats.mannwhitneyu(
                                    groups_data[group1], groups_data[group2],
                                    alternative='two-sided'
                                )

                                # 获取x轴位置
                                x1 = group_names.index(group1)
                                x2 = group_names.index(group2)

                                comparisons.append({
                                    'x1': x1, 'x2': x2, 'p': u_p,
                                    'group1': group1, 'group2': group2
                                })

                            except Exception as e:
                                continue

                    # 绘制显著性标注
                    for idx, comp in enumerate(comparisons):
                        if comp['p'] < 0.05:  # 只标注显著的比较
                            x1, x2 = comp['x1'], comp['x2']
                            p_val = comp['p']

                            # 确定显著性符号
                            if p_val < 0.001:
                                sig_symbol = '***'
                            elif p_val < 0.01:
                                sig_symbol = '**'
                            elif p_val < 0.05:
                                sig_symbol = '*'
                            else:
                                continue

                            # 计算标注位置
                            y_pos = y_offset + idx * y_range * 0.08

                            # 绘制连接线
                            ax.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=1)
                            ax.plot([x1, x1], [y_pos - y_range*0.01, y_pos], 'k-', linewidth=1)
                            ax.plot([x2, x2], [y_pos - y_range*0.01, y_pos], 'k-', linewidth=1)

                            # 添加p值和显著性符号
                            p_text = f'{sig_symbol}\np={p_val:.2e}' if p_val < 0.001 else f'{sig_symbol}\np={p_val:.3f}'
                            ax.text((x1 + x2) / 2, y_pos + y_range*0.02, p_text,
                                   ha='center', va='bottom', fontsize=8, fontweight='bold')

            except Exception as e:
                print(f"[WARNING]  统计检验失败: {e}")

        except ImportError:
            print("[WARNING]  scipy未安装，跳过显著性标注")
        except Exception as e:
            print(f"[WARNING]  添加显著性标注失败: {e}")

    def analyze_single_dataset(self, dataset_id: str) -> tuple:
        """分析单个数据集 - 对每种修饰类型分别进行三组比较"""

        print(f"\n[DATASET] 分析数据集: {dataset_id}")

        # 检查是否已完成分析
        if self.is_dataset_completed(dataset_id):
            print(f"[OK] 数据集 {dataset_id} 已完成分析，从缓存加载结果")
            cached_group_stats, cached_stat_tests = self.load_cached_results(dataset_id)
            if cached_group_stats is not None and cached_stat_tests is not None:
                return None, cached_group_stats, cached_stat_tests
            else:
                print(f"[WARNING]  缓存文件损坏，重新分析 {dataset_id}")
                # 从已完成列表中移除，重新分析
                self.completed_datasets.discard(dataset_id)

        try:
            # 获取肿瘤样本数据
            tumor_files = self.core_analyzer.get_tumor_samples(dataset_id)

            if len(tumor_files) == 0:
                print(f"[WARNING]  数据集 {dataset_id}: 无肿瘤样本")
                return None, None, None

            print(f"[FILES] 找到 {len(tumor_files)} 个肿瘤样本文件")

            # 收集所有文件的数据
            all_data = []

            for file_name in tumor_files:
                spectra_file = self.core_analyzer.find_spectra_file(dataset_id, file_name)

                if spectra_file is None:
                    continue

                file_data = self.core_analyzer.analyze_single_file(spectra_file, dataset_id)
                if file_data is not None and len(file_data) > 0:
                    all_data.append(file_data)

            if len(all_data) == 0:
                print(f"[ERROR] 数据集 {dataset_id}: 无有效数据")
                return None, None, None

            # 合并数据
            dataset_data = pd.concat(all_data, ignore_index=True)
            print(f"[OK] 合并得到 {len(dataset_data)} 条记录")

            # 对每种修饰类型分别进行分析
            all_group_stats = []
            all_stat_tests = []

            for modification_type in self.target_modifications:
                print(f"\n[TARGET] 分析修饰类型: {modification_type}")

                # 针对该修饰类型进行肽段分组（使用新逻辑）
                grouped_data = self.classify_peptides_by_detection_per_modification(
                    dataset_data, modification_type)

                if len(grouped_data) == 0:
                    print(f"[WARNING]  {modification_type}: 无有效分组数据")
                    continue

                # 计算统计量
                group_stats = self.calculate_group_statistics_per_modification(
                    grouped_data, dataset_id, modification_type)

                if len(group_stats) > 0:
                    all_group_stats.append(group_stats)

                # 统计检验
                stat_tests = self.perform_statistical_tests_per_modification(
                    grouped_data, dataset_id, modification_type)

                if len(stat_tests) > 0:
                    all_stat_tests.append(stat_tests)

                # 创建可视化
                self.create_box_violin_plots_per_modification(
                    grouped_data, dataset_id, modification_type)

            # 合并所有修饰类型的结果
            if len(all_group_stats) > 0:
                final_group_stats = pd.concat(all_group_stats, ignore_index=True)
            else:
                final_group_stats = pd.DataFrame()

            if len(all_stat_tests) > 0:
                final_stat_tests = pd.concat(all_stat_tests, ignore_index=True)
            else:
                final_stat_tests = pd.DataFrame()

            # 保存结果到缓存并标记为已完成
            if len(final_group_stats) > 0:
                self.save_cached_results(dataset_id, final_group_stats, final_stat_tests)
                self.save_progress(dataset_id)
                print(f"[OK] 数据集 {dataset_id} 分析完成并已缓存")

            return dataset_data, final_group_stats, final_stat_tests

        except Exception as e:
            print(f"[ERROR] 分析数据集 {dataset_id} 时出错: {e}")
            return None, None, None

    def run_full_analysis(self, max_datasets: int = None):
        """运行完整的分组分析"""

        print("开始肽段分组分析（修改版逻辑）...")
        print("=" * 60)
        print("新的分组逻辑说明:")
        print("- PTM-only: 只检测到特定修饰类型的肽段（保持原逻辑）")
        print("- PTM + WT: 同序列同时检测到任意修饰类型与未修饰肽（新逻辑）")
        print("- WT-only: 只检测到未修饰肽，从未检测到任何修饰（新逻辑）")
        print("=" * 60)

        # 获取癌症数据集
        cancer_datasets = self.core_analyzer.load_cancer_datasets()

        if len(cancer_datasets) == 0:
            print("[ERROR] 未找到癌症数据集")
            return

        if max_datasets:
            cancer_datasets = cancer_datasets[:max_datasets]
            print(f"[TARGET] 限制分析前 {max_datasets} 个数据集")

        print(f"[LIST] 总共需要分析 {len(cancer_datasets)} 个数据集")

        # 显示断点续传状态
        if self.enable_resume:
            remaining_datasets = [ds for ds in cancer_datasets if not self.is_dataset_completed(ds)]
            completed_count = len(cancer_datasets) - len(remaining_datasets)
            print(f"[RESUME] 断点续传: 已完成 {completed_count} 个，剩余 {len(remaining_datasets)} 个数据集")
            cancer_datasets = remaining_datasets

        if len(cancer_datasets) == 0:
            print("[OK] 所有数据集都已完成分析，加载缓存结果...")
            # 加载所有缓存结果
            all_group_stats = []
            all_stat_tests = []
            successful_datasets = []

            for dataset_id in self.completed_datasets:
                cached_group_stats, cached_stat_tests = self.load_cached_results(dataset_id)
                if cached_group_stats is not None:
                    all_group_stats.append(cached_group_stats)
                    successful_datasets.append(dataset_id)
                    if cached_stat_tests is not None:
                        all_stat_tests.append(cached_stat_tests)
        else:
            # 存储所有结果
            all_group_stats = []
            all_stat_tests = []
            successful_datasets = []

            # 先加载已完成的缓存结果
            for dataset_id in self.completed_datasets:
                cached_group_stats, cached_stat_tests = self.load_cached_results(dataset_id)
                if cached_group_stats is not None:
                    all_group_stats.append(cached_group_stats)
                    successful_datasets.append(dataset_id)
                    if cached_stat_tests is not None:
                        all_stat_tests.append(cached_stat_tests)

            # 分析剩余数据集
            if TQDM_AVAILABLE:
                dataset_iterator = tqdm(cancer_datasets, desc="分析数据集", unit="数据集")
            else:
                dataset_iterator = cancer_datasets

            for dataset_id in dataset_iterator:
                if not TQDM_AVAILABLE:
                    print(f"\n[STATS] 处理数据集: {dataset_id}")

                grouped_data, group_stats, stat_tests = self.analyze_single_dataset(dataset_id)

                if group_stats is not None and len(group_stats) > 0:
                    all_group_stats.append(group_stats)
                    successful_datasets.append(dataset_id)

                    if stat_tests is not None and len(stat_tests) > 0:
                        all_stat_tests.append(stat_tests)

        if len(all_group_stats) == 0:
            print("[ERROR] 没有成功分析的数据集")
            return

        # 合并所有结果
        print(f"\n[MERGE] 合并分析结果...")
        final_group_stats = pd.concat(all_group_stats, ignore_index=True)

        if len(all_stat_tests) > 0:
            final_stat_tests = pd.concat(all_stat_tests, ignore_index=True)
        else:
            final_stat_tests = pd.DataFrame()

        # 保存结果
        print(f"[SAVE] 保存结果...")

        group_stats_file = os.path.join(self.output_dir, 'group_stats_modified.csv')
        final_group_stats.to_csv(group_stats_file, index=False, encoding='utf-8')
        print(f"  [OK] 保存: group_stats_modified.csv")

        if len(final_stat_tests) > 0:
            stat_tests_file = os.path.join(self.output_dir, 'stat_tests_modified.csv')
            final_stat_tests.to_csv(stat_tests_file, index=False, encoding='utf-8')
            print(f"  [OK] 保存: stat_tests_modified.csv")

        # 创建全局热图
        self.create_global_heatmap(final_group_stats)

        # 创建数据集级别趋势图
        self.create_dataset_trend_plots(final_group_stats)

        # 生成分析报告
        self.generate_analysis_report(final_group_stats, final_stat_tests, successful_datasets)

        print(f"\n[SUCCESS] 分析完成！")
        print(f"[OK] 成功分析了 {len(successful_datasets)} 个数据集")
        print(f"[FILES] 结果保存在: {self.output_dir}/")
        print(f"[STATS] 图表保存在: {self.fig_dir}/")

        return final_group_stats, final_stat_tests

    def create_global_heatmap(self, group_stats_df: pd.DataFrame):
        """创建全局热图"""

        print("[HEATMAP] 创建全局热图...")

        try:
            # 计算相对于WT-only的差异
            heatmap_data = []

            for dataset_id in group_stats_df['dataset_id'].unique():
                dataset_data = group_stats_df[group_stats_df['dataset_id'] == dataset_id]

                for prop in self.property_names:
                    prop_data = dataset_data[dataset_data['property'] == prop]

                    # 获取WT-only作为基准
                    wt_only = prop_data[prop_data['peptide_group'] == 'WT-only']
                    if len(wt_only) == 0:
                        continue

                    wt_median = wt_only['median'].iloc[0]

                    # 计算其他组相对于WT-only的差异
                    for group in ['PTM-only', 'PTM + WT']:
                        group_data = prop_data[prop_data['peptide_group'] == group]
                        if len(group_data) > 0:
                            group_median = group_data['median'].iloc[0]
                            delta = group_median - wt_median

                            heatmap_data.append({
                                'dataset_id': dataset_id,
                                'property': prop,
                                'group': group,
                                'delta': delta
                            })

            if len(heatmap_data) == 0:
                print("[WARNING]  没有足够的数据创建热图")
                return

            heatmap_df = pd.DataFrame(heatmap_data)

            # 为每个理化性质创建热图
            for prop in self.property_names:
                prop_data = heatmap_df[heatmap_df['property'] == prop]

                if len(prop_data) == 0:
                    continue

                # 创建数据透视表
                pivot_data = prop_data.pivot(index='dataset_id', columns='group', values='delta')

                if pivot_data.empty:
                    continue

                # 创建热图
                fig, ax = plt.subplots(1, 1, figsize=(8, max(12, len(pivot_data) * 0.3)))

                # 选择合适的颜色映射
                if prop in ['pI', 'Net_Charge']:
                    cmap = 'RdBu_r'
                    center = 0
                elif prop == 'Hydrophobicity_KD':
                    cmap = 'RdYlBu_r'
                    center = 0
                else:  # Molecular_Weight
                    cmap = 'viridis'
                    center = None

                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap, center=center,
                           ax=ax, cbar_kws={'label': f'Δ{prop} (vs WT-only)'})

                ax.set_title(f'Global Heatmap: Δ{prop} Across Datasets (Modified Logic)',
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Peptide Group', fontsize=12)
                ax.set_ylabel('Dataset ID', fontsize=12)

                plt.tight_layout()

                # 保存图片
                filename = f'global_heatmap_{prop}_modified.png'
                filepath = os.path.join(self.fig_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"  [OK] 保存: {filename}")

            # 创建综合热图（所有性质在一张图上）
            self.create_comprehensive_heatmap(heatmap_df)

        except Exception as e:
            print(f"[ERROR] 创建热图失败: {e}")

    def create_comprehensive_heatmap(self, heatmap_df: pd.DataFrame):
        """创建综合热图（4个性质 × 2个组）"""

        try:
            # 重新组织数据：dataset × (property_group)
            comprehensive_data = []

            for dataset_id in heatmap_df['dataset_id'].unique():
                dataset_data = heatmap_df[heatmap_df['dataset_id'] == dataset_id]
                row_data = {'dataset_id': dataset_id}

                for prop in self.property_names:
                    prop_data = dataset_data[dataset_data['property'] == prop]

                    for group in ['PTM-only', 'PTM + WT']:
                        group_data = prop_data[prop_data['group'] == group]
                        col_name = f'{prop}_{group}'

                        if len(group_data) > 0:
                            row_data[col_name] = group_data['delta'].iloc[0]
                        else:
                            row_data[col_name] = np.nan

                comprehensive_data.append(row_data)

            comp_df = pd.DataFrame(comprehensive_data)
            comp_df = comp_df.set_index('dataset_id')

            # 移除全为NaN的列
            comp_df = comp_df.dropna(axis=1, how='all')

            if comp_df.empty:
                print("[WARNING]  没有足够的数据创建综合热图")
                return

            # 创建综合热图
            fig, ax = plt.subplots(1, 1, figsize=(12, max(10, len(comp_df) * 0.25)))

            sns.heatmap(comp_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                       ax=ax, cbar_kws={'label': 'Δ Value (vs WT-only)'})

            ax.set_title('Comprehensive Heatmap: All Properties Across Datasets (Modified Logic)',
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Property × Group', fontsize=12)
            ax.set_ylabel('Dataset ID', fontsize=12)

            # 旋转x轴标签
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # 保存图片
            filename = 'global_heatmap_comprehensive_modified.png'
            filepath = os.path.join(self.fig_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  [OK] 保存: {filename}")

        except Exception as e:
            print(f"[ERROR] 创建综合热图失败: {e}")

    def create_dataset_trend_plots(self, group_stats_df: pd.DataFrame):
        """创建数据集级别的趋势图"""

        print("[MERGE] 创建数据集级别趋势图...")

        try:
            # 获取所有修饰类型和数据集
            modifications = group_stats_df['modification_type'].unique()
            datasets = sorted(group_stats_df['dataset_id'].unique())

            if len(datasets) < 2:
                print("[WARNING]  数据集数量不足，无法创建趋势图")
                return

            # 为每种修饰类型创建趋势图
            for modification_type in modifications:
                print(f"  [STATS] 绘制 {modification_type} 的趋势图...")

                mod_data = group_stats_df[group_stats_df['modification_type'] == modification_type]

                if len(mod_data) == 0:
                    continue

                # 创建2x2子图布局
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'{modification_type} - Dataset-level Trends Across Properties (Modified Logic)',
                           fontsize=16, fontweight='bold')

                axes = axes.flatten()

                # 为每个理化性质创建趋势图
                for i, prop_name in enumerate(self.property_names):
                    ax = axes[i]

                    prop_data = mod_data[mod_data['property'] == prop_name]

                    if len(prop_data) == 0:
                        ax.text(0.5, 0.5, f'No data for {prop_name}',
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{prop_name}')
                        continue

                    # 准备趋势数据
                    trend_data = {}
                    for group in ['PTM-only', 'PTM + WT', 'WT-only']:
                        group_prop_data = prop_data[prop_data['peptide_group'] == group]

                        if len(group_prop_data) > 0:
                            # 按数据集排序
                            group_prop_data = group_prop_data.sort_values('dataset_id')
                            trend_data[group] = {
                                'datasets': group_prop_data['dataset_id'].tolist(),
                                'medians': group_prop_data['median'].tolist(),
                                'counts': group_prop_data['count'].tolist()
                            }

                    # 绘制趋势线
                    colors = {'PTM-only': '#e74c3c', 'PTM + WT': '#f39c12', 'WT-only': '#3498db'}
                    markers = {'PTM-only': 'o', 'PTM + WT': 's', 'WT-only': '^'}

                    for group, data in trend_data.items():
                        if len(data['datasets']) > 0:
                            # 创建x轴位置（数据集索引）
                            x_positions = [datasets.index(ds) for ds in data['datasets']]

                            # 绘制趋势线
                            ax.plot(x_positions, data['medians'],
                                   color=colors[group], marker=markers[group],
                                   linewidth=2, markersize=8, label=group, alpha=0.8)

                            # 添加数据点的计数信息（作为点的大小）
                            sizes = [min(max(count/100, 10), 200) for count in data['counts']]
                            ax.scatter(x_positions, data['medians'],
                                     s=sizes, color=colors[group], alpha=0.3, edgecolors='white')

                    # 设置图表属性
                    ax.set_title(f'{prop_name}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Dataset', fontsize=10)
                    ax.set_ylabel(f'{prop_name}', fontsize=10)
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)

                    # 设置x轴标签
                    ax.set_xticks(range(len(datasets)))
                    ax.set_xticklabels([ds.replace('PXD', '') for ds in datasets],
                                      rotation=45, ha='right', fontsize=8)

                    # 添加统计信息
                    info_text = f"Datasets: {len(set().union(*[data['datasets'] for data in trend_data.values()]))}"
                    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

                plt.tight_layout()

                # 保存图片
                clean_mod_name = modification_type.replace('[', '_').replace(']', '_').replace('/', '_')
                filename = f'dataset_trends_{clean_mod_name}_modified.png'
                filepath = os.path.join(self.fig_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"    [OK] 保存: {filename}")

            # 创建综合趋势图（所有修饰类型在一张图上）
            self.create_comprehensive_trend_plot(group_stats_df, datasets)

        except Exception as e:
            print(f"[ERROR] 创建趋势图失败: {e}")
            import traceback
            traceback.print_exc()

    def create_comprehensive_trend_plot(self, group_stats_df: pd.DataFrame, datasets: list):
        """创建综合趋势图（所有修饰类型）"""

        try:
            print("  [STATS] 绘制综合趋势图...")

            # 为每个理化性质创建一个综合图
            for prop_name in self.property_names:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f'All Modifications - {prop_name} Trends Across Datasets (Modified Logic)',
                           fontsize=16, fontweight='bold')

                groups = ['PTM-only', 'PTM + WT', 'WT-only']
                colors = plt.cm.Set3(np.linspace(0, 1, len(group_stats_df['modification_type'].unique())))

                for group_idx, group in enumerate(groups):
                    ax = axes[group_idx]
                    ax.set_title(f'{group}', fontsize=14, fontweight='bold')

                    # 为每种修饰类型绘制趋势线
                    for mod_idx, modification_type in enumerate(group_stats_df['modification_type'].unique()):
                        mod_group_data = group_stats_df[
                            (group_stats_df['modification_type'] == modification_type) &
                            (group_stats_df['peptide_group'] == group) &
                            (group_stats_df['property'] == prop_name)
                        ]

                        if len(mod_group_data) > 1:  # 至少需要2个数据点
                            # 按数据集排序
                            mod_group_data = mod_group_data.sort_values('dataset_id')

                            # 创建x轴位置
                            x_positions = [datasets.index(ds) for ds in mod_group_data['dataset_id']]
                            y_values = mod_group_data['median'].tolist()

                            # 绘制趋势线
                            ax.plot(x_positions, y_values,
                                   color=colors[mod_idx], marker='o',
                                   linewidth=2, markersize=6,
                                   label=modification_type, alpha=0.8)

                    # 设置图表属性
                    ax.set_xlabel('Dataset', fontsize=12)
                    ax.set_ylabel(f'{prop_name}', fontsize=12)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                    ax.grid(True, alpha=0.3)

                    # 设置x轴标签
                    ax.set_xticks(range(len(datasets)))
                    ax.set_xticklabels([ds.replace('PXD', '') for ds in datasets],
                                      rotation=45, ha='right', fontsize=10)

                plt.tight_layout()

                # 保存图片
                filename = f'comprehensive_trends_{prop_name}_modified.png'
                filepath = os.path.join(self.fig_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"    [OK] 保存: {filename}")

        except Exception as e:
            print(f"[ERROR] 创建综合趋势图失败: {e}")

    def generate_analysis_report(self, group_stats_df: pd.DataFrame,
                               stat_tests_df: pd.DataFrame, successful_datasets: list):
        """生成分析报告"""

        report_file = os.path.join(self.output_dir, 'analysis_report_modified.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("肽段分组分析报告（修改版逻辑）\n")
            f.write("=" * 50 + "\n\n")

            # 新逻辑说明
            f.write("分组逻辑说明:\n")
            f.write("- PTM-only: 只检测到特定修饰类型的肽段（保持原逻辑）\n")
            f.write("- PTM + WT: 同序列同时检测到任意修饰类型与未修饰肽（新逻辑）\n")
            f.write("- WT-only: 只检测到未修饰肽，从未检测到任何修饰（新逻辑）\n\n")

            # 总体统计
            f.write("总体统计:\n")
            f.write(f"  成功分析数据集数量: {len(successful_datasets)}\n")
            f.write(f"  总分析记录数: {len(group_stats_df)}\n")
            f.write(f"  统计检验记录数: {len(stat_tests_df)}\n\n")

            # 数据集列表
            f.write("成功分析的数据集:\n")
            for i, dataset in enumerate(successful_datasets, 1):
                f.write(f"  {i:2d}. {dataset}\n")
            f.write("\n")

            # 分组统计
            f.write("肽段分组统计:\n")
            group_summary = group_stats_df.groupby('peptide_group')['count'].agg(['sum', 'mean', 'std']).round(2)
            for group, stats in group_summary.iterrows():
                f.write(f"  {group}:\n")
                f.write(f"    总肽段数: {stats['sum']:,.0f}\n")
                f.write(f"    平均每数据集: {stats['mean']:.0f} ± {stats['std']:.0f}\n")
            f.write("\n")

            # 理化性质统计
            f.write("理化性质统计 (各组中位数):\n")
            for prop in self.property_names:
                f.write(f"\n  {prop}:\n")
                prop_data = group_stats_df[group_stats_df['property'] == prop]
                prop_summary = prop_data.groupby('peptide_group')['median'].agg(['mean', 'std']).round(3)

                for group, stats in prop_summary.iterrows():
                    f.write(f"    {group}: {stats['mean']:.3f} ± {stats['std']:.3f}\n")

            # 显著性检验汇总
            if len(stat_tests_df) > 0:
                f.write("\n\n显著性检验汇总:\n")

                # Kruskal-Wallis检验结果
                kw_tests = stat_tests_df[stat_tests_df['test_type'] == 'Kruskal-Wallis']
                if len(kw_tests) > 0:
                    f.write("  Kruskal-Wallis检验 (总体差异):\n")
                    for prop in self.property_names:
                        prop_tests = kw_tests[kw_tests['property'] == prop]
                        if len(prop_tests) > 0:
                            sig_count = prop_tests['significant'].sum()
                            total_count = len(prop_tests)
                            f.write(f"    {prop}: {sig_count}/{total_count} 数据集显著 "
                                   f"({sig_count/total_count*100:.1f}%)\n")

                # 两两比较结果
                pairwise_tests = stat_tests_df[stat_tests_df['test_type'] == 'Mann-Whitney U']
                if len(pairwise_tests) > 0:
                    f.write("\n  两两比较 (Mann-Whitney U):\n")
                    comparison_summary = pairwise_tests.groupby('comparison')['significant'].agg(['sum', 'count'])
                    for comparison, stats in comparison_summary.iterrows():
                        sig_rate = stats['sum'] / stats['count'] * 100
                        f.write(f"    {comparison}: {stats['sum']}/{stats['count']} 显著 "
                               f"({sig_rate:.1f}%)\n")

        print(f"  [OK] 保存: analysis_report_modified.txt")


def main():
    """主函数"""
    print("肽段分组分析工具（修改版）")
    print("=" * 60)
    print("根据修饰检测情况将肽段分为三组进行比较分析（新逻辑）")
    print("- PTM-only: 只检测到特定修饰类型的肽段")
    print("- PTM + WT: 同序列同时检测到任意修饰类型与未修饰肽")
    print("- WT-only: 只检测到未修饰肽，从未检测到任何修饰")
    print("=" * 60)

    # 创建分析器
    analyzer = PeptideGroupAnalyzer()

    # 运行分析（可以设置限制用于测试）
    # results = analyzer.run_full_analysis(max_datasets=3)  # 测试用
    results = analyzer.run_full_analysis()  # 完整分析

    if results is not None:
        group_stats, stat_tests = results
        print(f"\n[STATS] 分析结果汇总:")
        print(f"- 分组统计记录: {len(group_stats)}")
        print(f"- 统计检验记录: {len(stat_tests)}")
        print(f"- 输出目录: {analyzer.output_dir}")
        print(f"- 图表目录: {analyzer.fig_dir}")
    else:
        print("[ERROR] 分析失败")


if __name__ == "__main__":
    main()