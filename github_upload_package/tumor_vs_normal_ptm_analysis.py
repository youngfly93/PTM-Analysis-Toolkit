#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肿瘤 vs 正常样本 PTM 效应比较分析

分析特定PTM在肿瘤样本vs正常样本中对理化性质的相对影响
方法：ratio = (该PTM肽中位数) ÷ (同一样本中未修饰肽中位数)，再比较 Tumor vs Normal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 尝试导入statsmodels，如果没有则使用简单的多重比较校正
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("⚠️  statsmodels未安装，将使用简单的Bonferroni校正")

# 导入现有的分析模块
try:
    from peptide_properties_analyzer import PeptidePropertiesAnalyzer
    HAS_PROPERTIES_ANALYZER = True
except ImportError:
    HAS_PROPERTIES_ANALYZER = False
    print("⚠️  peptide_properties_analyzer未找到，将使用简化的理化性质计算")

class TumorNormalPTMAnalyzer:
    """肿瘤vs正常样本PTM效应比较分析器"""
    
    def __init__(self):
        if HAS_PROPERTIES_ANALYZER:
            self.properties_analyzer = PeptidePropertiesAnalyzer()
        else:
            self.properties_analyzer = None
        
        # 目标修饰类型
        self.target_modifications = [
            'Phospho[S]', 'Phospho[T]', 'Phospho[Y]',
            'Acetyl[K]', 'Methyl[K]', 'Dimethyl[K]', 'Trimethyl[K]',
            'Deamidated[N]', 'Deamidated[Q]',
            'Ubiquitination[K]', 'Citrullination[R]'
        ]
        
        # 理化性质列名
        self.property_columns = [
            'corrected_pi', 'corrected_charge_at_ph7', 
            'corrected_hydrophobicity_kd', 'corrected_molecular_weight'
        ]
        
        # 理化性质显示名称
        self.property_names = ['pI', 'Net_Charge', 'Hydrophobicity_KD', 'Molecular_Weight']
        
        # 输出目录
        self.output_dir = 'tumor_vs_normal_ptm_results'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_and_prepare_data(self, dataset_id: str):
        """加载并准备数据"""
        print(f"📊 加载数据集: {dataset_id}")
        
        # 加载summary文件获取样本类型信息
        summary_file = f'tumor_summary/{dataset_id}_summary.csv'
        if not os.path.exists(summary_file):
            print(f"❌ 找不到summary文件: {summary_file}")
            return None
        
        summary_df = pd.read_csv(summary_file)
        
        # 检查是否有Tumor和Normal样本
        sample_types = summary_df['Type'].unique()
        if not ('Tumor' in sample_types and 'Normal' in sample_types):
            print(f"⚠️  数据集 {dataset_id} 没有同时包含Tumor和Normal样本")
            print(f"   样本类型: {sample_types}")
            return None
        
        print(f"✅ 发现样本类型: {sample_types}")
        print(f"   Tumor样本: {len(summary_df[summary_df['Type'] == 'Tumor'])} 个")
        print(f"   Normal样本: {len(summary_df[summary_df['Type'] == 'Normal'])} 个")
        
        # 加载原始数据
        dataset_dir = f'{dataset_id}_human'
        if not os.path.exists(dataset_dir):
            print(f"❌ 找不到数据目录: {dataset_dir}")
            return None
        
        # 处理所有样本文件
        all_data = []
        
        for _, row in summary_df.iterrows():
            file_name = row['File Name']
            sample_name = row['Sample Name']
            sample_type = row['Type']
            
            file_path = os.path.join(dataset_dir, file_name)
            if not os.path.exists(file_path):
                print(f"⚠️  文件不存在: {file_path}")
                continue
            
            try:
                # 读取单个文件
                sample_data = pd.read_csv(file_path, sep='\t')

                # 调试：检查列名
                if len(all_data) == 0:  # 只对第一个文件打印
                    print(f"  第一个文件列名: {sample_data.columns.tolist()}")
                    print(f"  第一个文件形状: {sample_data.shape}")
                    if 'Modification' in sample_data.columns:
                        mod_sample = sample_data['Modification'].dropna().head(3)
                        print(f"  修饰列样例: {mod_sample.tolist()}")

                # 添加样本信息
                sample_data['sample_id'] = sample_name
                sample_data['sample_type'] = sample_type
                sample_data['dataset_id'] = dataset_id

                all_data.append(sample_data)

            except Exception as e:
                print(f"⚠️  读取文件失败 {file_path}: {e}")
                continue
        
        if not all_data:
            print(f"❌ 没有成功读取任何数据文件")
            return None
        
        # 合并所有数据
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"📊 合并数据: {len(combined_data)} 条记录")
        
        # 数据预处理
        processed_data = self.preprocess_data(combined_data)
        
        return processed_data
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        print("🔧 数据预处理...")
        
        # 重命名列以匹配期望格式
        column_mapping = {
            'Sequence': 'sequence',
            'Modification': 'modification_info'
        }

        print(f"  合并后列名: {data.columns.tolist()}")
        print(f"  检查Modification列是否存在: {'Modification' in data.columns}")

        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                print(f"  重命名 {old_col} -> {new_col}")
                data = data.rename(columns={old_col: new_col})
            else:
                print(f"  ⚠️  列 {old_col} 不存在")

        print(f"  重命名后列名: {data.columns.tolist()}")
        if 'modification_info' in data.columns:
            mod_sample = data['modification_info'].dropna().head(3)
            print(f"  重命名后修饰列样例: {mod_sample.tolist()}")
        
        # 解析修饰信息
        data = self.parse_modifications(data)
        
        # 计算理化性质
        data = self.calculate_properties(data)
        
        # 过滤有效数据
        data = data.dropna(subset=['sequence'] + self.property_columns)
        
        print(f"✅ 预处理完成: {len(data)} 条有效记录")
        
        return data
    
    def parse_modifications(self, data: pd.DataFrame) -> pd.DataFrame:
        """解析修饰信息"""
        print("🔍 解析修饰信息...")

        # 初始化修饰相关列
        data['modification_type'] = ''
        data['num_modifications'] = 0

        # 解析Modification列
        for idx, row in data.iterrows():
            mod_info = str(row.get('modification_info', ''))

            # 调试前几行
            if idx < 5:
                print(f"  行 {idx}: 修饰信息 = '{mod_info}'")

            if pd.isna(mod_info) or mod_info == '' or mod_info == 'nan':
                # 未修饰
                data.at[idx, 'modification_type'] = ''
                data.at[idx, 'num_modifications'] = 0
            else:
                # 解析修饰
                modifications = []
                mod_count = 0

                # 分割多个修饰（用分号分隔）
                mod_parts = mod_info.split(';')

                for part in mod_parts:
                    part = part.strip()
                    if not part:
                        continue

                    # 格式: position,ModificationType
                    if ',' in part:
                        try:
                            pos, mod_type = part.split(',', 1)
                            mod_type = mod_type.strip()

                            # 检查是否是目标修饰
                            target_found = None

                            # 磷酸化修饰
                            if 'Phospho' in mod_type:
                                if '[S]' in mod_type:
                                    target_found = 'Phospho[S]'
                                elif '[T]' in mod_type:
                                    target_found = 'Phospho[T]'
                                elif '[Y]' in mod_type:
                                    target_found = 'Phospho[Y]'

                            # 乙酰化修饰
                            elif 'Acetyl' in mod_type:
                                if '[K]' in mod_type:
                                    target_found = 'Acetyl[K]'
                                elif 'ProteinN-term' in mod_type:
                                    target_found = 'Acetyl[K]'  # N端乙酰化也算

                            # 甲基化修饰
                            elif 'Methyl' in mod_type:
                                if 'Dimethyl' in mod_type and '[K]' in mod_type:
                                    target_found = 'Dimethyl[K]'
                                elif 'Trimethyl' in mod_type and '[K]' in mod_type:
                                    target_found = 'Trimethyl[K]'
                                elif '[K]' in mod_type:
                                    target_found = 'Methyl[K]'

                            # 脱酰胺修饰
                            elif 'Deamidated' in mod_type:
                                if '[N]' in mod_type:
                                    target_found = 'Deamidated[N]'
                                elif '[Q]' in mod_type:
                                    target_found = 'Deamidated[Q]'

                            if target_found:
                                modifications.append(target_found)
                                mod_count += 1
                            else:
                                # 其他修饰
                                mod_count += 1

                        except:
                            continue

                # 设置修饰信息
                if modifications:
                    # 如果有多个目标修饰，取第一个
                    data.at[idx, 'modification_type'] = modifications[0]
                    data.at[idx, 'num_modifications'] = len(modifications)
                else:
                    # 有修饰但不是目标修饰
                    data.at[idx, 'modification_type'] = 'Other'
                    data.at[idx, 'num_modifications'] = mod_count if mod_count > 0 else 1

        return data
    
    def calculate_properties(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算理化性质"""
        print("⚗️  计算理化性质...")

        # 为每个肽段计算理化性质
        properties_list = []

        for idx, row in data.iterrows():
            sequence = row['sequence']

            try:
                if self.properties_analyzer:
                    # 使用现有的分析器计算理化性质
                    properties = self.properties_analyzer.calculate_peptide_properties(sequence)
                    properties_list.append(properties)
                else:
                    # 使用简化的理化性质计算
                    properties = self.calculate_simple_properties(sequence)
                    properties_list.append(properties)
            except Exception as e:
                # 如果计算失败，使用NaN
                properties_list.append({
                    'corrected_pi': np.nan,
                    'corrected_charge_at_ph7': np.nan,
                    'corrected_hydrophobicity_kd': np.nan,
                    'corrected_molecular_weight': np.nan
                })

        # 添加理化性质到数据框
        properties_df = pd.DataFrame(properties_list)
        for col in self.property_columns:
            data[col] = properties_df[col]

        return data

    def calculate_simple_properties(self, sequence: str) -> dict:
        """简化的理化性质计算"""
        # 氨基酸基本性质
        aa_properties = {
            'A': {'mw': 71.04, 'pka': 0, 'hydro': 1.8},
            'R': {'mw': 156.10, 'pka': 12.48, 'hydro': -4.5},
            'N': {'mw': 114.04, 'pka': 0, 'hydro': -3.5},
            'D': {'mw': 115.03, 'pka': 3.65, 'hydro': -3.5},
            'C': {'mw': 103.01, 'pka': 8.18, 'hydro': 2.5},
            'E': {'mw': 129.04, 'pka': 4.25, 'hydro': -3.5},
            'Q': {'mw': 128.06, 'pka': 0, 'hydro': -3.5},
            'G': {'mw': 57.02, 'pka': 0, 'hydro': -0.4},
            'H': {'mw': 137.06, 'pka': 6.00, 'hydro': -3.2},
            'I': {'mw': 113.08, 'pka': 0, 'hydro': 4.5},
            'L': {'mw': 113.08, 'pka': 0, 'hydro': 3.8},
            'K': {'mw': 128.09, 'pka': 10.53, 'hydro': -3.9},
            'M': {'mw': 131.04, 'pka': 0, 'hydro': 1.9},
            'F': {'mw': 147.07, 'pka': 0, 'hydro': 2.8},
            'P': {'mw': 97.05, 'pka': 0, 'hydro': -1.6},
            'S': {'mw': 87.03, 'pka': 0, 'hydro': -0.8},
            'T': {'mw': 101.05, 'pka': 0, 'hydro': -0.7},
            'W': {'mw': 186.08, 'pka': 0, 'hydro': -0.9},
            'Y': {'mw': 163.06, 'pka': 10.07, 'hydro': -1.3},
            'V': {'mw': 99.07, 'pka': 0, 'hydro': 4.2}
        }

        # 计算分子量
        mw = 18.015  # 水分子
        for aa in sequence:
            if aa in aa_properties:
                mw += aa_properties[aa]['mw']

        # 计算净电荷（pH 7）
        charge = 0
        for aa in sequence:
            if aa in aa_properties:
                pka = aa_properties[aa]['pka']
                if pka > 0:
                    if aa in ['R', 'K', 'H']:  # 碱性
                        charge += 1 / (1 + 10**(7 - pka))
                    elif aa in ['D', 'E']:  # 酸性
                        charge -= 1 / (1 + 10**(pka - 7))

        # N端和C端
        charge += 1 / (1 + 10**(7 - 9.6))  # N端
        charge -= 1 / (1 + 10**(2.34 - 7))  # C端

        # 计算疏水性（Kyte-Doolittle）
        hydro = 0
        for aa in sequence:
            if aa in aa_properties:
                hydro += aa_properties[aa]['hydro']
        hydro = hydro / len(sequence) if len(sequence) > 0 else 0

        # 简化的pI计算（近似）
        pi = 7.0 + charge * 2  # 粗略估计

        return {
            'corrected_pi': pi,
            'corrected_charge_at_ph7': charge,
            'corrected_hydrophobicity_kd': hydro,
            'corrected_molecular_weight': mw
        }
    
    def analyze_ptm_effects(self, data: pd.DataFrame, target_ptm: str) -> tuple:
        """分析特定PTM的效应"""
        print(f"🔬 分析 {target_ptm} 的效应...")

        # 调试：检查修饰类型分布
        mod_counts = data['modification_type'].value_counts()
        print(f"   修饰类型分布:")
        for mod_type, count in mod_counts.head(10).items():
            print(f"     {mod_type}: {count}")

        # 筛选数据
        ptm_data = data[data['modification_type'] == target_ptm].copy()
        wt_data = data[data['num_modifications'] == 0].copy()

        if len(ptm_data) == 0:
            print(f"⚠️  没有找到 {target_ptm} 修饰的肽段")
            return None, None

        if len(wt_data) == 0:
            print(f"⚠️  没有找到未修饰的肽段")
            return None, None

        print(f"   {target_ptm} 肽段: {len(ptm_data)} 条")
        print(f"   未修饰肽段: {len(wt_data)} 条")

        # 检查样本分布
        ptm_samples = ptm_data['sample_id'].unique()
        wt_samples = wt_data['sample_id'].unique()
        print(f"   {target_ptm} 样本数: {len(ptm_samples)}")
        print(f"   未修饰样本数: {len(wt_samples)}")

        # 检查样本类型分布
        ptm_sample_types = ptm_data['sample_type'].value_counts()
        wt_sample_types = wt_data['sample_type'].value_counts()
        print(f"   {target_ptm} 样本类型: {ptm_sample_types.to_dict()}")
        print(f"   未修饰样本类型: {wt_sample_types.to_dict()}")
        
        # 计算每个样本的中位数
        results = []
        
        for sample_id in data['sample_id'].unique():
            sample_info = data[data['sample_id'] == sample_id].iloc[0]
            sample_type = sample_info['sample_type']
            
            # 该样本的PTM和WT数据
            sample_ptm = ptm_data[ptm_data['sample_id'] == sample_id]
            sample_wt = wt_data[wt_data['sample_id'] == sample_id]
            
            if len(sample_ptm) == 0 or len(sample_wt) == 0:
                continue
            
            # 计算每个理化性质的ratio
            for prop_col, prop_name in zip(self.property_columns, self.property_names):
                ptm_median = sample_ptm[prop_col].median()
                wt_median = sample_wt[prop_col].median()
                
                if pd.notna(ptm_median) and pd.notna(wt_median) and wt_median != 0:
                    ratio = ptm_median / wt_median
                    log2_ratio = np.log2(ratio)
                    
                    results.append({
                        'sample_id': sample_id,
                        'sample_type': sample_type,
                        'property': prop_name,
                        'ptm_median': ptm_median,
                        'wt_median': wt_median,
                        'ratio': ratio,
                        'log2_ratio': log2_ratio,
                        'ptm_count': len(sample_ptm),
                        'wt_count': len(sample_wt)
                    })
        
        if not results:
            print(f"⚠️  没有有效的ratio计算结果")
            return None, None
        
        summary_df = pd.DataFrame(results)
        
        # 统计检验
        stats_results = self.perform_statistical_tests(summary_df, target_ptm)
        
        return summary_df, stats_results
    
    def perform_statistical_tests(self, summary_df: pd.DataFrame, target_ptm: str) -> pd.DataFrame:
        """执行统计检验"""
        print(f"📈 执行统计检验...")
        
        stats_results = []
        
        for prop_name in self.property_names:
            prop_data = summary_df[summary_df['property'] == prop_name]
            
            if len(prop_data) == 0:
                continue
            
            tumor_data = prop_data[prop_data['sample_type'] == 'Tumor']['log2_ratio']
            normal_data = prop_data[prop_data['sample_type'] == 'Normal']['log2_ratio']
            
            if len(tumor_data) == 0 or len(normal_data) == 0:
                continue
            
            # Mann-Whitney U检验（非配对）
            try:
                statistic, p_value = stats.mannwhitneyu(
                    tumor_data, normal_data, alternative='two-sided'
                )
                
                # 计算效应量（Cliff's delta）
                cliff_delta = self.calculate_cliff_delta(tumor_data, normal_data)
                
                stats_results.append({
                    'ptm': target_ptm,
                    'property': prop_name,
                    'test_type': 'Mann-Whitney U',
                    'statistic': statistic,
                    'p_value': p_value,
                    'cliff_delta': cliff_delta,
                    'tumor_n': len(tumor_data),
                    'normal_n': len(normal_data),
                    'tumor_median': tumor_data.median(),
                    'normal_median': normal_data.median(),
                    'tumor_mean': tumor_data.mean(),
                    'normal_mean': normal_data.mean()
                })
                
            except Exception as e:
                print(f"⚠️  统计检验失败 {prop_name}: {e}")
                continue
        
        if not stats_results:
            return pd.DataFrame()
        
        stats_df = pd.DataFrame(stats_results)
        
        # 多重比较校正
        if len(stats_df) > 1:
            if HAS_STATSMODELS:
                _, p_adj, _, _ = multipletests(stats_df['p_value'], method='fdr_bh')
                stats_df['p_adj'] = p_adj
            else:
                # 简单的Bonferroni校正
                stats_df['p_adj'] = stats_df['p_value'] * len(stats_df)
                stats_df['p_adj'] = np.minimum(stats_df['p_adj'], 1.0)
        else:
            stats_df['p_adj'] = stats_df['p_value']
        
        # 添加显著性标记
        stats_df['significant'] = stats_df['p_adj'] < 0.05
        
        return stats_df
    
    def calculate_cliff_delta(self, x, y):
        """计算Cliff's delta效应量"""
        try:
            n1, n2 = len(x), len(y)
            if n1 == 0 or n2 == 0:
                return np.nan
            
            # 计算所有配对比较
            comparisons = []
            for xi in x:
                for yi in y:
                    if xi > yi:
                        comparisons.append(1)
                    elif xi < yi:
                        comparisons.append(-1)
                    else:
                        comparisons.append(0)
            
            cliff_delta = np.mean(comparisons)
            return cliff_delta
            
        except:
            return np.nan

    def create_visualizations(self, summary_df: pd.DataFrame, stats_df: pd.DataFrame, target_ptm: str):
        """创建可视化图表"""
        print(f"📊 创建 {target_ptm} 的可视化图表...")

        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")

        # 设置中文字体以避免乱码
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # 为每个理化性质创建图表
        for prop_name in self.property_names:
            prop_data = summary_df[summary_df['property'] == prop_name]

            if len(prop_data) == 0:
                continue

            # 创建子图
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # 1. 箱线图/小提琴图
            ax1 = axes[0]

            # 小提琴图
            sns.violinplot(data=prop_data, x='sample_type', y='log2_ratio',
                          ax=ax1, inner='box', width=0.6)

            # 添加散点
            sns.stripplot(data=prop_data, x='sample_type', y='log2_ratio',
                         ax=ax1, size=4, alpha=0.7, color='black')

            ax1.set_title(f'{target_ptm} - {prop_name}\nlog2(PTM/WT ratio)')
            ax1.set_xlabel('Sample Type')
            ax1.set_ylabel('log2(PTM/WT ratio)')
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)

            # 添加统计信息
            prop_stats = stats_df[stats_df['property'] == prop_name]
            if len(prop_stats) > 0:
                stat_info = prop_stats.iloc[0]
                p_val = stat_info['p_adj']

                if p_val < 0.001:
                    sig_text = '***'
                elif p_val < 0.01:
                    sig_text = '**'
                elif p_val < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'ns'

                ax1.text(0.5, 0.95, f'p = {p_val:.3f} {sig_text}',
                        transform=ax1.transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # 2. 配对连线图（如果有配对样本）
            ax2 = axes[1]

            # 检查是否有配对样本
            tumor_samples = set(prop_data[prop_data['sample_type'] == 'Tumor']['sample_id'])
            normal_samples = set(prop_data[prop_data['sample_type'] == 'Normal']['sample_id'])

            # 简化的配对逻辑：基于样本名称相似性
            paired_data = []
            for tumor_sample in tumor_samples:
                # 寻找最相似的normal样本
                best_match = None
                best_score = 0

                for normal_sample in normal_samples:
                    # 计算样本名称相似性
                    common_parts = set(tumor_sample.split('_')) & set(normal_sample.split('_'))
                    score = len(common_parts)

                    if score > best_score:
                        best_score = score
                        best_match = normal_sample

                if best_match and best_score >= 2:  # 至少有2个共同部分
                    tumor_value = prop_data[
                        (prop_data['sample_id'] == tumor_sample) &
                        (prop_data['sample_type'] == 'Tumor')
                    ]['log2_ratio'].iloc[0]

                    normal_value = prop_data[
                        (prop_data['sample_id'] == best_match) &
                        (prop_data['sample_type'] == 'Normal')
                    ]['log2_ratio'].iloc[0]

                    paired_data.append({
                        'tumor_sample': tumor_sample,
                        'normal_sample': best_match,
                        'tumor_value': tumor_value,
                        'normal_value': normal_value
                    })

            if paired_data:
                # 绘制配对连线图
                paired_df = pd.DataFrame(paired_data)

                for _, row in paired_df.iterrows():
                    ax2.plot([0, 1], [row['normal_value'], row['tumor_value']],
                            'o-', alpha=0.6, linewidth=1)

                ax2.set_xlim(-0.2, 1.2)
                ax2.set_xticks([0, 1])
                ax2.set_xticklabels(['Normal', 'Tumor'])
                ax2.set_ylabel('log2(PTM/WT ratio)')
                ax2.set_title(f'Paired Samples (n={len(paired_df)})')
                ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

                # 配对t检验
                if len(paired_df) >= 3:
                    try:
                        t_stat, t_p = stats.ttest_rel(paired_df['tumor_value'],
                                                     paired_df['normal_value'])
                        ax2.text(0.5, 0.95, f'Paired t-test: p = {t_p:.3f}',
                                transform=ax2.transAxes, ha='center', va='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
            else:
                # 如果没有配对样本，显示分组箱线图
                sns.boxplot(data=prop_data, x='sample_type', y='log2_ratio', ax=ax2)
                ax2.set_title('Unpaired Comparison')
                ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

            plt.tight_layout()

            # 保存图表
            filename = f'{target_ptm}_{prop_name}_comparison.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"   保存图表: {filename}")

    def save_results(self, summary_df: pd.DataFrame, stats_df: pd.DataFrame, target_ptm: str):
        """保存分析结果"""
        print(f"💾 保存 {target_ptm} 的分析结果...")

        # 保存summary数据
        summary_file = os.path.join(self.output_dir, f'summary_{target_ptm}.csv')
        summary_df.to_csv(summary_file, index=False)

        # 保存统计结果
        stats_file = os.path.join(self.output_dir, f'stats_{target_ptm}.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"Statistical Analysis Results for {target_ptm}\n")
            f.write("=" * 50 + "\n\n")

            for _, row in stats_df.iterrows():
                f.write(f"Property: {row['property']}\n")
                f.write(f"Test: {row['test_type']}\n")
                f.write(f"Statistic: {row['statistic']:.4f}\n")
                f.write(f"P-value: {row['p_value']:.6f}\n")
                f.write(f"Adjusted P-value: {row['p_adj']:.6f}\n")
                f.write(f"Significant: {'Yes' if row['significant'] else 'No'}\n")
                f.write(f"Cliff's Delta: {row['cliff_delta']:.4f}\n")
                f.write(f"Tumor median: {row['tumor_median']:.4f}\n")
                f.write(f"Normal median: {row['normal_median']:.4f}\n")
                f.write(f"Sample sizes: Tumor={row['tumor_n']}, Normal={row['normal_n']}\n")
                f.write("-" * 30 + "\n\n")

        print(f"   保存数据: summary_{target_ptm}.csv")
        print(f"   保存统计: stats_{target_ptm}.txt")

    def analyze_dataset(self, dataset_id: str, target_ptms: list = None):
        """分析单个数据集"""
        if target_ptms is None:
            target_ptms = self.target_modifications

        print(f"\n🔬 开始分析数据集: {dataset_id}")
        print("=" * 60)

        # 加载数据
        data = self.load_and_prepare_data(dataset_id)
        if data is None:
            return

        # 分析每种PTM
        all_results = {}

        for ptm in target_ptms:
            print(f"\n📊 分析 {ptm}...")

            summary_df, stats_df = self.analyze_ptm_effects(data, ptm)

            if summary_df is not None and stats_df is not None:
                # 创建可视化
                self.create_visualizations(summary_df, stats_df, ptm)

                # 保存结果
                self.save_results(summary_df, stats_df, ptm)

                all_results[ptm] = {
                    'summary': summary_df,
                    'stats': stats_df
                }

                print(f"✅ {ptm} 分析完成")
            else:
                print(f"❌ {ptm} 分析失败")

        # 生成综合报告
        self.generate_comprehensive_report(all_results, dataset_id)

        print(f"\n🎉 数据集 {dataset_id} 分析完成！")
        print(f"📁 结果保存在: {self.output_dir}")

    def generate_comprehensive_report(self, all_results: dict, dataset_id: str):
        """生成综合分析报告"""
        print("📝 生成综合分析报告...")

        report_file = os.path.join(self.output_dir, f'comprehensive_report_{dataset_id}.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Tumor vs Normal PTM Analysis Report\n")
            f.write(f"Dataset: {dataset_id}\n")
            f.write("=" * 60 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Analyzed PTMs: {len(all_results)}\n")
            f.write(f"Analysis method: log₂(PTM median / WT median) ratio comparison\n")
            f.write(f"Statistical test: Mann-Whitney U test\n")
            f.write(f"Multiple testing correction: FDR (Benjamini-Hochberg)\n\n")

            # 显著性结果汇总
            f.write("SIGNIFICANT RESULTS\n")
            f.write("-" * 20 + "\n")

            significant_count = 0
            for ptm, results in all_results.items():
                stats_df = results['stats']
                significant_results = stats_df[stats_df['significant']]

                if len(significant_results) > 0:
                    f.write(f"\n{ptm}:\n")
                    for _, row in significant_results.iterrows():
                        direction = "↑" if row['tumor_median'] > row['normal_median'] else "↓"
                        f.write(f"  {row['property']}: p_adj = {row['p_adj']:.4f} {direction}\n")
                        f.write(f"    Tumor median: {row['tumor_median']:.4f}\n")
                        f.write(f"    Normal median: {row['normal_median']:.4f}\n")
                        f.write(f"    Effect size (Cliff's δ): {row['cliff_delta']:.4f}\n")
                    significant_count += len(significant_results)

            if significant_count == 0:
                f.write("No significant differences found.\n")

            f.write(f"\nTotal significant comparisons: {significant_count}\n")

        print(f"   保存报告: comprehensive_report_{dataset_id}.txt")

    def find_tumor_normal_datasets(self):
        """查找包含Tumor和Normal样本的数据集"""
        print("🔍 查找包含Tumor和Normal样本的数据集...")

        tumor_normal_datasets = []

        # 检查tumor_summary目录中的所有summary文件
        summary_dir = 'tumor_summary'
        if not os.path.exists(summary_dir):
            print(f"❌ 找不到summary目录: {summary_dir}")
            return []

        for file in os.listdir(summary_dir):
            if file.endswith('_summary.csv'):
                dataset_id = file.replace('_summary.csv', '')
                summary_file = os.path.join(summary_dir, file)

                try:
                    df = pd.read_csv(summary_file)
                    if 'Type' in df.columns:
                        types = df['Type'].unique()
                        has_both = 'Tumor' in types and 'Normal' in types

                        if has_both:
                            tumor_count = len(df[df['Type'] == 'Tumor'])
                            normal_count = len(df[df['Type'] == 'Normal'])

                            # 检查是否有对应的数据目录
                            data_dir = f'{dataset_id}_human'
                            if os.path.exists(data_dir):
                                tumor_normal_datasets.append({
                                    'dataset_id': dataset_id,
                                    'tumor_count': tumor_count,
                                    'normal_count': normal_count
                                })
                                print(f"   ✅ {dataset_id}: Tumor={tumor_count}, Normal={normal_count}")
                            else:
                                print(f"   ⚠️  {dataset_id}: 有summary但缺少数据目录")
                except Exception as e:
                    print(f"   ❌ {dataset_id}: 读取summary失败 - {e}")

        print(f"\n📊 总共找到 {len(tumor_normal_datasets)} 个可分析的数据集")
        return tumor_normal_datasets

    def check_dataset_completion(self, dataset_id: str, target_ptms: list) -> bool:
        """检查数据集是否已经完成分析"""
        dataset_output_dir = f'tumor_vs_normal_ptm_results/{dataset_id}'

        if not os.path.exists(dataset_output_dir):
            return False

        # 检查是否有综合报告
        report_file = os.path.join(dataset_output_dir, f'comprehensive_report_{dataset_id}.txt')
        if not os.path.exists(report_file):
            return False

        # 检查每个PTM的结果文件是否存在
        for ptm in target_ptms:
            summary_file = os.path.join(dataset_output_dir, f'summary_{ptm}.csv')
            stats_file = os.path.join(dataset_output_dir, f'stats_{ptm}.txt')

            if not (os.path.exists(summary_file) and os.path.exists(stats_file)):
                return False

        return True

    def save_progress(self, completed_datasets: list, failed_datasets: list):
        """保存分析进度"""
        progress_file = os.path.join(self.output_dir, 'analysis_progress.json')

        progress_data = {
            'completed_datasets': completed_datasets,
            'failed_datasets': failed_datasets,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        import json
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)

    def load_progress(self):
        """加载分析进度"""
        progress_file = os.path.join(self.output_dir, 'analysis_progress.json')

        if not os.path.exists(progress_file):
            return [], []

        try:
            import json
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)

            return progress_data.get('completed_datasets', []), progress_data.get('failed_datasets', [])
        except Exception as e:
            print(f"⚠️  读取进度文件失败: {e}")
            return [], []

    def analyze_all_datasets(self, target_ptms: list = None, resume: bool = True):
        """分析所有包含Tumor和Normal样本的数据集"""
        if target_ptms is None:
            target_ptms = ['Phospho[S]', 'Acetyl[K]', 'Deamidated[N]']

        # 查找所有可分析的数据集
        datasets = self.find_tumor_normal_datasets()

        if not datasets:
            print("❌ 没有找到包含Tumor和Normal样本的数据集")
            return

        # 加载之前的进度
        completed_datasets, failed_datasets = [], []
        if resume:
            completed_datasets, failed_datasets = self.load_progress()

            if completed_datasets or failed_datasets:
                print(f"\n� 检测到之前的分析进度:")
                print(f"   已完成: {len(completed_datasets)} 个数据集")
                print(f"   已失败: {len(failed_datasets)} 个数据集")

        # 检查每个数据集的完成状态
        datasets_to_analyze = []
        for dataset_info in datasets:
            dataset_id = dataset_info['dataset_id']

            if dataset_id in completed_datasets:
                # 验证文件是否真的存在
                if self.check_dataset_completion(dataset_id, target_ptms):
                    print(f"   ✅ {dataset_id}: 已完成，跳过")
                    continue
                else:
                    print(f"   ⚠️  {dataset_id}: 标记为完成但文件不完整，重新分析")
                    completed_datasets.remove(dataset_id)

            datasets_to_analyze.append(dataset_info)

        if not datasets_to_analyze:
            print(f"\n🎉 所有数据集都已完成分析！")
            self.generate_overall_report(completed_datasets, failed_datasets, datasets)
            return

        print(f"\n🚀 开始分析 {len(datasets_to_analyze)} 个数据集...")
        print("=" * 60)

        for i, dataset_info in enumerate(datasets_to_analyze, 1):
            dataset_id = dataset_info['dataset_id']

            print(f"\n📊 [{i}/{len(datasets_to_analyze)}] 分析数据集: {dataset_id}")
            print(f"   Tumor样本: {dataset_info['tumor_count']}")
            print(f"   Normal样本: {dataset_info['normal_count']}")

            try:
                # 为每个数据集创建单独的输出目录
                original_output_dir = self.output_dir
                self.output_dir = f'tumor_vs_normal_ptm_results/{dataset_id}'
                os.makedirs(self.output_dir, exist_ok=True)

                # 分析数据集
                self.analyze_dataset(dataset_id, target_ptms)
                completed_datasets.append(dataset_id)

                # 恢复原始输出目录
                self.output_dir = original_output_dir

                # 保存进度
                self.save_progress(completed_datasets, failed_datasets)

                print(f"   ✅ {dataset_id} 分析完成并保存进度")

            except Exception as e:
                print(f"❌ 数据集 {dataset_id} 分析失败: {e}")
                failed_datasets.append((dataset_id, str(e)))

                # 恢复原始输出目录
                self.output_dir = original_output_dir

                # 保存进度
                self.save_progress(completed_datasets, failed_datasets)
                continue

        # 生成总体报告
        self.generate_overall_report(completed_datasets, failed_datasets, datasets)

        # 生成数据集级别汇总分析
        if completed_datasets:
            self.create_dataset_level_summary(completed_datasets)

        print(f"\n🎉 批量分析完成！")
        print(f"   成功: {len(completed_datasets)} 个数据集")
        print(f"   失败: {len(failed_datasets)} 个数据集")
        print(f"   数据集级别汇总: dataset_level_summary.csv")
        print(f"   汇总可视化: dataset_level_*.png")

    def generate_overall_report(self, successful_analyses, failed_analyses, all_datasets):
        """生成总体分析报告"""
        print("📝 生成总体分析报告...")

        report_file = os.path.join(self.output_dir, 'overall_tumor_vs_normal_report.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Tumor vs Normal PTM Analysis - Overall Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total datasets analyzed: {len(successful_analyses)}\n")
            f.write(f"Failed analyses: {len(failed_analyses)}\n")
            f.write(f"Analysis method: log₂(PTM median / WT median) ratio comparison\n")
            f.write(f"Statistical test: Mann-Whitney U test\n")
            f.write(f"PTMs analyzed: {', '.join(self.target_modifications[:3])}\n\n")

            f.write("SUCCESSFUL ANALYSES\n")
            f.write("-" * 20 + "\n")
            for dataset_id in successful_analyses:
                dataset_info = next(d for d in all_datasets if d['dataset_id'] == dataset_id)
                f.write(f"{dataset_id}: Tumor={dataset_info['tumor_count']}, Normal={dataset_info['normal_count']}\n")

            if failed_analyses:
                f.write(f"\nFAILED ANALYSES\n")
                f.write("-" * 20 + "\n")
                for dataset_id, error in failed_analyses:
                    f.write(f"{dataset_id}: {error}\n")

            f.write(f"\nRESULTS LOCATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Individual results: tumor_vs_normal_ptm_results/[DATASET_ID]/\n")
            f.write(f"Overall report: {report_file}\n")

        print(f"   保存总体报告: overall_tumor_vs_normal_report.txt")

    def create_dataset_level_summary(self, successful_analyses: list):
        """创建数据集级别的汇总分析"""
        print("📊 创建数据集级别的汇总分析...")

        # 收集所有数据集的结果
        all_dataset_results = []
        target_ptms = ['Phospho[S]', 'Acetyl[K]', 'Deamidated[N]']

        for dataset_id in successful_analyses:
            dataset_dir = f'tumor_vs_normal_ptm_results/{dataset_id}'

            for ptm in target_ptms:
                stats_file = os.path.join(dataset_dir, f'stats_{ptm}.txt')
                summary_file = os.path.join(dataset_dir, f'summary_{ptm}.csv')

                if os.path.exists(stats_file) and os.path.exists(summary_file):
                    # 读取统计结果
                    stats_data = self.parse_stats_file(stats_file, dataset_id, ptm)

                    # 读取样本数量信息
                    summary_df = pd.read_csv(summary_file)
                    sample_counts = summary_df.groupby('sample_type').size()

                    for stat_result in stats_data:
                        stat_result.update({
                            'dataset_id': dataset_id,
                            'ptm': ptm,
                            'tumor_samples': sample_counts.get('Tumor', 0),
                            'normal_samples': sample_counts.get('Normal', 0)
                        })
                        all_dataset_results.append(stat_result)

        if not all_dataset_results:
            print("   ⚠️  没有找到可汇总的结果")
            return

        # 创建汇总数据框
        summary_df = pd.DataFrame(all_dataset_results)

        # 保存数据集级别汇总
        summary_file = os.path.join(self.output_dir, 'dataset_level_summary.csv')
        summary_df.to_csv(summary_file, index=False)

        # 创建汇总可视化
        self.create_dataset_level_visualizations(summary_df)

        # 生成汇总报告
        self.generate_dataset_level_report(summary_df)

        print(f"   保存数据集汇总: dataset_level_summary.csv")

    def parse_stats_file(self, stats_file: str, dataset_id: str, ptm: str) -> list:
        """解析统计结果文件"""
        results = []

        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 按分隔线分割每个property的结果
            sections = content.split('------------------------------')

            for section in sections:
                lines = [line.strip() for line in section.strip().split('\n') if line.strip()]

                if len(lines) < 5:  # 至少需要几个关键字段
                    continue

                result = {'dataset_id': dataset_id, 'ptm': ptm}

                # 解析各个字段
                for line in lines:
                    if ':' not in line or line.startswith('=') or line.startswith('Statistical'):
                        continue

                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if key == 'Property':
                        result['property'] = value
                    elif key == 'P-value':
                        try:
                            if value.lower() == 'nan':
                                result['p_value'] = None
                            else:
                                result['p_value'] = float(value)
                        except:
                            result['p_value'] = None
                    elif key == 'Adjusted P-value':
                        try:
                            if value.lower() == 'nan':
                                result['p_adj'] = None
                            else:
                                result['p_adj'] = float(value)
                        except:
                            result['p_adj'] = None
                    elif key == 'Significant':
                        result['significant'] = 'Yes' in value
                    elif key == "Cliff's Delta":
                        try:
                            if value.lower() == 'nan':
                                result['cliff_delta'] = None
                            else:
                                result['cliff_delta'] = float(value)
                        except:
                            result['cliff_delta'] = None
                    elif key == 'Tumor median':
                        try:
                            if value.lower() == 'nan':
                                result['tumor_median'] = None
                            else:
                                result['tumor_median'] = float(value)
                        except:
                            result['tumor_median'] = None
                    elif key == 'Normal median':
                        try:
                            if value.lower() == 'nan':
                                result['normal_median'] = None
                            else:
                                result['normal_median'] = float(value)
                        except:
                            result['normal_median'] = None
                    elif key == 'Sample sizes':
                        # 解析样本量信息: "Tumor=17, Normal=7"
                        if 'Tumor=' in value and 'Normal=' in value:
                            try:
                                tumor_part = value.split('Tumor=')[1].split(',')[0]
                                normal_part = value.split('Normal=')[1]
                                result['tumor_n'] = int(tumor_part.strip())
                                result['normal_n'] = int(normal_part.strip())
                            except:
                                pass

                # 只有包含property的记录才添加
                if 'property' in result:
                    results.append(result)

        except Exception as e:
            print(f"   ⚠️  解析统计文件失败 {stats_file}: {e}")

        return results

    def create_dataset_level_visualizations(self, summary_df: pd.DataFrame):
        """创建数据集级别的可视化"""
        print("   创建数据集级别可视化...")

        # 设置图表样式
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # 1. 显著性结果热图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ptms = ['Phospho[S]', 'Acetyl[K]', 'Deamidated[N]']
        properties = ['pI', 'Net_Charge', 'Hydrophobicity_KD', 'Molecular_Weight']

        for i, ptm in enumerate(ptms):
            if i >= 3:  # 只显示前3个PTM
                break

            ax = axes[i//2, i%2] if i < 2 else axes[1, 0]

            ptm_data = summary_df[summary_df['ptm'] == ptm]

            if len(ptm_data) == 0:
                ax.text(0.5, 0.5, f'No data for {ptm}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{ptm} - No Data')
                continue

            # 创建热图数据
            heatmap_data = ptm_data.pivot_table(
                index='dataset_id',
                columns='property',
                values='p_adj',
                fill_value=1.0
            )

            # 创建显著性标记
            sig_data = ptm_data.pivot_table(
                index='dataset_id',
                columns='property',
                values='significant',
                fill_value=False
            )

            # 绘制热图
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                       ax=ax, cbar_kws={'label': 'Adjusted P-value'})

            # 添加显著性标记
            for y, dataset in enumerate(heatmap_data.index):
                for x, prop in enumerate(heatmap_data.columns):
                    if prop in sig_data.columns and dataset in sig_data.index:
                        if sig_data.loc[dataset, prop]:
                            ax.text(x+0.5, y+0.5, '*', ha='center', va='center',
                                   color='white', fontsize=16, fontweight='bold')

            ax.set_title(f'{ptm} Significance Across Datasets')
            ax.set_xlabel('Physicochemical Properties')
            ax.set_ylabel('Dataset ID')

        # 隐藏第4个子图
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dataset_level_significance_heatmap.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 效应量分布图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, ptm in enumerate(ptms):
            ptm_data = summary_df[summary_df['ptm'] == ptm]

            if len(ptm_data) == 0:
                continue

            # 计算log2 fold change
            ptm_data = ptm_data.copy()
            ptm_data['log2_fc'] = ptm_data.apply(
                lambda row: np.log2(row['tumor_median'] / row['normal_median'])
                if pd.notna(row['tumor_median']) and pd.notna(row['normal_median']) and row['normal_median'] != 0
                else np.nan, axis=1
            )

            # 按property分组绘制
            for prop in properties:
                prop_data = ptm_data[ptm_data['property'] == prop]
                if len(prop_data) > 0:
                    x_pos = [j + i*0.2 for j in range(len(prop_data))]
                    colors = ['red' if sig else 'gray' for sig in prop_data['significant']]

                    axes[i].scatter(x_pos, prop_data['log2_fc'],
                                  c=colors, alpha=0.7, s=60, label=prop)

            axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[i].set_title(f'{ptm} Effect Sizes')
            axes[i].set_xlabel('Datasets')
            axes[i].set_ylabel('log2(Tumor/Normal)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dataset_level_effect_sizes.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 样本量分布图
        plt.figure(figsize=(12, 8))

        # 计算每个数据集的总样本量
        dataset_samples = summary_df.groupby('dataset_id').agg({
            'tumor_samples': 'first',
            'normal_samples': 'first'
        }).reset_index()

        dataset_samples['total_samples'] = dataset_samples['tumor_samples'] + dataset_samples['normal_samples']
        dataset_samples = dataset_samples.sort_values('total_samples')

        # 创建堆叠条形图
        x_pos = range(len(dataset_samples))
        plt.bar(x_pos, dataset_samples['tumor_samples'], label='Tumor Samples', color='red', alpha=0.7)
        plt.bar(x_pos, dataset_samples['normal_samples'],
               bottom=dataset_samples['tumor_samples'], label='Normal Samples', color='blue', alpha=0.7)

        plt.xlabel('Dataset ID')
        plt.ylabel('Number of Samples')
        plt.title('Sample Distribution Across Datasets')
        plt.xticks(x_pos, dataset_samples['dataset_id'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 添加总数标签
        for i, (_, row) in enumerate(dataset_samples.iterrows()):
            plt.text(i, row['total_samples'] + 5, str(row['total_samples']),
                    ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dataset_sample_distribution.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"      保存可视化: dataset_level_significance_heatmap.png")
        print(f"      保存可视化: dataset_level_effect_sizes.png")
        print(f"      保存可视化: dataset_sample_distribution.png")

    def generate_dataset_level_report(self, summary_df: pd.DataFrame):
        """生成数据集级别的分析报告"""
        print("   生成数据集级别报告...")

        report_file = os.path.join(self.output_dir, 'dataset_level_analysis_report.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Dataset-Level Tumor vs Normal PTM Analysis Report\n")
            f.write("=" * 60 + "\n\n")

            # 总体统计
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total datasets analyzed: {summary_df['dataset_id'].nunique()}\n")
            f.write(f"Total PTMs analyzed: {summary_df['ptm'].nunique()}\n")
            f.write(f"Total comparisons: {len(summary_df)}\n")
            f.write(f"Significant comparisons: {summary_df['significant'].sum()}\n")
            f.write(f"Significance rate: {summary_df['significant'].mean()*100:.1f}%\n\n")

            # 按PTM统计
            f.write("PTM-SPECIFIC RESULTS\n")
            f.write("-" * 20 + "\n")

            for ptm in summary_df['ptm'].unique():
                ptm_data = summary_df[summary_df['ptm'] == ptm]
                sig_count = ptm_data['significant'].sum()
                total_count = len(ptm_data)

                f.write(f"\n{ptm}:\n")
                f.write(f"  Total comparisons: {total_count}\n")
                f.write(f"  Significant results: {sig_count}\n")
                f.write(f"  Significance rate: {sig_count/total_count*100:.1f}%\n")

                # 按property细分
                for prop in ptm_data['property'].unique():
                    prop_data = ptm_data[ptm_data['property'] == prop]
                    prop_sig = prop_data['significant'].sum()
                    prop_total = len(prop_data)

                    f.write(f"    {prop}: {prop_sig}/{prop_total} ({prop_sig/prop_total*100:.1f}%)\n")

            # 按数据集统计
            f.write(f"\nDATASET-SPECIFIC RESULTS\n")
            f.write("-" * 20 + "\n")

            for dataset in summary_df['dataset_id'].unique():
                dataset_data = summary_df[summary_df['dataset_id'] == dataset]
                sig_count = dataset_data['significant'].sum()
                total_count = len(dataset_data)

                # 获取样本量信息
                tumor_samples = dataset_data['tumor_samples'].iloc[0] if len(dataset_data) > 0 else 0
                normal_samples = dataset_data['normal_samples'].iloc[0] if len(dataset_data) > 0 else 0

                f.write(f"\n{dataset}:\n")
                f.write(f"  Sample sizes: Tumor={tumor_samples}, Normal={normal_samples}\n")
                f.write(f"  Total comparisons: {total_count}\n")
                f.write(f"  Significant results: {sig_count}\n")
                f.write(f"  Significance rate: {sig_count/total_count*100:.1f}%\n")

                # 显著性结果详情
                sig_results = dataset_data[dataset_data['significant'] == True]
                if len(sig_results) > 0:
                    f.write(f"  Significant findings:\n")
                    for _, row in sig_results.iterrows():
                        direction = "↑" if row['tumor_median'] > row['normal_median'] else "↓"
                        f.write(f"    {row['ptm']} - {row['property']}: p_adj={row['p_adj']:.4f} {direction}\n")

            # 跨数据集一致性分析
            f.write(f"\nCROSS-DATASET CONSISTENCY\n")
            f.write("-" * 20 + "\n")

            # 计算每个PTM-property组合在多少个数据集中显著
            consistency_results = []

            for ptm in summary_df['ptm'].unique():
                for prop in summary_df['property'].unique():
                    subset = summary_df[(summary_df['ptm'] == ptm) & (summary_df['property'] == prop)]
                    if len(subset) > 0:
                        sig_datasets = subset[subset['significant'] == True]['dataset_id'].tolist()
                        total_datasets = len(subset)
                        consistency_rate = len(sig_datasets) / total_datasets

                        consistency_results.append({
                            'ptm': ptm,
                            'property': prop,
                            'significant_datasets': len(sig_datasets),
                            'total_datasets': total_datasets,
                            'consistency_rate': consistency_rate,
                            'datasets': sig_datasets
                        })

            # 按一致性排序
            consistency_results.sort(key=lambda x: x['consistency_rate'], reverse=True)

            f.write("Most consistent findings across datasets:\n")
            for result in consistency_results[:10]:  # 显示前10个
                if result['consistency_rate'] > 0:
                    f.write(f"  {result['ptm']} - {result['property']}: ")
                    f.write(f"{result['significant_datasets']}/{result['total_datasets']} datasets ")
                    f.write(f"({result['consistency_rate']*100:.1f}%)\n")
                    if result['datasets']:
                        f.write(f"    Datasets: {', '.join(result['datasets'])}\n")

            # 效应量分析
            f.write(f"\nEFFECT SIZE ANALYSIS\n")
            f.write("-" * 20 + "\n")

            # 计算平均效应量
            for ptm in summary_df['ptm'].unique():
                ptm_data = summary_df[summary_df['ptm'] == ptm]

                if 'cliff_delta' in ptm_data.columns:
                    mean_effect = ptm_data['cliff_delta'].mean()
                    f.write(f"{ptm} average effect size (Cliff's delta): {mean_effect:.3f}\n")

                    # 按property分析
                    for prop in ptm_data['property'].unique():
                        prop_data = ptm_data[ptm_data['property'] == prop]
                        prop_effect = prop_data['cliff_delta'].mean()
                        f.write(f"  {prop}: {prop_effect:.3f}\n")

            f.write(f"\nFILES GENERATED\n")
            f.write("-" * 20 + "\n")
            f.write(f"- dataset_level_summary.csv: Raw data for all comparisons\n")
            f.write(f"- dataset_level_significance_heatmap.png: Significance patterns\n")
            f.write(f"- dataset_level_effect_sizes.png: Effect size distributions\n")
            f.write(f"- dataset_sample_distribution.png: Sample size information\n")
            f.write(f"- dataset_level_analysis_report.txt: This report\n")

        print(f"      保存报告: dataset_level_analysis_report.txt")

def main():
    """主函数"""
    print("🧬 肿瘤 vs 正常样本 PTM 效应比较分析")
    print("=" * 60)

    # 创建分析器
    analyzer = TumorNormalPTMAnalyzer()

    # 可以指定特定的PTM进行分析
    target_ptms = ['Phospho[S]', 'Acetyl[K]', 'Deamidated[N]']

    try:
        # 分析所有包含Tumor和Normal样本的数据集
        analyzer.analyze_all_datasets(target_ptms)

        print("\n🎯 分析完成！")
        print("📊 查看结果:")
        print(f"  - 各数据集结果: tumor_vs_normal_ptm_results/[DATASET_ID]/")
        print(f"  - 总体报告: tumor_vs_normal_ptm_results/overall_tumor_vs_normal_report.txt")

    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
