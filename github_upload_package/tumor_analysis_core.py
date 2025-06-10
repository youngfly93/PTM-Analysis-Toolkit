#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肿瘤蛋白质组学修饰肽理化性质分析 - 核心版本

专注于数据分析，可选择性生成可视化
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 尝试导入tqdm进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 导入我们的肽段分析器
from peptide_properties_analyzer import PeptidePropertiesAnalyzer

# 设置日志 - 只显示INFO及以上级别，过滤WARNING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置peptide_properties_analyzer的日志级别为ERROR，避免WARNING信息
peptide_logger = logging.getLogger('peptide_properties_analyzer')
peptide_logger.setLevel(logging.ERROR)

class TumorAnalysisCore:
    """肿瘤蛋白质组学分析核心"""
    
    def __init__(self):
        self.analyzer = PeptidePropertiesAnalyzer()

        # 设置静默模式，减少日志输出
        # 注意：PeptidePropertiesAnalyzer可能没有logger属性，所以我们在模块级别设置
        
        # 目标修饰类型
        self.target_modifications = {
            'Oxidation[M]', 'Acetyl[K]', 'Phospho[S]', 'Phospho[T]', 'Phospho[Y]',
            'Deamidated[N]', 'Deamidated[Q]', 'Methyl[K]', 'Dimethyl[K]', 'Trimethyl[K]'
        }
        
        # 修饰类型的中文名称映射
        self.mod_name_mapping = {
            'Oxidation[M]': '甲硫氨酸氧化',
            'Acetyl[K]': '赖氨酸乙酰化',
            'Phospho[S]': '丝氨酸磷酸化',
            'Phospho[T]': '苏氨酸磷酸化',
            'Phospho[Y]': '酪氨酸磷酸化',
            'Deamidated[N]': '天冬酰胺脱酰胺',
            'Deamidated[Q]': '谷氨酰胺脱酰胺',
            'Methyl[K]': '赖氨酸甲基化',
            'Dimethyl[K]': '赖氨酸二甲基化',
            'Trimethyl[K]': '赖氨酸三甲基化'
        }
    
    def load_cancer_datasets(self) -> List[str]:
        """加载癌症数据集列表"""
        logger.info("加载癌症数据集...")
        
        try:
            meta_df = pd.read_csv('meta.txt', sep='\t')
            cancer_datasets = meta_df[meta_df['分析场景'] == 'Cancer']['all_accession'].tolist()
            logger.info(f"找到 {len(cancer_datasets)} 个癌症数据集")
            return cancer_datasets
        except Exception as e:
            logger.error(f"加载元数据失败: {e}")
            return []
    
    def get_tumor_samples(self, dataset_id: str) -> List[str]:
        """获取数据集的肿瘤样本列表"""
        summary_file = f"tumor_summary/{dataset_id}_summary.csv"
        
        if not os.path.exists(summary_file):
            return []
        
        try:
            summary_df = pd.read_csv(summary_file)
            tumor_samples = summary_df[summary_df['Type'] == 'Tumor']
            return tumor_samples['File Name'].tolist()
        except Exception as e:
            logger.error(f"读取肿瘤摘要失败 {dataset_id}: {e}")
            return []
    
    def find_spectra_file(self, dataset_id: str, file_name: str) -> Optional[str]:
        """查找对应的.spectra文件"""
        dataset_dir = f"{dataset_id}_human"
        
        if not os.path.exists(dataset_dir):
            return None
        
        # 直接查找文件
        file_path = os.path.join(dataset_dir, file_name)
        if os.path.exists(file_path):
            return file_path
        
        # 如果直接查找失败，尝试模糊匹配
        for file in os.listdir(dataset_dir):
            if file.endswith('.spectra') and file_name.replace('.spectra', '') in file:
                return os.path.join(dataset_dir, file)
        
        return None
    
    def extract_modification_type(self, modification_string: str) -> Optional[str]:
        """提取修饰类型"""
        if pd.isna(modification_string) or modification_string == '':
            return None
        
        # 解析修饰字符串
        mod_parts = modification_string.split(';')
        for part in mod_parts:
            if not part.strip():
                continue
            
            if ',' in part:
                mod_name = part.split(',')[1].strip()
                if mod_name in self.target_modifications:
                    return mod_name
        
        return None
    
    def analyze_single_file(self, spectra_file: str, dataset_id: str) -> Optional[pd.DataFrame]:
        """分析单个.spectra文件"""
        print(f"[FILE] 分析文件: {os.path.basename(spectra_file)}")

        try:
            # 使用肽段分析器处理文件
            result_df = self.analyzer.process_spectra_file(spectra_file)

            if result_df is None or len(result_df) == 0:
                return None

            # 添加数据集信息
            result_df['dataset_id'] = dataset_id
            result_df['file_name'] = os.path.basename(spectra_file)

            # 提取修饰类型（添加进度条）
            print("[SEARCH] 提取修饰类型...")
            if TQDM_AVAILABLE and len(result_df) > 1000:
                tqdm.pandas(desc="提取修饰")
                result_df['modification_type'] = result_df['modifications'].progress_apply(
                    self.extract_modification_type)
            else:
                result_df['modification_type'] = result_df['modifications'].apply(
                    self.extract_modification_type)

            # 只保留目标修饰和无修饰的肽段
            target_data = result_df[
                (result_df['modification_type'].isin(self.target_modifications)) |
                (result_df['num_modifications'] == 0)
            ].copy()

            print(f"[OK] 提取到 {len(target_data)} 条目标数据")
            return target_data

        except Exception as e:
            logger.error(f"分析文件失败 {spectra_file}: {e}")
            return None
    
    def calculate_modification_effects(self, dataset_data: pd.DataFrame, dataset_id: str) -> List[Dict]:
        """计算修饰效应"""
        logger.info(f"计算数据集 {dataset_id} 的修饰效应...")
        
        effects = []
        
        # 获取无修饰肽段作为对照
        unmodified_data = dataset_data[dataset_data['num_modifications'] == 0]
        
        if len(unmodified_data) == 0:
            logger.warning(f"数据集 {dataset_id}: 无无修饰肽段数据")
            return effects
        
        # 计算无修饰肽段的理化性质统计
        unmod_stats = {
            'pi_median': unmodified_data['corrected_pi'].median(),
            'charge_median': unmodified_data['corrected_charge_at_ph7'].median(),
            'hydro_median': unmodified_data['corrected_hydrophobicity_kd'].median(),
            'mass_median': unmodified_data['corrected_molecular_weight'].median()
        }
        
        # 分析每种修饰类型
        for mod_type in self.target_modifications:
            modified_data = dataset_data[dataset_data['modification_type'] == mod_type]
            
            if len(modified_data) < 10:  # 至少需要10个肽段
                continue
            
            # 计算修饰肽段的理化性质统计
            mod_stats = {
                'pi_median': modified_data['corrected_pi'].median(),
                'charge_median': modified_data['corrected_charge_at_ph7'].median(),
                'hydro_median': modified_data['corrected_hydrophobicity_kd'].median(),
                'mass_median': modified_data['corrected_molecular_weight'].median()
            }
            
            # 计算差异
            effect = {
                'dataset_id': dataset_id,
                'modification_type': mod_type,
                'modification_name': self.mod_name_mapping.get(mod_type, mod_type),
                'n_modified': len(modified_data),
                'n_unmodified': len(unmodified_data),
                'pi_diff': mod_stats['pi_median'] - unmod_stats['pi_median'],
                'charge_diff': mod_stats['charge_median'] - unmod_stats['charge_median'],
                'hydro_diff': mod_stats['hydro_median'] - unmod_stats['hydro_median'],
                'mass_diff': mod_stats['mass_median'] - unmod_stats['mass_median']
            }
            
            effects.append(effect)
            logger.info(f"  {mod_type}: {len(modified_data)} 个修饰肽段")
        
        return effects
    
    def analyze_dataset(self, dataset_id: str, max_files: int = None) -> List[Dict]:
        """分析单个数据集"""
        print(f"\n[DATASET] 开始分析数据集: {dataset_id}")

        # 获取肿瘤样本文件
        tumor_files = self.get_tumor_samples(dataset_id)

        if len(tumor_files) == 0:
            print(f"[WARNING]  数据集 {dataset_id}: 无肿瘤样本")
            return []

        if max_files:
            tumor_files = tumor_files[:max_files]

        print(f"[FILES] 找到 {len(tumor_files)} 个肿瘤样本文件")

        # 分析每个文件（添加进度条）
        all_data = []

        if TQDM_AVAILABLE:
            file_iterator = tqdm(tumor_files, desc=f"处理{dataset_id}文件", unit="文件")
        else:
            file_iterator = tumor_files

        for file_name in file_iterator:
            spectra_file = self.find_spectra_file(dataset_id, file_name)

            if spectra_file is None:
                if not TQDM_AVAILABLE:
                    print(f"[WARNING]  未找到文件: {file_name}")
                continue

            file_data = self.analyze_single_file(spectra_file, dataset_id)
            if file_data is not None and len(file_data) > 0:
                all_data.append(file_data)

        if len(all_data) == 0:
            print(f"[ERROR] 数据集 {dataset_id}: 无有效数据")
            return []

        # 合并所有文件的数据
        print("[RESUME] 合并数据...")
        dataset_data = pd.concat(all_data, ignore_index=True)
        print(f"[OK] 数据集 {dataset_id}: 总共 {len(dataset_data)} 条数据")

        # 计算修饰效应
        print("[PLOT] 计算修饰效应...")
        effects = self.calculate_modification_effects(dataset_data, dataset_id)

        return effects
    
    def run_analysis(self, max_datasets: int = None, max_files_per_dataset: int = None):
        """运行完整分析"""
        print("[ANALYSIS] 开始肿瘤蛋白质组学修饰肽分析...")
        print("=" * 60)

        # 加载癌症数据集
        cancer_datasets = self.load_cancer_datasets()

        if len(cancer_datasets) == 0:
            print("[ERROR] 未找到癌症数据集")
            return None

        if max_datasets:
            cancer_datasets = cancer_datasets[:max_datasets]
            print(f"[TARGET] 限制分析前 {max_datasets} 个数据集")

        print(f"[LIST] 总共需要分析 {len(cancer_datasets)} 个数据集")

        # 分析每个数据集（添加进度条）
        all_effects = []

        if TQDM_AVAILABLE:
            dataset_iterator = tqdm(cancer_datasets, desc="分析数据集", unit="数据集")
        else:
            dataset_iterator = cancer_datasets

        for dataset_id in dataset_iterator:
            if not TQDM_AVAILABLE:
                print(f"\n[PLOT] 处理数据集: {dataset_id}")

            try:
                effects = self.analyze_dataset(dataset_id, max_files_per_dataset)
                if len(effects) > 0:
                    all_effects.extend(effects)
                    if not TQDM_AVAILABLE:
                        print(f"[OK] 数据集 {dataset_id}: 计算得到 {len(effects)} 个修饰效应")
                else:
                    if not TQDM_AVAILABLE:
                        print(f"[WARNING]  数据集 {dataset_id}: 未计算出修饰效应")

            except Exception as e:
                if not TQDM_AVAILABLE:
                    print(f"[ERROR] 处理数据集 {dataset_id} 时出错: {e}")
                continue

        if len(all_effects) == 0:
            print("[ERROR] 没有有效的分析结果")
            return None

        # 转换为DataFrame
        print(f"\n[MERGE] 处理分析结果...")
        effects_df = pd.DataFrame(all_effects)
        print(f"[OK] 总共分析得到 {len(effects_df)} 个修饰效应")

        # 保存结果
        output_dir = 'tumor_analysis_results'
        os.makedirs(output_dir, exist_ok=True)

        print(f"[SAVE] 保存结果到: {output_dir}/")
        effects_df.to_csv(f'{output_dir}/modification_effects.csv', index=False, encoding='utf-8')

        # 生成统计报告
        print("[REPORT] 生成统计报告...")
        self.generate_summary_report(effects_df, output_dir)

        print("[SUCCESS] 分析完成！")
        return effects_df
    
    def generate_summary_report(self, effects_df: pd.DataFrame, output_dir: str):
        """生成统计报告"""
        logger.info("生成统计报告...")
        
        report_file = f'{output_dir}/tumor_analysis_summary.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("肿瘤蛋白质组学修饰肽理化性质分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 总体统计
            f.write("总体统计:\n")
            f.write(f"  分析数据集数量: {effects_df['dataset_id'].nunique()}\n")
            f.write(f"  分析修饰类型数量: {effects_df['modification_type'].nunique()}\n")
            f.write(f"  总分析记录数: {len(effects_df)}\n\n")
            
            # 修饰类型统计
            f.write("修饰类型统计:\n")
            for mod_type in self.target_modifications:
                mod_data = effects_df[effects_df['modification_type'] == mod_type]
                if len(mod_data) > 0:
                    mod_name = self.mod_name_mapping.get(mod_type, mod_type)
                    f.write(f"\n  {mod_name} ({mod_type}):\n")
                    f.write(f"    出现在 {len(mod_data)} 个数据集中\n")
                    f.write(f"    总修饰肽段数: {mod_data['n_modified'].sum()}\n")
                    f.write(f"    平均pI差异: {mod_data['pi_diff'].mean():.3f}\n")
                    f.write(f"    平均电荷差异: {mod_data['charge_diff'].mean():.3f}\n")
                    f.write(f"    平均疏水性差异: {mod_data['hydro_diff'].mean():.3f}\n")
                    f.write(f"    平均质量差异: {mod_data['mass_diff'].mean():.3f} Da\n")
            
            # 显著效应分析
            f.write("\n\n显著修饰效应分析:\n")
            
            significant_effects = effects_df[
                (abs(effects_df['pi_diff']) > 0.5) |
                (abs(effects_df['charge_diff']) > 0.5) |
                (abs(effects_df['hydro_diff']) > 1.0)
            ]
            
            if len(significant_effects) > 0:
                f.write(f"\n  显著效应 (pI>0.5 或 电荷>0.5 或 疏水性>1.0):\n")
                for _, row in significant_effects.iterrows():
                    f.write(f"    {row['dataset_id']} - {row['modification_name']}:\n")
                    f.write(f"      pI差异: {row['pi_diff']:.3f}\n")
                    f.write(f"      电荷差异: {row['charge_diff']:.3f}\n")
                    f.write(f"      疏水性差异: {row['hydro_diff']:.3f}\n")
        
        logger.info(f"统计报告已保存到: {report_file}")


def main():
    """主函数"""
    print("肿瘤蛋白质组学修饰肽理化性质分析 - 核心版本")
    print("=" * 60)
    
    # 创建分析器
    analyzer = TumorAnalysisCore()
    
    # 运行分析（可以设置限制用于测试）
    # results = analyzer.run_analysis(max_datasets=2, max_files_per_dataset=3)  # 测试用
    results = analyzer.run_analysis()  # 完整分析
    
    if results is not None:
        print(f"\n分析完成！共分析了 {len(results)} 个修饰效应")
        print("结果文件保存在 'tumor_analysis_results' 目录中")
        print("\n生成的文件:")
        print("- modification_effects.csv: 详细的修饰效应数据")
        print("- tumor_analysis_summary.txt: 统计报告")
        
        # 显示一些关键统计
        print(f"\n关键统计:")
        print(f"- 分析数据集数量: {results['dataset_id'].nunique()}")
        print(f"- 发现修饰类型数量: {results['modification_type'].nunique()}")
        print(f"- 总修饰肽段数: {results['n_modified'].sum()}")
        
    else:
        print("分析失败，请检查数据和日志")


if __name__ == "__main__":
    main()
