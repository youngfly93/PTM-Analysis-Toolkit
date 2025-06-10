#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肽段理化性质分析脚本
使用Pyteomics计算肽段的PI、charge和疏水性，支持修饰肽段分析

作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np
import re
import os
import glob
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Pyteomics imports
try:
    from pyteomics import electrochem, mass, auxiliary, parser
    from pyteomics.mass import calculate_mass
    from pyteomics.electrochem import charge, pI
    from pyteomics.auxiliary import PyteomicsError
    from pyteomics.parser import parse
except ImportError as e:
    print("请安装Pyteomics: pip install pyteomics")
    raise e

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PeptidePropertiesAnalyzer:
    """肽段理化性质分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.modification_table = self._load_modification_table()
        self.hydrophobicity_scales = {
            'kyte_doolittle': {
                'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
            },
            'eisenberg': {
                'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
                'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
                'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
                'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
            }
        }
        
    def _load_modification_table(self) -> Dict:
        """加载修饰表信息，包含pKa和疏水性数据"""
        # 基于您提供的flat_file信息创建修饰表
        modifications = {
            'Oxidation[M]': {
                'target_aa': 'M',
                'new_pka': None,  # 氧化不改变pKa
                'new_hydro_kd': 1.13,  # 1.9 + (-0.77) = 修饰后的绝对值
                'new_hydro_ei': -0.13,  # 0.64 + (-0.77)
                'unimod_id': 'ox',
                'delta_mass': 15.995
            },
            'Acetyl[K]': {
                'target_aa': 'K',
                'new_pka': None,  # 乙酰化移除正电荷，位点失活
                'new_hydro_kd': -7.96,  # -3.9 + (-4.06)
                'new_hydro_ei': -5.56,  # -1.50 + (-4.06)
                'unimod_id': 'ac',
                'delta_mass': 42.011
            },
            'Phospho[S]': {
                'target_aa': 'S',
                'new_pka': 6.7,  # 磷酸化的pKa
                'new_hydro_kd': -1.61,  # -0.8 + (-0.81)
                'new_hydro_ei': -0.99,  # -0.18 + (-0.81)
                'unimod_id': 'ph',
                'delta_mass': 79.966
            },
            'Phospho[T]': {
                'target_aa': 'T',
                'new_pka': 6.7,
                'new_hydro_kd': -2.20,  # -0.7 + (-1.5)
                'new_hydro_ei': -1.55,  # -0.05 + (-1.5)
                'unimod_id': 'ph',
                'delta_mass': 79.966
            },
            'Phospho[Y]': {
                'target_aa': 'Y',
                'new_pka': 6.7,
                'new_hydro_kd': -2.64,  # -1.3 + (-1.34)
                'new_hydro_ei': -1.08,  # 0.26 + (-1.34)
                'unimod_id': 'ph',
                'delta_mass': 79.966
            },
            'Deamidated[N]': {
                'target_aa': 'N',
                'new_pka': None,
                'new_hydro_kd': -5.10,  # -3.5 + (-1.6)
                'new_hydro_ei': -2.38,  # -0.78 + (-1.6)
                'unimod_id': 'deam',
                'delta_mass': 0.984
            },
            'Deamidated[Q]': {
                'target_aa': 'Q',
                'new_pka': None,
                'new_hydro_kd': -3.89,  # -3.5 + (-0.39)
                'new_hydro_ei': -1.24,  # -0.85 + (-0.39)
                'unimod_id': 'deam',
                'delta_mass': 0.984
            },
            'Methyl[K]': {
                'target_aa': 'K',
                'new_pka': 10.55,  # 10.5 + 0.05
                'new_hydro_kd': -7.66,  # -3.9 + (-3.76)
                'new_hydro_ei': -5.26,  # -1.50 + (-3.76)
                'unimod_id': 'me1',
                'delta_mass': 14.016
            },
            'Dimethyl[K]': {
                'target_aa': 'K',
                'new_pka': 9.95,  # 10.5 - 0.55
                'new_hydro_kd': -7.86,  # -3.9 + (-3.96)
                'new_hydro_ei': -5.46,  # -1.50 + (-3.96)
                'unimod_id': 'me2',
                'delta_mass': 28.031
            },
            'Trimethyl[K]': {
                'target_aa': 'K',
                'new_pka': 9.50,  # 10.5 - 1.00
                'new_hydro_kd': -7.96,  # -3.9 + (-4.06)
                'new_hydro_ei': -5.56,  # -1.50 + (-4.06)
                'unimod_id': 'me3',
                'delta_mass': 42.047
            },
            # 其他修饰保持原有疏水性值
            'Gln->pyro-Glu[AnyN-termQ]': {
                'target_aa': 'Q',
                'new_pka': None,
                'new_hydro_kd': -3.5,
                'new_hydro_ei': -0.85
            },
            'Amidated[AnyC-term]': {
                'target_aa': '*',
                'new_pka': None,
                'new_hydro_kd': 0,
                'new_hydro_ei': 0
            },
            'Cysteinyl[C]': {
                'target_aa': 'C',
                'new_pka': None,
                'new_hydro_kd': 2.5,
                'new_hydro_ei': 0.29
            },
            'Met-loss+Acetyl[ProteinN-termM]': {
                'target_aa': 'M',
                'new_pka': None,
                'new_hydro_kd': 1.9,
                'new_hydro_ei': 0.64
            },
            'Acetyl[ProteinN-term]': {
                'target_aa': '*',
                'new_pka': None,
                'new_hydro_kd': 0,
                'new_hydro_ei': 0
            },
            'Dehydrated[S]': {
                'target_aa': 'S',
                'new_pka': None,
                'new_hydro_kd': -0.8 + 0.5,
                'new_hydro_ei': -0.18 + 0.5
            },
            'Glutathione[C]': {
                'target_aa': 'C',
                'new_pka': None,
                'new_hydro_kd': 2.5 - 2.0,
                'new_hydro_ei': 0.29 - 2.0
            },
            # 添加常见的修饰类型以减少警告
            'Carbamidomethyl[C]': {
                'target_aa': 'C',
                'new_pka': None,
                'new_hydro_kd': 2.5,  # 保持原有疏水性
                'new_hydro_ei': 0.29,
                'unimod_id': 'cam',
                'delta_mass': 57.021
            },
            'BDMAPP[Y]': {
                'target_aa': 'Y',
                'new_pka': None,
                'new_hydro_kd': -1.3,  # 保持原有疏水性
                'new_hydro_ei': 0.26,
                'unimod_id': 'bdmapp',
                'delta_mass': 0.0  # 未知质量变化
            }
        }
        return modifications

    def build_modified_sequence(self, sequence: str, modifications: List[Tuple[int, str]]) -> str:
        """
        构建包含修饰的序列字符串，用于Pyteomics计算

        Args:
            sequence: 原始肽段序列
            modifications: 修饰列表 [(position, mod_name), ...]

        Returns:
            修饰序列字符串，如 "PEP(Phospho)TIDE"
        """
        if not modifications:
            return sequence

        # 按位置排序修饰（从后往前处理避免位置偏移）
        sorted_mods = sorted(modifications, key=lambda x: x[0], reverse=True)

        mod_seq = list(sequence)

        for position, mod_name in sorted_mods:
            # 位置从1开始，转换为0开始的索引
            idx = position - 1
            if 0 <= idx < len(mod_seq):
                # 获取UniMod标签用于Pyteomics
                unimod_tag = self._get_unimod_tag(mod_name)
                if unimod_tag:
                    mod_seq[idx] = f"{mod_seq[idx]}({unimod_tag})"

        return ''.join(mod_seq)

    def _get_unimod_tag(self, mod_name: str) -> str:
        """获取修饰的UniMod标签，用于Pyteomics解析"""
        if mod_name in self.modification_table:
            return self.modification_table[mod_name].get('unimod_id', '')
        return ''

    def build_custom_pka_dict(self, sequence: str, modifications: List[Tuple[int, str]]) -> Dict:
        """
        构建自定义pKa字典，用于修饰肽段的pI和电荷计算

        Args:
            sequence: 肽段序列
            modifications: 修饰列表

        Returns:
            自定义pKa字典，兼容Pyteomics格式
        """
        try:
            # 尝试获取Pyteomics的默认pKa字典
            from pyteomics.electrochem import pKa as default_pka_dict
            custom_pka = default_pka_dict.copy()
        except (ImportError, AttributeError):
            # 如果无法获取，使用我们的默认值
            custom_pka = {
                'K': 10.5, 'R': 12.5, 'H': 6.0,
                'D': 3.9, 'E': 4.3, 'C': 8.3, 'Y': 10.1,
                'N-term': 9.6, 'C-term': 2.3
            }

        # 添加修饰的pKa值
        for position, mod_name in modifications:
            if mod_name in self.modification_table:
                mod_info = self.modification_table[mod_name]
                target_aa = mod_info['target_aa']
                new_pka = mod_info['new_pka']

                # 位置从1开始，转换为0开始的索引
                idx = position - 1
                if 0 <= idx < len(sequence) and sequence[idx] == target_aa:
                    if 'Phospho' in mod_name:
                        # 磷酸化：添加磷酸基团的pKa
                        custom_pka[(target_aa, 'Phospho')] = 6.7
                    elif 'Acetyl' in mod_name and target_aa == 'K':
                        # 乙酰化：移除赖氨酸的pKa（通过极高pKa近似失活）
                        custom_pka[(target_aa, 'Acetyl')] = 99.0
                    elif new_pka is not None:
                        # 其他修饰：修改原有pKa
                        custom_pka[(target_aa, mod_name.split('[')[0])] = new_pka

        return custom_pka

    def calculate_modified_pi_and_charge(self, sequence: str, modifications: List[Tuple[int, str]], ph: float = 7.0) -> Tuple[float, float]:
        """
        计算修饰肽段的pI和电荷（简化实现）

        Args:
            sequence: 肽段序列
            modifications: 修饰列表
            ph: pH值

        Returns:
            (pI, charge_at_ph)
        """
        # 基础pKa值
        pka_values = {
            'K': 10.5, 'R': 12.5, 'H': 6.0,  # 碱性
            'D': 3.9, 'E': 4.3, 'C': 8.3, 'Y': 10.1,  # 酸性
            'N_term': 9.6, 'C_term': 2.3  # 末端
        }

        # 收集所有电离基团
        ionizable_groups = []

        # N端
        ionizable_groups.append(('N_term', pka_values['N_term'], 1))  # (name, pKa, charge_when_protonated)

        # C端
        ionizable_groups.append(('C_term', pka_values['C_term'], -1))  # (name, pKa, charge_when_deprotonated)

        # 创建修饰位置映射
        mod_map = {}
        for position, mod_name in modifications:
            mod_map[position - 1] = mod_name  # 转换为0开始的索引

        # 遍历序列中的电离基团
        for i, aa in enumerate(sequence):
            if aa in pka_values:
                current_pka = pka_values[aa]
                charge_sign = 1 if aa in ['K', 'R', 'H'] else -1

                # 检查是否有修饰
                if i in mod_map:
                    mod_name = mod_map[i]
                    if mod_name in self.modification_table:
                        mod_info = self.modification_table[mod_name]
                        new_pka = mod_info['new_pka']

                        if 'Phospho' in mod_name:
                            # 磷酸化：添加新的电离基团
                            ionizable_groups.append((f'Phospho_{i}', 6.7, -1))
                        elif 'Acetyl' in mod_name and aa == 'K':
                            # 乙酰化：移除赖氨酸的正电荷
                            continue  # 跳过这个基团
                        elif new_pka is not None:
                            # 其他修饰：改变pKa
                            current_pka = new_pka

                ionizable_groups.append((f'{aa}_{i}', current_pka, charge_sign))

        # 计算pI（简化的二分法）
        def calculate_charge_at_ph(ph_val):
            total_charge = 0.0
            for name, pka, charge_sign in ionizable_groups:
                if charge_sign > 0:  # 碱性基团
                    fraction_protonated = 1.0 / (1.0 + 10**(ph_val - pka))
                    total_charge += charge_sign * fraction_protonated
                else:  # 酸性基团
                    fraction_deprotonated = 1.0 / (1.0 + 10**(pka - ph_val))
                    total_charge += charge_sign * fraction_deprotonated
            return total_charge

        # 二分法求pI
        ph_low, ph_high = 0.0, 14.0
        for _ in range(50):  # 迭代50次应该足够精确
            ph_mid = (ph_low + ph_high) / 2.0
            charge = calculate_charge_at_ph(ph_mid)
            if abs(charge) < 0.001:  # 足够接近0
                break
            elif charge > 0:
                ph_low = ph_mid
            else:
                ph_high = ph_mid

        calculated_pi = (ph_low + ph_high) / 2.0
        charge_at_target_ph = calculate_charge_at_ph(ph)

        return calculated_pi, charge_at_target_ph

    def parse_modifications(self, mod_string) -> List[Tuple[int, str]]:
        """
        解析修饰字符串

        Args:
            mod_string: 修饰字符串，如 "11,Oxidation[M];16,Phospho[S];"

        Returns:
            List of (position, modification_name) tuples
        """
        modifications = []

        # 处理NaN值或空值
        if pd.isna(mod_string) or not mod_string or str(mod_string).strip() == '':
            return modifications

        # 确保是字符串类型
        mod_string = str(mod_string)
            
        # 分割多个修饰
        mod_parts = mod_string.strip().rstrip(';').split(';')
        
        for part in mod_parts:
            if not part.strip():
                continue
                
            # 解析位置和修饰名称（增强抗噪性，支持空格）
            match = re.match(r'\s*(\d+)\s*,\s*([^;]+)', part.strip())
            if match:
                position = int(match.group(1))
                mod_name = match.group(2).strip()
                modifications.append((position, mod_name))
            else:
                logger.warning(f"无法解析修饰: {part}")
                
        return modifications
    
    def calculate_hydrophobicity_with_modifications(self, sequence: str, modifications: List[Tuple[int, str]],
                                                   scale: str = 'kyte_doolittle') -> float:
        """
        计算包含修饰的肽段疏水性（按残基级别处理修饰）

        Args:
            sequence: 肽段序列
            modifications: 修饰列表
            scale: 疏水性标度 ('kyte_doolittle' 或 'eisenberg')

        Returns:
            疏水性指数
        """
        if scale not in self.hydrophobicity_scales:
            raise ValueError(f"不支持的疏水性标度: {scale}")

        scale_values = self.hydrophobicity_scales[scale]
        total_hydro = 0.0
        valid_residues = 0

        # 创建修饰位置映射
        mod_map = {}
        for position, mod_name in modifications:
            mod_map[position - 1] = mod_name  # 转换为0开始的索引

        for i, aa in enumerate(sequence):
            hydro_value = 0.0

            if i in mod_map:
                # 有修饰的残基，使用修饰后的疏水性
                mod_name = mod_map[i]
                if mod_name in self.modification_table:
                    mod_info = self.modification_table[mod_name]
                    if scale == 'kyte_doolittle':
                        hydro_value = mod_info['new_hydro_kd']
                    else:  # eisenberg
                        hydro_value = mod_info['new_hydro_ei']
                else:
                    # 未知修饰，使用原始值
                    hydro_value = scale_values.get(aa, 0)
                    logger.warning(f"未知修饰: {mod_name}")
            else:
                # 无修饰的残基，使用原始疏水性
                hydro_value = scale_values.get(aa, 0)
                if aa not in scale_values:
                    logger.warning(f"未知氨基酸: {aa}")

            total_hydro += hydro_value
            valid_residues += 1

        return total_hydro / valid_residues if valid_residues > 0 else 0.0

    def calculate_hydrophobicity(self, sequence: str, scale: str = 'kyte_doolittle') -> float:
        """
        计算基础肽段疏水性（无修饰）

        Args:
            sequence: 肽段序列
            scale: 疏水性标度 ('kyte_doolittle' 或 'eisenberg')

        Returns:
            疏水性指数
        """
        return self.calculate_hydrophobicity_with_modifications(sequence, [], scale)
    

    
    def calculate_peptide_properties(self, sequence: str, modifications: str = '',
                                   charge_state: int = None, ph: float = 7.0) -> Dict:
        """
        计算肽段理化性质（使用改进的Pyteomics方法）

        Args:
            sequence: 肽段序列
            modifications: 修饰字符串
            charge_state: 电荷态
            ph: pH值

        Returns:
            包含各种理化性质的字典
        """
        try:
            # 解析修饰
            mod_list = self.parse_modifications(modifications)

            # 基础计算（无修饰）
            base_molecular_weight = calculate_mass(sequence)
            base_pi = pI(sequence)
            base_charge_at_ph = charge(sequence, pH=ph)
            base_hydro_kd = self.calculate_hydrophobicity(sequence, 'kyte_doolittle')
            base_hydro_ei = self.calculate_hydrophobicity(sequence, 'eisenberg')

            # 修饰后的计算
            if mod_list:
                # 构建修饰序列用于质量和pI/电荷计算
                modified_sequence = self.build_modified_sequence(sequence, mod_list)

                # 使用修饰序列直接计算质量
                try:
                    corrected_molecular_weight = calculate_mass(modified_sequence)
                except Exception as e:
                    # 如果修饰序列解析失败，使用质量增量方法
                    logger.warning(f"修饰序列质量计算失败，使用增量方法: {e}")
                    corrected_molecular_weight = base_molecular_weight
                    for position, mod_name in mod_list:
                        if mod_name in self.modification_table:
                            mod_info = self.modification_table[mod_name]
                            if 'delta_mass' in mod_info:
                                corrected_molecular_weight += mod_info['delta_mass']

                # 使用Pyteomics计算pI和电荷
                try:
                    custom_pka = self.build_custom_pka_dict(sequence, mod_list)
                    corrected_pi = pI(modified_sequence, pKa=custom_pka)
                    corrected_charge_at_ph = charge(modified_sequence, pH=ph, pKa=custom_pka)
                except Exception as e:
                    # 如果Pyteomics计算失败，使用自定义方法
                    logger.warning(f"Pyteomics pI/电荷计算失败，使用自定义方法: {e}")
                    try:
                        corrected_pi, corrected_charge_at_ph = self.calculate_modified_pi_and_charge(
                            sequence, mod_list, ph)
                    except Exception as e2:
                        # 如果都失败，使用基础值
                        corrected_pi = base_pi
                        corrected_charge_at_ph = base_charge_at_ph
                        logger.warning(f"所有pI/电荷计算方法失败，使用基础值: {e2}")

                # 使用残基级别的疏水性计算
                corrected_hydro_kd = self.calculate_hydrophobicity_with_modifications(
                    sequence, mod_list, 'kyte_doolittle')
                corrected_hydro_ei = self.calculate_hydrophobicity_with_modifications(
                    sequence, mod_list, 'eisenberg')
            else:
                # 无修饰情况
                corrected_molecular_weight = base_molecular_weight
                corrected_pi = base_pi
                corrected_charge_at_ph = base_charge_at_ph
                corrected_hydro_kd = base_hydro_kd
                corrected_hydro_ei = base_hydro_ei

            return {
                'sequence': sequence,
                'length': len(sequence),
                'molecular_weight': base_molecular_weight,
                'corrected_molecular_weight': corrected_molecular_weight,
                'base_pi': base_pi,
                'corrected_pi': corrected_pi,
                'base_charge_at_ph7': base_charge_at_ph,
                'corrected_charge_at_ph7': corrected_charge_at_ph,
                'charge_state': charge_state,
                'base_hydrophobicity_kd': base_hydro_kd,
                'corrected_hydrophobicity_kd': corrected_hydro_kd,
                'base_hydrophobicity_ei': base_hydro_ei,
                'corrected_hydrophobicity_ei': corrected_hydro_ei,
                'modifications': modifications,
                'num_modifications': len(mod_list),
                'modification_list': '; '.join([f"{pos}:{mod}" for pos, mod in mod_list]),
                'modified_sequence': self.build_modified_sequence(sequence, mod_list) if mod_list else sequence
            }

        except Exception as e:
            logger.error(f"计算肽段性质时出错 {sequence}: {e}")
            return None
    
    def read_spectra_file(self, file_path: str) -> pd.DataFrame:
        """
        读取.spectra文件

        Args:
            file_path: 文件路径

        Returns:
            DataFrame containing peptide data
        """
        try:
            df = pd.read_csv(file_path, sep='\t')
            logger.info(f"成功读取文件: {file_path}, 共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return None

    def process_spectra_file(self, file_path: str, output_path: str = None) -> pd.DataFrame:
        """
        处理单个.spectra文件

        Args:
            file_path: 输入文件路径
            output_path: 输出文件路径

        Returns:
            处理后的DataFrame
        """
        # 读取文件
        df = self.read_spectra_file(file_path)
        if df is None:
            return None

        # 检查必要的列
        required_columns = ['Sequence', 'Charge', 'Modification']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"缺少必要的列: {missing_columns}")
            return None

        # 计算理化性质（使用itertuples提升性能）
        results = []
        total_peptides = len(df)

        # 获取列索引以提高性能
        col_indices = {col: idx for idx, col in enumerate(df.columns)}

        for idx, row in enumerate(df.itertuples(index=False, name=None)):
            if idx % 100 == 0:
                logger.info(f"处理进度: {idx}/{total_peptides}")

            # 通过索引访问列值
            sequence = row[col_indices['Sequence']]
            charge_state = row[col_indices['Charge']]
            modifications = row[col_indices.get('Modification', 0)] if 'Modification' in col_indices else ''

            # 处理修饰列的NaN值
            if pd.isna(modifications):
                modifications = ''

            # 计算性质
            properties = self.calculate_peptide_properties(
                sequence=sequence,
                modifications=modifications,
                charge_state=charge_state
            )

            if properties:
                # 添加原始数据（安全地获取列值）
                properties.update({
                    'file_name': row[col_indices.get('File_Name', 0)] if 'File_Name' in col_indices else '',
                    'scan_no': row[col_indices.get('Scan_No', 0)] if 'Scan_No' in col_indices else '',
                    'exp_mh': row[col_indices.get('Exp.MH+', 0)] if 'Exp.MH+' in col_indices else '',
                    'q_value': row[col_indices.get('Q-value', 0)] if 'Q-value' in col_indices else '',
                    'calc_mh': row[col_indices.get('Calc.MH+', 0)] if 'Calc.MH+' in col_indices else '',
                    'mass_shift': row[col_indices.get('Mass_Shift(Exp.-Calc.)', 0)] if 'Mass_Shift(Exp.-Calc.)' in col_indices else '',
                    'raw_score': row[col_indices.get('Raw_Score', 0)] if 'Raw_Score' in col_indices else '',
                    'final_score': row[col_indices.get('Final_Score', 0)] if 'Final_Score' in col_indices else '',
                    'proteins': row[col_indices.get('Proteins', 0)] if 'Proteins' in col_indices else '',
                    'target_decoy': row[col_indices.get('Target/Decoy', 0)] if 'Target/Decoy' in col_indices else ''
                })
                results.append(properties)

        # 创建结果DataFrame
        result_df = pd.DataFrame(results)

        # 保存结果
        if output_path:
            result_df.to_csv(output_path, index=False)
            logger.info(f"结果已保存到: {output_path}")

        return result_df

    def process_multiple_files(self, input_pattern: str, output_dir: str = "results") -> None:
        """
        批量处理多个.spectra文件

        Args:
            input_pattern: 输入文件模式，如 "*.spectra" 或 "PXD*/*.spectra"
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 查找匹配的文件
        files = glob.glob(input_pattern)
        if not files:
            logger.error(f"未找到匹配的文件: {input_pattern}")
            return

        logger.info(f"找到 {len(files)} 个文件待处理")

        all_results = []

        for file_path in files:
            logger.info(f"处理文件: {file_path}")

            # 生成输出文件名
            file_name = Path(file_path).stem
            output_path = os.path.join(output_dir, f"{file_name}_properties.csv")

            # 处理文件
            result_df = self.process_spectra_file(file_path, output_path)

            if result_df is not None:
                # 添加文件来源信息
                result_df['source_file'] = file_path
                all_results.append(result_df)

        # 合并所有结果
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_output = os.path.join(output_dir, "combined_peptide_properties.csv")
            combined_df.to_csv(combined_output, index=False)
            logger.info(f"合并结果已保存到: {combined_output}")

            # 生成统计报告
            self.generate_summary_report(combined_df, output_dir)

    def generate_summary_report(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        生成统计报告

        Args:
            df: 结果DataFrame
            output_dir: 输出目录
        """
        report_path = os.path.join(output_dir, "summary_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("肽段理化性质分析报告\n")
            f.write("=" * 50 + "\n\n")

            # 基本统计
            f.write(f"总肽段数: {len(df)}\n")
            f.write(f"唯一序列数: {df['sequence'].nunique()}\n")
            f.write(f"含修饰肽段数: {len(df[df['num_modifications'] > 0])}\n")
            f.write(f"修饰比例: {len(df[df['num_modifications'] > 0]) / len(df) * 100:.2f}%\n\n")

            # 长度分布
            f.write("肽段长度分布:\n")
            length_stats = df['length'].describe()
            for stat, value in length_stats.items():
                f.write(f"  {stat}: {value:.2f}\n")
            f.write("\n")

            # 分子量分布
            f.write("分子量分布 (Da):\n")
            mw_stats = df['molecular_weight'].describe()
            for stat, value in mw_stats.items():
                f.write(f"  {stat}: {value:.2f}\n")
            f.write("\n")

            # 等电点分布
            f.write("等电点分布:\n")
            pi_col = 'corrected_pi' if 'corrected_pi' in df.columns else 'corrected_pi_kd'
            pi_stats = df[pi_col].describe()
            for stat, value in pi_stats.items():
                f.write(f"  {stat}: {value:.2f}\n")
            f.write("\n")

            # 疏水性分布
            f.write("疏水性分布 (Kyte-Doolittle):\n")
            hydro_col = 'corrected_hydrophobicity_kd'
            if hydro_col in df.columns:
                hydro_stats = df[hydro_col].describe()
                for stat, value in hydro_stats.items():
                    f.write(f"  {stat}: {value:.2f}\n")
            f.write("\n")

            # 电荷分布
            f.write("电荷分布 (pH 7.0):\n")
            charge_col = 'corrected_charge_at_ph7' if 'corrected_charge_at_ph7' in df.columns else 'charge_at_ph7'
            charge_stats = df[charge_col].describe()
            for stat, value in charge_stats.items():
                f.write(f"  {stat}: {value:.2f}\n")
            f.write("\n")

            # 修饰统计
            if len(df[df['num_modifications'] > 0]) > 0:
                f.write("修饰统计:\n")
                mod_counts = df[df['num_modifications'] > 0]['num_modifications'].value_counts().sort_index()
                for num_mods, count in mod_counts.items():
                    f.write(f"  {num_mods}个修饰: {count} 肽段\n")
                f.write("\n")

        logger.info(f"统计报告已保存到: {report_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='肽段理化性质分析工具')
    parser.add_argument('input', help='输入文件或文件模式 (如 "*.spectra")')
    parser.add_argument('-o', '--output', default='results', help='输出目录 (默认: results)')
    parser.add_argument('-s', '--single', action='store_true', help='处理单个文件')

    args = parser.parse_args()

    # 创建分析器
    analyzer = PeptidePropertiesAnalyzer()

    if args.single:
        # 处理单个文件
        output_path = os.path.join(args.output, "peptide_properties.csv")
        os.makedirs(args.output, exist_ok=True)
        analyzer.process_spectra_file(args.input, output_path)
    else:
        # 批量处理
        analyzer.process_multiple_files(args.input, args.output)


if __name__ == "__main__":
    main()
