import pandas as pd
import re
import numpy as np
import matplotlib
# 设置matplotlib使用Agg后端，这样可以在无GUI环境下使用
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os
import glob
import argparse
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# 全局变量，用于存储数据集到MHC类型的映射
DATASET_MHC_TYPE = {}

def load_meta_data(meta_file):
    """加载meta_pass.txt文件，获取数据集MHC类型信息"""
    global DATASET_MHC_TYPE
    
    try:
        df = pd.read_csv(meta_file, sep='\t')
        if 'all_accession' not in df.columns or 'HLA(I/II)' not in df.columns:
            print(f"警告: meta文件缺少必要的列: all_accession或HLA(I/II)")
            return False
        
        # 创建数据集到MHC类型的映射
        for _, row in df.iterrows():
            dataset_id = row['all_accession']
            hla_type = row['HLA(I/II)']
            
            # 统一处理为I, II或混合类型
            if pd.isna(hla_type) or hla_type == '未明确':
                # 默认按MHC-I处理未指定类型
                DATASET_MHC_TYPE[dataset_id] = 'I'
            elif '/' in hla_type:  # 处理I/II这样的混合类型
                DATASET_MHC_TYPE[dataset_id] = 'mixed'
            else:
                DATASET_MHC_TYPE[dataset_id] = hla_type
                
        print(f"从meta文件加载了{len(DATASET_MHC_TYPE)}个数据集的MHC类型信息")
        return True
    except Exception as e:
        print(f"读取meta文件错误: {e}")
        return False

def get_mhc_type_for_dataset(dataset_id):
    """获取指定数据集的MHC类型"""
    global DATASET_MHC_TYPE
    
    # 精确匹配
    if dataset_id in DATASET_MHC_TYPE:
        return DATASET_MHC_TYPE[dataset_id]
    
    # 处理带后缀的情况（如PXD013649_human）
    base_id = dataset_id.split('_')[0]
    if base_id in DATASET_MHC_TYPE:
        return DATASET_MHC_TYPE[base_id]
    
    # 默认返回MHC-I
    print(f"警告: 未在meta文件中找到数据集{dataset_id}的MHC类型信息，默认使用MHC-I")
    return 'I'

def parse_modification(mod_str):
    """解析修饰字符串，返回修饰位点和类型的列表"""
    if not mod_str or pd.isna(mod_str):
        return []
    
    modifications = []
    for mod in mod_str.split(';'):
        mod = mod.strip()
        if not mod:
            continue
            
        parts = mod.split(',')
        if len(parts) >= 2:
            position = int(parts[0])
            mod_type = parts[1]
            modifications.append((position, mod_type))
    
    return modifications

def extract_modification_type(mod_type_str):
    """
    从修饰类型字符串中提取基本修饰类型和氨基酸标识
    例如: 从"Oxidation[M]"提取出"Oxidation"和"M"
    """
    # 匹配基本修饰类型和可选的氨基酸标识
    match = re.match(r'([^\[\]]+)(?:\[([^\[\]]+)\])?', mod_type_str)
    if match:
        basic_type = match.group(1).strip()
        aa_type = match.group(2) if match.group(2) else ""
        full_type = f"{basic_type}[{aa_type}]" if aa_type else basic_type
        return basic_type, aa_type, full_type
    return mod_type_str, "", mod_type_str

def calculate_site_score_for_9mer(peptides, mod_type, mhc_class):
    """
    专门为9肽计算指定修饰类型的site_score，考虑氨基酸特异性背景频率
    
    参数:
    peptides: 9肽列表，每个肽段包含序列和修饰信息
    mod_type: 要分析的修饰类型，如"Oxidation[M]"
    mhc_class: MHC类型，'I'或'II'
    
    返回:
    dict: 包含site_score及相关统计数据
    """
    # 提取修饰类型和氨基酸标识
    basic_mod, aa_type, full_mod = extract_modification_type(mod_type)
    
    # 初始化统计计数器
    total_peptides = len(peptides)
    
    # 如果氨基酸标识为空，则分析所有位置；否则只分析特定氨基酸的位置
    analyze_specific_aa = aa_type != ""
    
    # 按位置跟踪：修饰出现次数、特定氨基酸出现次数
    mod_counts = defaultdict(int)  # 每个位置的修饰计数
    aa_counts = defaultdict(int)   # 每个位置特定氨基酸的出现计数
    
    # 收集统计数据
    for peptide in peptides:
        sequence = peptide['Sequence']
        
        # 记录修饰位置
        mod_positions = set()  # 避免重复计数同一位置
        for pos, mod in peptide['Modification_List']:
            curr_basic, curr_aa, _ = extract_modification_type(mod)
            if 0 <= pos <= 8 and (
                (analyze_specific_aa and curr_basic == basic_mod and curr_aa == aa_type) or
                (not analyze_specific_aa and curr_basic == basic_mod)
            ):
                mod_positions.add(pos)
        
        # 统计修饰次数
        for pos in mod_positions:
            mod_counts[pos] += 1
        
        # 统计特定氨基酸出现次数（如果需要）
        if analyze_specific_aa:
            for pos, aa in enumerate(sequence):
                if pos < 9 and aa == aa_type:  # 确保在9肽范围内
                    aa_counts[pos] += 1
    
    # 根据MHC类别定义锚定位点和中心位置
    # MHC-I: P2和C端(P9)为锚定位点
    anchor_positions = [1, 8]  # 0-based位置: P2和P9
    # 中心位置P3-P7
    center_positions = [2, 3, 4, 5, 6]  # 0-based位置
    
    # 计算各位置的修饰频率和差值
    position_freq = {}       # 修饰频率
    position_bg_freq = {}    # 背景频率
    position_diff = {}       # 差值
    
    # 计算总体背景频率（如果需要特异性氨基酸分析）
    if analyze_specific_aa:
        total_aa_count = sum(aa_counts.values())
        total_mod_count = sum(mod_counts.values())
        overall_bg_freq = total_mod_count / total_aa_count if total_aa_count > 0 else 0
    else:
        overall_bg_freq = sum(mod_counts.values()) / (9 * total_peptides)  # 平均每个位置的修饰频率
    
    # 计算每个位置的频率和差值
    for pos in range(9):
        if analyze_specific_aa:
            # 特定氨基酸分析
            aa_count = aa_counts.get(pos, 0)
            mod_count = mod_counts.get(pos, 0)
            
            # 修饰频率 = 该位置修饰次数 / 该位置氨基酸出现次数
            freq = mod_count / aa_count if aa_count > 0 else 0
            
            # 使用总体背景频率
            bg_freq = overall_bg_freq
            
            # 差值 = 观察频率 - 背景频率
            diff = freq - bg_freq
        else:
            # 非特异性分析
            mod_count = mod_counts.get(pos, 0)
            freq = mod_count / total_peptides
            bg_freq = overall_bg_freq
            diff = freq - bg_freq
        
        position_freq[pos] = freq
        position_bg_freq[pos] = bg_freq
        position_diff[pos] = diff
    
    # 计算锚定位置和中心位置的平均差值
    anchor_diff_sum = sum(position_diff.get(pos, 0) for pos in anchor_positions)
    center_diff_sum = sum(position_diff.get(pos, 0) for pos in center_positions)
    
    anchor_diff_avg = anchor_diff_sum / len(anchor_positions) if anchor_positions else 0
    center_diff_avg = center_diff_sum / len(center_positions) if center_positions else 0
    
    # 计算site_score (锚定位置平均差值 - 中心位置平均差值)
    site_score = anchor_diff_avg - center_diff_avg
    
    # 计算锚定位置和中心位置的原始频率（用于报告）
    anchor_freq = sum(position_freq.get(pos, 0) for pos in anchor_positions) / len(anchor_positions)
    center_freq = sum(position_freq.get(pos, 0) for pos in center_positions) / len(center_positions)
    
    # 统计总修饰计数
    total_count = sum(mod_counts.values())
    
    return {
        'site_score': site_score,
        'anchor_freq': anchor_freq,
        'center_freq': center_freq,
        'anchor_diff_avg': anchor_diff_avg,
        'center_diff_avg': center_diff_avg,
        'position_freq': position_freq,
        'position_bg_freq': position_bg_freq,
        'position_diff': position_diff,
        'count': total_count,
        'aa_specific': analyze_specific_aa,
        'aa_type': aa_type,
        'overall_bg_freq': overall_bg_freq
    }

def analyze_modifications(file_path, dataset_name=None):
    """
    分析修饰文件并为每种修饰类型计算site_score (只针对9肽)
    """
    # 读取数据文件
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"读取文件错误: {e}")
        return None
    
    # 确保必要的列存在
    required_cols = ['Sequence', 'Modification']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"文件缺少必要的列: {', '.join(missing_cols)}")
        return None
    
    # 获取该数据集的MHC类型
    mhc_type = None
    if dataset_name:
        mhc_type = get_mhc_type_for_dataset(dataset_name)
        print(f"数据集 {dataset_name} 的MHC类型: {mhc_type}")
    else:
        mhc_type = 'I'  # 默认使用MHC-I
    
    # 添加必要的列
    df['Peptide_Length'] = df['Sequence'].apply(len)
    df['MHC_Class_Predicted'] = mhc_type
    df['Modification_List'] = df['Modification'].apply(parse_modification)
    
    # 初始化Site_Scores列为空字典
    df['Site_Scores'] = [{} for _ in range(len(df))]
    
    # 筛选出9肽
    df_9mer = df[df['Peptide_Length'] == 9].copy()
    
    if len(df_9mer) == 0:
        print(f"警告: 文件 {file_path} 中没有9肽")
        return None
    
    # 收集所有修饰类型
    all_mod_types = set()
    for _, row in df_9mer.iterrows():
        for _, mod in row['Modification_List']:
            _, _, full_mod = extract_modification_type(mod)
            all_mod_types.add(full_mod)
    
    # 为每种修饰类型计算site_score
    mod_site_scores = {}
    peptides_data = df_9mer.to_dict('records')
    
    for mod_type in all_mod_types:
        scores = calculate_site_score_for_9mer(peptides_data, mod_type, mhc_type)
        if scores and scores['count'] > 0:  # 只保留有数据的修饰类型
            mod_site_scores[mod_type] = scores
    
    # 将site_score添加到9肽DataFrame
    for i, row in df_9mer.iterrows():
        for pos, mod in row['Modification_List']:
            _, _, full_mod = extract_modification_type(mod)
            if full_mod in mod_site_scores:
                df_9mer.at[i, 'Site_Scores'][full_mod] = mod_site_scores[full_mod]['site_score']
    
    # 更新原始DataFrame中的9肽部分
    df.update(df_9mer[['Site_Scores']])
    
    # 添加数据集名称列（如果提供）
    if dataset_name:
        df['Dataset'] = dataset_name
    
    # 返回结果
    return {
        'data': df,
        'mod_site_scores': mod_site_scores,
        'dataset_name': dataset_name
    }

def write_site_scores_report(results, output_file):
    """
    将修饰类型的site_score结果写入报告文件
    """
    with open(output_file, 'w') as f:
        f.write("修饰类型Site Score分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        if results['dataset_name']:
            f.write(f"数据集: {results['dataset_name']}\n\n")
        
        f.write("9肽修饰类型Site Score:\n")
        f.write("-" * 40 + "\n")
        
        for mod_type, scores in results['mod_site_scores'].items():
            f.write(f"修饰类型: {mod_type}\n")
            f.write(f"  Site Score: {scores['site_score']:.4f}\n")
            f.write(f"  修饰数量: {scores['count']}\n")
            
            # 氨基酸特异性分析
            if scores['aa_specific']:
                f.write(f"  氨基酸特异性分析: {scores['aa_type']}\n")
                f.write(f"  整体背景频率: {scores['overall_bg_freq']:.4f}\n")
            
            f.write(f"  锚定位置频率: {scores['anchor_freq']:.4f}\n")
            f.write(f"  中心位置频率: {scores['center_freq']:.4f}\n")
            f.write(f"  锚定位置平均差值: {scores['anchor_diff_avg']:.4f}\n")
            f.write(f"  中心位置平均差值: {scores['center_diff_avg']:.4f}\n")
            
            f.write(f"  位置分布详情:\n")
            for pos in range(9):
                obs_freq = scores['position_freq'].get(pos, 0)
                bg_freq = scores['position_bg_freq'].get(pos, 0)
                diff = scores['position_diff'].get(pos, 0)
                
                f.write(f"    位置 {pos+1} (P{pos+1}): 观察频率={obs_freq:.4f}, 背景频率={bg_freq:.4f}, 差值={diff:.4f}\n")
            
            f.write("\n")

def create_site_score_matrix(results_list, dataset_id):
    """
    创建修饰类型的Site_Score矩阵，用于热图可视化
    
    参数:
    results_list: 包含各个样本分析结果的列表
    dataset_id: 数据集ID
    
    返回:
    pandas.DataFrame: 修饰类型 x 样本的Site_Score矩阵
    """
    # 收集所有修饰类型和样本名称
    all_mod_types = set()
    sample_names = []
    sample_file_map = {}  # 用于存储原始文件名到简化名称的映射
    
    for i, result in enumerate(results_list):
        if result is None or 'mod_site_scores' not in result:
            continue
            
        # 获取原始样本名称
        orig_name = os.path.splitext(os.path.basename(result.get('sample_file', f'unknown_{i}')))[0]
        # 创建简化名称 (file1, file2, ...)
        simple_name = f"file{i+1}"
        
        # 保存映射关系
        sample_file_map[simple_name] = orig_name
        sample_names.append(simple_name)
        all_mod_types.update(result['mod_site_scores'].keys())
    
    # 创建修饰类型 x 样本的矩阵
    site_score_data = {}
    mod_counts = {}  # 用于统计每种修饰类型的总出现次数
    
    for mod_type in all_mod_types:
        site_score_data[mod_type] = []
        mod_counts[mod_type] = 0
        
        for result in results_list:
            if result is None or 'mod_site_scores' not in result:
                site_score_data[mod_type].append(np.nan)  # 对于缺失的数据使用NaN
                continue
                
            # 获取该样本中该修饰类型的Site_Score和计数
            if mod_type in result['mod_site_scores']:
                scores = result['mod_site_scores'][mod_type]
                site_score = scores['site_score']
                mod_counts[mod_type] += scores.get('count', 0)  # 累积计数
                site_score_data[mod_type].append(site_score)
            else:
                site_score_data[mod_type].append(np.nan)  # 对于缺失的数据使用NaN
    
    # 转换为DataFrame
    df = pd.DataFrame(site_score_data, index=sample_names).T
    
    # 为了更好的可视化，将修饰类型提取为氨基酸+修饰类型格式
    mod_labels = []
    for mod_type in df.index:
        basic_mod, aa_type, _ = extract_modification_type(mod_type)
        if aa_type:
            # 对于特殊标记处理
            if aa_type in ['AnyN-term', 'ProteinN-term']:
                mod_label = f"[N-term].{basic_mod}"
            else:
                mod_label = f"{aa_type}.{basic_mod}"
        else:
            mod_label = basic_mod
        mod_labels.append(mod_label)
    
    df.index = mod_labels
    
    # 保存样本名称映射表
    name_map_file = os.path.join(os.path.dirname(dataset_id), f"{os.path.basename(dataset_id)}_sample_name_map.csv")
    pd.DataFrame(list(sample_file_map.items()), columns=['简化名称', '原始文件名']).to_csv(name_map_file, index=False)
    print(f"保存样本名称映射表到: {name_map_file}")
    
    # 保存矩阵到CSV文件（包含所有修饰类型）
    output_file = f"{dataset_id}_site_score_matrix.csv"
    df.to_csv(output_file)
    print(f"保存完整Site_Score矩阵到: {output_file}")
    
    # 计算每种修饰类型的site_score绝对值平均值
    abs_mean_scores = {}
    for mod_type in df.index:
        # 计算每行的site_score绝对值的平均值（忽略NaN值）
        abs_mean = df.loc[mod_type].abs().mean(skipna=True)
        # 检查abs_mean是否为单个值而不是Series
        if hasattr(abs_mean, 'size') and abs_mean.size > 1:
            # 如果是Series，取其平均值
            abs_mean = abs_mean.mean()
        # 检查是否为nan
        abs_mean_scores[mod_type] = 0 if pd.isna(abs_mean) else abs_mean
    
    # 按site_score绝对值的平均值排序并仅保留前20个（用于热图展示）
    sorted_mods = sorted([(mod, abs_mean_scores.get(mod, 0)) for mod in df.index], 
                         key=lambda x: x[1], reverse=True)
    
    # 取前20个修饰类型
    top_mods = [mod for mod, score in sorted_mods[:20]]
    df_top = df.loc[top_mods] if len(top_mods) > 0 else df
    
    # 同时将排序依据保存到文件
    mod_stats = pd.DataFrame([
        {'Modification': mod, 'AbsMeanScore': score, 'Count': mod_counts.get(mod, 0)}
        for mod, score in sorted_mods
    ])
    stats_file = f"{dataset_id}_modification_stats.csv"
    mod_stats.to_csv(stats_file, index=False)
    print(f"保存修饰类型统计信息到: {stats_file}")
    
    # 保存热图使用的TOP修饰类型数据
    top_output_file = f"{dataset_id}_site_score_matrix_top20.csv"
    df_top.to_csv(top_output_file)
    print(f"保存前20个修饰类型的Site_Score矩阵到: {top_output_file}")
    
    return df_top, sample_file_map  # 返回排序后的前20个修饰类型数据和样本名称映射

def plot_site_score_heatmap(site_score_df, dataset_id, output_dir, sample_file_map=None):
    """
    绘制修饰类型Site_Score的热图，仅展示前10-20个修饰类型
    
    参数:
    site_score_df: 修饰类型 x 样本的Site_Score矩阵
    dataset_id: 数据集ID
    output_dir: 输出目录
    sample_file_map: 简化样本名称到原始文件名的映射
    """
    # 如果数据为空，则返回
    if site_score_df.empty:
        print(f"警告: 数据集 {dataset_id} 没有足够的数据绘制热图")
        return
    
    # 数据预处理：处理NaN值
    # 保存一份原始数据用于后续绘图
    original_df = site_score_df.copy()
    
    # 1. 去除全部为NaN的行和列
    site_score_df = site_score_df.dropna(axis=0, how='all')  # 删除全为NaN的行
    site_score_df = site_score_df.dropna(axis=1, how='all')  # 删除全为NaN的列
    
    if site_score_df.empty:
        print(f"警告: 数据集 {dataset_id} 在去除NaN后没有数据，无法绘制热图")
        return
    
    # 2. 填充剩余的NaN值为0，避免聚类算法错误
    site_score_df = site_score_df.fillna(0)
    
    # 检查是否还有非有限值（如inf）
    if not np.isfinite(site_score_df.values).all():
        print(f"警告: 数据集 {dataset_id} 包含无限值，将替换为0")
        site_score_df = site_score_df.replace([np.inf, -np.inf], 0)
    
    # 检查是否有足够的数据进行聚类
    if len(site_score_df) <= 1 or len(site_score_df.columns) <= 1:
        print(f"警告: 数据集 {dataset_id} 的行或列数量不足，使用简单热图代替聚类热图")
        # 如果数据太少，使用简单热图而非聚类热图
        plt.figure(figsize=(max(8, len(site_score_df.columns) * 0.8), max(6, len(site_score_df) * 0.4)))
        
        # 使用从-1到1的色标范围，以0为中点
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        # 绘制简单热图
        ax = sns.heatmap(
            site_score_df,
            cmap=cmap,
            center=0,
            vmin=-1, 
            vmax=1,
            annot=True,  # 显示数值
            fmt=".2f",
            linewidths=.5,
            cbar_kws={'label': 'Site Score'},
        )
        
        # 设置标题
        plt.title(f'Site Score Heatmap - {dataset_id}', fontsize=14)
        
        # 调整标签
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        # 保存图片
        output_file = os.path.join(output_dir, f"{dataset_id}_site_score_heatmap.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"保存Site_Score热图到: {output_file}")
        plt.close()
        return
    
    try:
        # 设置绘图参数 - 减小图形大小来适应更紧凑的布局
        width = max(8, len(site_score_df.columns) * 0.6)  # 减少每列宽度
        height = max(6, len(site_score_df) * 0.35)  # 减少每行高度
        plt.figure(figsize=(width, height))
        
        # 根据行数调整字体大小
        fontsize = max(7, min(10, 150 / len(site_score_df)))
        
        # 绘制聚类热图
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        g = sns.clustermap(
            site_score_df,
            cmap=cmap,
            center=0,
            vmin=-1, 
            vmax=1,
            yticklabels=True,
            xticklabels=True,
            figsize=(width, height),
            dendrogram_ratio=0.15,  # 减小树状图比例
            cbar_pos=(0.02, 0.8, 0.05, 0.18),  # [left, bottom, width, height]
            cbar_kws={'label': 'Site Score'},
        )
        
        # 设置标题
        plt.suptitle(f'Site Score Heatmap - {dataset_id}', fontsize=14, y=0.95)
        
        # 调整横轴标签 - 使用简化的样本名称
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=fontsize)
        
    except Exception as e:
        print(f"聚类热图生成失败: {e}，尝试使用简单热图")
        
        # 如果聚类热图失败，退回到简单热图
        plt.close()  # 关闭之前的图形
        plt.figure(figsize=(max(8, len(site_score_df.columns) * 0.7), max(6, len(site_score_df) * 0.35)))
        
        # 使用简单热图
        ax = sns.heatmap(
            site_score_df,
            cmap=cmap,
            center=0,
            vmin=-1, 
            vmax=1,
            linewidths=.5,
            cbar_kws={'label': 'Site Score'},
        )
        
        plt.title(f'Site Score Heatmap - {dataset_id}', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
    
    # 保存图片
    output_file = os.path.join(output_dir, f"{dataset_id}_site_score_heatmap.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"保存Site_Score热图到: {output_file}")
    plt.close()
    
    # 额外保存一个CSV文件，包含使用的数据
    cleaned_data_file = os.path.join(output_dir, f"{dataset_id}_site_score_heatmap_data.csv")
    site_score_df.to_csv(cleaned_data_file)
    print(f"保存用于绘图的处理后数据到: {cleaned_data_file}")

def process_metadata_file(meta_file, data_root_dir, output_dir):
    """批量处理元数据文件中列出的所有数据集，按分析场景分组组织结果"""
    # 加载meta文件中的MHC类型信息
    if not load_meta_data(meta_file):
        print("警告: 无法加载meta文件中的MHC类型信息，将使用基于肽段长度的自动判断")
    
    # 读取元数据文件
    try:
        meta_df = pd.read_csv(meta_file, sep='\t')
    except Exception as e:
        print(f"读取元数据文件出错: {e}")
        return
    
    # 确保必要的列存在
    required_cols = ['all_accession', '分析场景']
    missing_cols = [col for col in required_cols if col not in meta_df.columns]
    if missing_cols:
        print(f"元数据文件 {meta_file} 中缺少必要的列: {', '.join(missing_cols)}")
        if '分析场景' in missing_cols:
            print("无法按分析场景组织结果，将使用默认输出结构")
            meta_df['分析场景'] = 'Unknown'  # 添加默认场景
    
    # 创建主输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 按分析场景分组处理数据集
    scene_groups = meta_df.groupby('分析场景')
    
    # 记录所有数据集的汇总信息
    all_dataset_summaries = {}
    
    # 处理每个分析场景
    for scene_name, scene_group in scene_groups:
        # 跳过空场景名
        if pd.isna(scene_name) or scene_name.strip() == '':
            scene_name = 'Unknown'
            
        # 为分析场景创建目录
        scene_dir = os.path.join(output_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        print(f"\n处理分析场景: {scene_name}")
        
        # 获取该场景下的数据集
        datasets = scene_group['all_accession'].tolist()
        print(f"场景 {scene_name} 包含 {len(datasets)} 个数据集")
        
        # 处理每个数据集
        for dataset_id in datasets:
            if pd.isna(dataset_id) or dataset_id.strip() == '':
                continue
                
            # 首先尝试精确匹配目录
            exact_dataset_dir = os.path.join(data_root_dir, dataset_id)
            if os.path.exists(exact_dataset_dir) and os.path.isdir(exact_dataset_dir):
                print(f"\n开始处理数据集: {dataset_id}")
                dataset_summary = process_dataset(dataset_id, exact_dataset_dir, scene_dir)
                
                if dataset_summary:
                    all_dataset_summaries[dataset_id] = dataset_summary
            continue
            
            # 如果精确匹配不存在，尝试查找以dataset_id为前缀的目录（如 PXD005084_human）
            potential_dirs = glob.glob(os.path.join(data_root_dir, f"{dataset_id}*"))
            
            if potential_dirs:
                # 使用第一个匹配的目录
                dataset_dir = potential_dirs[0]
                actual_dir_name = os.path.basename(dataset_dir)
                print(f"\n找到数据集 {dataset_id} 的目录: {actual_dir_name}")
                print(f"开始处理数据集: {dataset_id}")
                
                dataset_summary = process_dataset(dataset_id, dataset_dir, scene_dir)
                
                if dataset_summary:
                    all_dataset_summaries[dataset_id] = dataset_summary
            else:
                print(f"警告: 未找到数据集 {dataset_id} 的目录，跳过")
    
    # 创建跨数据集的Site Score报告
    cross_dataset_dir = os.path.join(output_dir, "cross_dataset_analysis")
    os.makedirs(cross_dataset_dir, exist_ok=True)
    
    with open(os.path.join(cross_dataset_dir, "all_datasets_site_scores.txt"), 'w') as f:
        f.write("跨数据集修饰类型Site Score汇总\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总共分析的数据集数: {len(all_dataset_summaries)}\n\n")
        
        # 收集所有修饰类型
        all_mod_types = set()
        for summary in all_dataset_summaries.values():
            all_mod_types.update(summary['mod_site_scores'].keys())
        
        # 按修饰类型汇总
        for mod_type in sorted(all_mod_types):
            f.write(f"修饰类型: {mod_type}\n")
            f.write("-" * 40 + "\n")
            
            # 收集该修饰类型在所有数据集中的site score
            mod_datasets = []
            for dataset_id, summary in all_dataset_summaries.items():
                if mod_type in summary['mod_site_scores']:
                    scores = summary['mod_site_scores'][mod_type]
                    mod_datasets.append({
                        'dataset_id': dataset_id,
                        'site_score': scores['site_score'],
                        'count': scores['count']
                    })
            
            # 按数量排序
            mod_datasets.sort(key=lambda x: x['count'], reverse=True)
            
            # 计算加权平均site score
            total_count = sum(dataset['count'] for dataset in mod_datasets)
            weighted_score = sum(dataset['site_score'] * dataset['count'] for dataset in mod_datasets) / total_count if total_count > 0 else 0
            
            f.write(f"总修饰数量: {total_count}\n")
            f.write(f"加权平均Site Score: {weighted_score:.4f}\n\n")
            
            f.write("各数据集Site Score:\n")
            for dataset in mod_datasets:
                f.write(f"  {dataset['dataset_id']}: {dataset['site_score']:.4f} (数量: {dataset['count']})\n")
            
            f.write("\n\n")
    
    # 创建按分析场景的汇总热图
    try:
        create_scene_summary_heatmaps(all_dataset_summaries, meta_df, output_dir)
    except Exception as e:
        print(f"创建分析场景汇总热图时出错: {e}")
    
    print(f"\n跨数据集分析完成，结果保存在 {cross_dataset_dir}")

def process_dataset(dataset_id, dataset_dir, output_dir):
    """处理单个数据集目录下的所有.spectra文件"""
    # 创建数据集输出目录
    try:
        dataset_output_dir = os.path.join(output_dir, dataset_id)
        os.makedirs(dataset_output_dir, exist_ok=True)
        print(f"为数据集 {dataset_id} 创建输出目录: {dataset_output_dir}")
    except Exception as e:
        print(f"警告: 无法为数据集 {dataset_id} 创建输出目录: {e}")
        # 尝试使用替代目录
        dataset_output_dir = os.path.join(output_dir, "dataset_" + re.sub(r'[^\w]', '_', dataset_id))
        try:
            os.makedirs(dataset_output_dir, exist_ok=True)
            print(f"使用替代输出目录: {dataset_output_dir}")
        except Exception as e2:
            print(f"错误: 无法创建替代输出目录: {e2}")
            return None
    
    # 搜索数据集目录中的所有.spectra文件
    try:
        spectra_files = glob.glob(os.path.join(dataset_dir, "**", "*.spectra"), recursive=True)
    except Exception as e:
        print(f"警告: 搜索数据集 {dataset_id} 的.spectra文件时出错: {e}")
        spectra_files = []
    
    if not spectra_files:
        print(f"警告: 数据集 {dataset_id} 目录 {dataset_dir} 中未找到.spectra文件")
        return None
    
    print(f"数据集 {dataset_id} 中找到 {len(spectra_files)} 个.spectra文件")
    
    # 处理每个.spectra文件
    all_results_data = []
    all_sample_results = []  # 用于存储每个样本的分析结果，用于后续创建热图
    all_mod_site_scores = defaultdict(lambda: {'count': 0, 'score_sum': 0.0, 'position_freq': defaultdict(float)})
    
    for spectra_file in spectra_files:
        try:
            file_name = os.path.basename(spectra_file)
            print(f"  处理文件: {file_name}")
            
            # 为文件创建输出目录
            file_prefix = os.path.splitext(file_name)[0]
            safe_prefix = re.sub(r'[^\w]', '_', file_prefix)
            file_output_dir = os.path.join(dataset_output_dir, safe_prefix)
            
            try:
                os.makedirs(file_output_dir, exist_ok=True)
                report_file = os.path.join(file_output_dir, f"{safe_prefix}_site_scores.txt")
            except Exception as e:
                print(f"  警告: 无法为文件 {file_name} 创建输出目录: {e}")
                file_output_dir = dataset_output_dir
                report_file = os.path.join(file_output_dir, f"{safe_prefix}_site_scores.txt")
                print(f"  使用替代输出目录: {file_output_dir}")
    
    # 分析文件
            result = analyze_modifications(spectra_file, dataset_id)
            
            if result is None:
                print(f"  分析文件 {file_name} 失败，跳过")
                continue
            
            # 添加样本文件信息，用于后续创建热图
            result['sample_file'] = file_name
            all_sample_results.append(result)
            
            # 保存单个样本的Site_Score统计文件
            sample_site_scores = {}
            for mod_type, scores in result['mod_site_scores'].items():
                sample_site_scores[mod_type] = scores['site_score']
            
            # 保存为CSV文件
            sample_scores_df = pd.DataFrame(list(sample_site_scores.items()), columns=['Modification_Type', 'Site_Score'])
            sample_scores_file = os.path.join(file_output_dir, f"{safe_prefix}_site_scores.csv")
            sample_scores_df.to_csv(sample_scores_file, index=False)
            print(f"  保存样本Site_Score统计到: {sample_scores_file}")
            
            # 保存site score报告
            write_site_scores_report(result, report_file)
            print(f"  保存Site Score报告到: {report_file}")
            
            # 保存详细结果
            try:
                result_file = os.path.join(file_output_dir, f"{safe_prefix}_analysis.csv")
                result['data'].to_csv(result_file, index=False)
                print(f"  保存分析结果到: {result_file}")
            except Exception as e:
                print(f"  保存分析结果文件时出错: {e}")
            
            # 累积结果数据
            all_results_data.append(result['data'])
            
            # 合并所有修饰类型的site score结果
            for mod_type, scores in result['mod_site_scores'].items():
                all_mod_site_scores[mod_type]['count'] += scores['count']
                all_mod_site_scores[mod_type]['score_sum'] += scores['site_score'] * scores['count']
                
                # 合并位置频率（加权平均）
                for pos, freq in scores['position_freq'].items():
                    weight = scores['count'] / (all_mod_site_scores[mod_type]['count'] or 1)
                    all_mod_site_scores[mod_type]['position_freq'][pos] += freq * weight
            
        except Exception as e:
            print(f"  处理文件 {os.path.basename(spectra_file)} 时出错: {e}")
    
    if not all_results_data:
        print(f"数据集 {dataset_id} 没有成功处理的文件")
        return None
    
    # 创建并保存修饰类型的Site_Score矩阵
    if all_sample_results:
        site_score_df, sample_file_map = create_site_score_matrix(all_sample_results, os.path.join(dataset_output_dir, dataset_id))
        # 绘制并保存热图
        plot_site_score_heatmap(site_score_df, dataset_id, dataset_output_dir, sample_file_map)
    
    # 合并所有文件的结果
    try:
        all_data = pd.concat(all_results_data, ignore_index=True)
        
        # 计算每种修饰类型的平均site score
        dataset_mod_site_scores = {}
        for mod_type, data in all_mod_site_scores.items():
            if data['count'] > 0:
                avg_site_score = data['score_sum'] / data['count']
                dataset_mod_site_scores[mod_type] = {
                    'site_score': avg_site_score,
                    'count': data['count'],
                    'position_freq': data['position_freq']
                }
        
        # 创建数据集级别的汇总结果
        dataset_summary = {
            'data': all_data,
            'mod_site_scores': dataset_mod_site_scores,
            'dataset_name': dataset_id
        }
        
        # 保存数据集级别的汇总结果
        all_data.to_csv(os.path.join(dataset_output_dir, f"{dataset_id}_all_analysis.csv"), index=False)
        
        # 保存数据集级别的site score报告
        dataset_report_file = os.path.join(dataset_output_dir, f"{dataset_id}_site_scores_summary.txt")
        write_site_scores_report(dataset_summary, dataset_report_file)
        print(f"保存数据集site score汇总报告到: {dataset_report_file}")
        
        # 保存数据集级别的Site_Score统计CSV
        dataset_scores_df = pd.DataFrame([
            {'Modification_Type': mod_type, 'Site_Score': data['site_score'], 'Count': data['count']}
            for mod_type, data in dataset_mod_site_scores.items()
        ])
        dataset_scores_file = os.path.join(dataset_output_dir, f"{dataset_id}_site_scores_summary.csv")
        dataset_scores_df.to_csv(dataset_scores_file, index=False)
        print(f"保存数据集Site_Score汇总统计到: {dataset_scores_file}")
        
        print(f"数据集 {dataset_id} 分析完成")
        return dataset_summary
    except Exception as e:
        print(f"处理数据集 {dataset_id} 的结果时出错: {e}")
        return None

def create_scene_summary_heatmaps(all_dataset_summaries, meta_df, output_dir):
    """
    创建按分析场景的汇总热图
    
    参数:
    all_dataset_summaries: 包含所有数据集分析结果的字典
    meta_df: 元数据DataFrame
    output_dir: 输出目录
    """
    # 创建汇总目录
    summary_dir = os.path.join(output_dir, "summary_by_scene")
    os.makedirs(summary_dir, exist_ok=True)
    
    # 为元数据创建索引，便于查找
    meta_dict = {}
    for _, row in meta_df.iterrows():
        if 'all_accession' in row and '分析场景' in row:
            meta_dict[row['all_accession']] = row['分析场景']
    
    # 按场景分组数据集
    scene_datasets = defaultdict(list)
    for dataset_id, summary in all_dataset_summaries.items():
        # 获取数据集的场景，如果不在meta_dict中则归类为"Unknown"
        scene = meta_dict.get(dataset_id, 'Unknown')
        if pd.isna(scene) or scene.strip() == '':
            scene = 'Unknown'
        scene_datasets[scene].append((dataset_id, summary))
    
    # 为每个场景创建汇总热图
    for scene, datasets in scene_datasets.items():
        # 收集该场景下所有数据集的修饰类型和site score
        all_mod_types = set()
        scene_site_scores = {}
        
        for dataset_id, summary in datasets:
            for mod_type, data in summary['mod_site_scores'].items():
                all_mod_types.add(mod_type)
                if mod_type not in scene_site_scores:
                    scene_site_scores[mod_type] = []
                scene_site_scores[mod_type].append({
                    'dataset': dataset_id,
                    'site_score': data['site_score'],
                    'count': data['count']
                })
        
        # 创建场景级别的site score矩阵
        df_data = {}
        for mod_type in all_mod_types:
            df_data[mod_type] = []
            dataset_scores = {d[0]: None for d in datasets}  # 初始化为所有数据集为None
            
            # 填充已有的数据
            for score_data in scene_site_scores.get(mod_type, []):
                dataset_scores[score_data['dataset']] = score_data['site_score']
            
            # 将字典转换为列表，保持数据集顺序
            for dataset_id, _ in datasets:
                df_data[mod_type].append(dataset_scores[dataset_id])
        
        # 创建DataFrame
        df_columns = [d[0] for d in datasets]
        df = pd.DataFrame(df_data, index=df_columns).T
        
        if df.empty:
            print(f"场景 {scene} 没有足够的数据创建热图")
            continue
        
        # 为了更好的可视化，调整标签格式
        mod_labels = []
        for mod_type in df.index:
            basic_mod, aa_type, _ = extract_modification_type(mod_type)
            if aa_type:
                if aa_type in ['AnyN-term', 'ProteinN-term']:
                    mod_label = f"[N-term].{basic_mod}"
                else:
                    mod_label = f"{aa_type}.{basic_mod}"
            else:
                mod_label = basic_mod
            mod_labels.append(mod_label)
        df.index = mod_labels
        
        # 计算每种修饰类型的site_score绝对值平均值
        abs_mean_scores = {}
        for mod_type in df.index:
            abs_mean = df.loc[mod_type].abs().mean(skipna=True)
            if hasattr(abs_mean, 'size') and abs_mean.size > 1:
                abs_mean = abs_mean.mean()
            abs_mean_scores[mod_type] = 0 if pd.isna(abs_mean) else abs_mean
        
        # 按site_score绝对值平均值排序
        sorted_mods = sorted([(mod, abs_mean_scores.get(mod, 0)) for mod in df.index], 
                           key=lambda x: x[1], reverse=True)
        
        # 保存完整矩阵
        full_output_file = os.path.join(summary_dir, f"{scene}_site_score_matrix.csv")
        df.to_csv(full_output_file)
        print(f"保存场景 {scene} 的完整Site_Score矩阵到: {full_output_file}")
        
        # 选择前20个修饰类型
        top_mods = [mod for mod, _ in sorted_mods[:20]]
        df_top = df.loc[top_mods] if len(top_mods) > 0 else df
        
        # 保存TOP20矩阵
        top_output_file = os.path.join(summary_dir, f"{scene}_site_score_matrix_top20.csv")
        df_top.to_csv(top_output_file)
        print(f"保存场景 {scene} 的TOP20 Site_Score矩阵到: {top_output_file}")
        
        # 绘制热图
        plot_site_score_heatmap(df_top, f"{scene}_summary", summary_dir)

def sanitize_filename(filename):
    """
    清理文件名，移除或替换不允许的字符
    """
    # 替换Windows文件系统不允许的字符
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # 替换其他可能导致问题的字符
    filename = filename.replace('[', '(').replace(']', ')')
    filename = filename.replace('->', '_to_')
    filename = filename.replace(' ', '_')
    
    return filename

def plot_trend_comparison(output_dir):
    """
    生成Site Score趋势比较热图
    使用Kruskal-Wallis检验识别不同场景间差异显著的修饰类型
    按照统计检验结果排序并展示前50个修饰类型
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import glob
    import numpy as np
    from matplotlib.patches import Rectangle, Patch
    from scipy.stats import kruskal, f_oneway
    from statsmodels.sandbox.stats.multicomp import multipletests
    
    # 创建趋势图输出目录
    trend_dir = os.path.join(output_dir, "trend_comparison")
    os.makedirs(trend_dir, exist_ok=True)
    
    # 收集所有数据集的数据
    scene_data = []
    for scene_dir in glob.glob(os.path.join(output_dir, "*")):
        if not os.path.isdir(scene_dir) or scene_dir.endswith('trend_comparison') or scene_dir.endswith('cross_dataset_analysis'):
            continue
            
        scene_name = os.path.basename(scene_dir)
        print(f"处理场景: {scene_name}")
        
        # 在场景目录中查找数据集目录
        for dataset_dir in glob.glob(os.path.join(scene_dir, "*")):
            if not os.path.isdir(dataset_dir):
                continue
                
            dataset_id = os.path.basename(dataset_dir)
            if dataset_id.endswith('_human'):
                dataset_id = dataset_id[:-6]  # 去掉_human后缀
                
            # 查找site score矩阵文件
            matrix_file = os.path.join(dataset_dir, f"{dataset_id}_site_score_matrix.csv")
            if os.path.exists(matrix_file):
                try:
                    df = pd.read_csv(matrix_file, index_col=0)
                    # 将数据转换为长格式
                    df_long = df.reset_index()
                    df_long = pd.melt(df_long, id_vars=['index'], var_name='Sample', value_name='Site_Score')
                    df_long['Modification'] = df_long['index']
                    df_long['Dataset'] = dataset_id
                    df_long['Scene'] = scene_name
                    
                    # 处理无限值
                    df_long['Site_Score'] = df_long['Site_Score'].replace([np.inf, -np.inf], np.nan)
                    
                    scene_data.append(df_long)
                    print(f"  找到数据集 {dataset_id} 的矩阵文件")
                except Exception as e:
                    print(f"  读取数据集 {dataset_id} 的矩阵文件时出错: {e}")
    
    if not scene_data:
        print("没有找到可用的数据集矩阵文件")
        return
        
    # 合并所有数据
    all_data = pd.concat(scene_data, ignore_index=True)
    
    # ============= 使用统计检验评估修饰类型在不同场景间的差异 =============
    # 计算每个修饰类型在每个场景中的样本数量
    mod_scene_counts = all_data.groupby(['Modification', 'Scene']).size().reset_index(name='count')
    
    # 找出在至少2个场景中出现且每个场景至少有3个样本的修饰类型
    mod_counts = mod_scene_counts.groupby('Modification').agg(
        num_scenes=('Scene', 'nunique'),
        min_count=('count', 'min')
    ).reset_index()
    
    valid_mods = mod_counts[(mod_counts['num_scenes'] >= 2) & (mod_counts['min_count'] >= 3)]['Modification'].tolist()
    
    print(f"找到满足统计检验条件的修饰类型: {len(valid_mods)}")
    
    # 执行Kruskal-Wallis检验
    kw_results = []
    
    for mod in valid_mods:
        mod_data = all_data[all_data['Modification'] == mod]
        
        # 收集每个场景的Site Score
        scene_scores = {scene: mod_data[mod_data['Scene'] == scene]['Site_Score'].dropna().tolist() 
                       for scene in mod_data['Scene'].unique()}
        
        # 确保至少有2个场景，且每个场景至少有3个样本
        valid_scenes = [scene for scene, scores in scene_scores.items() if len(scores) >= 3]
        
        if len(valid_scenes) < 2:
            continue
            
        # 执行Kruskal-Wallis检验
        try:
            scene_score_lists = [scene_scores[scene] for scene in valid_scenes]
            stat, p_value = kruskal(*scene_score_lists)
            
            # 计算场景间的平均值差异（最大值-最小值）
            scene_means = {scene: np.mean(scores) for scene, scores in scene_scores.items() if len(scores) >= 3}
            max_diff = max(scene_means.values()) - min(scene_means.values())
            
            # 计算总样本数
            total_samples = sum(len(scores) for scores in scene_score_lists)
            
            kw_results.append({
                'Modification': mod,
                'KW_Statistic': stat,
                'P_Value': p_value,
                'Max_Diff': max_diff,
                'Num_Scenes': len(valid_scenes),
                'Total_Samples': total_samples
            })
            
        except Exception as e:
            print(f"执行修饰类型 {mod} 的Kruskal-Wallis检验时出错: {e}")
            continue
    
    # 转换为DataFrame
    kw_df = pd.DataFrame(kw_results)
    
    if len(kw_df) == 0:
        print("没有找到足够的数据执行统计检验，无法生成热图")
        return
        
    # 执行多重检验校正
    if len(kw_df) > 1:
        _, adj_p_values, _, _ = multipletests(kw_df['P_Value'], method='fdr_bh')
        kw_df['Adjusted_P_Value'] = adj_p_values
    else:
        kw_df['Adjusted_P_Value'] = kw_df['P_Value']
    
    # 按KW统计量降序排序
    kw_df = kw_df.sort_values('KW_Statistic', ascending=False)
    
    # 保存统计检验结果
    stats_output_file = os.path.join(trend_dir, "modification_kruskal_wallis_tests.csv")
    kw_df.to_csv(stats_output_file, index=False)
    print(f"保存Kruskal-Wallis检验结果到: {stats_output_file}")
    
    # 选择前50个差异显著的修饰类型
    significant_mods = kw_df[kw_df['Adjusted_P_Value'] < 0.05].copy()
    
    # 如果显著的修饰类型不足50个，则使用所有有结果的修饰类型
    if len(significant_mods) < 50:
        print(f"警告: 只有 {len(significant_mods)} 个修饰类型显示出统计学显著差异")
        if len(significant_mods) == 0:
            # 如果没有显著差异，使用所有检验结果
            significant_mods = kw_df.copy()
    
    # 选择前50个（或所有可用的）
    top_50_mods = significant_mods.head(50)['Modification'].tolist()
    
    print(f"选择了 {len(top_50_mods)} 个修饰类型用于热图显示")
    
    try:
        # 准备热图数据
        pivot_data = all_data[all_data['Modification'].isin(top_50_mods)].groupby(
            ['Scene', 'Dataset', 'Modification']
        )['Site_Score'].mean().reset_index()
        
        # 获取所有场景和数据集
        scenes = sorted(pivot_data['Scene'].unique())
        scene_datasets = {scene: pivot_data[pivot_data['Scene'] == scene]['Dataset'].unique() 
                         for scene in scenes}
        
        # 创建新的列顺序，按场景分组
        new_columns = []
        for scene in scenes:
            new_columns.extend([(scene, dataset) for dataset in sorted(scene_datasets[scene])])
        
        # 创建透视表并重新排序列
        pivot_data = pivot_data.pivot_table(
            values='Site_Score',
            index='Modification',
            columns=['Scene', 'Dataset'],
            aggfunc='mean'
        )
        
        # 处理数据中的NaN和无限值
        pivot_data = pivot_data.fillna(0)
        pivot_data = pivot_data.replace([np.inf, -np.inf], 0)
        
        # 按Kruskal-Wallis统计量排序修饰类型
        # 创建映射从修饰类型到排序顺序
        mod_order_map = {mod: i for i, mod in enumerate(top_50_mods)}
        
        # 按统计量排序顺序重新排列行
        mod_order = sorted(
            [mod for mod in pivot_data.index if mod in mod_order_map],
            key=lambda x: mod_order_map.get(x, 9999)
        )
        pivot_data = pivot_data.reindex(mod_order)
        
        # =================== 可视化部分 ===================
        # 设置场景颜色
        scene_colors = sns.color_palette("Set2", n_colors=len(scenes))
        
        # 创建图形，进一步增加宽度以确保有足够空间给右侧图例
        fig = plt.figure(figsize=(24, 15))
        
        # 创建网格
        gs = fig.add_gridspec(nrows=1, ncols=1)
        
        # 创建主热图
        ax_main = fig.add_subplot(gs[0])
        
        # 创建热图
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        im = sns.heatmap(
            pivot_data,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={
                'label': 'Site Score', 
                'orientation': 'vertical',
                'pad': 0.02,
                'fraction': 0.02,
                'aspect': 20
            },
            ax=ax_main
        )
        
        # 添加场景分隔线和顶部颜色条
        current_x = 0
        
        # 创建图例元素
        legend_elements = []
        
        for scene_idx, scene in enumerate(scenes):
            n_datasets = len(scene_datasets[scene])
            
            # 在热图顶部添加颜色条
            ax_main.add_patch(Rectangle(
                (current_x, -0.5),  # y坐标改为-0.5，使其紧贴热图顶部
                n_datasets,
                0.5,  # 高度调小
                facecolor=scene_colors[scene_idx],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8,
                clip_on=False  # 确保可以超出轴范围
            ))
            
            # 添加场景分隔线
            if current_x > 0:
                ax_main.axvline(x=current_x, color='white', linewidth=2)
            
            # 为图例添加元素
            legend_elements.append(Patch(
                facecolor=scene_colors[scene_idx],
                edgecolor='black',
                alpha=0.8,
                label=scene
            ))
            
            current_x += n_datasets
        
        # 添加右侧图例，将位置调整得更靠右以避免与颜色条重叠
        ax_main.legend(
            handles=legend_elements,
            title="Scene Categories",  # 英文标题，避免乱码
            loc='center left',
            bbox_to_anchor=(1.25, 0.5),  # 将图例向右移动
            fontsize=10,
            title_fontsize=12,
            frameon=True,
            framealpha=0.95,
            edgecolor='black'
        )
        
        # 设置主图的标签
        ax_main.set_xticklabels(
            ax_main.get_xticklabels(), 
            rotation=45, 
            ha='right',
            fontsize=8
        )
        ax_main.set_yticklabels(
            ax_main.get_yticklabels(),
            fontsize=8
        )
        
        # 设置标题，包含使用的统计方法信息
        plt.suptitle('Top 50 Modifications with Significant Scene Differences (Kruskal-Wallis Test)',
                    fontsize=16,
                    y=0.98)
        
        # 调整布局以适应图例
        plt.tight_layout()
        
        # 保存图片
        output_file = os.path.join(trend_dir, "top_50_scene_diff_heatmap.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存数据
        heatmap_data_file = os.path.join(trend_dir, "top_50_scene_diff_data.csv")
        pivot_data.to_csv(heatmap_data_file)
        
        print("生成Top 50修饰类型场景差异热图完成")
        
    except Exception as e:
        print(f"生成热图时出错: {e}")
        import traceback
        traceback.print_exc()
        # 保存导致错误的数据以供检查
        error_data_file = os.path.join(trend_dir, "error_data.csv")
        try:
            if 'pivot_data' in locals():
                pivot_data.to_csv(error_data_file)
                print(f"已保存导致错误的数据到: {error_data_file}")
        except:
            pass
    
    print(f"趋势比较结果已保存到: {trend_dir}")

def plot_cancer_heatmap(output_dir, meta_file):
    """
    为癌症数据集生成Site Score热图，标注不同的癌症类型

    参数:
    output_dir: 输出目录
    meta_file: 元数据文件路径，包含数据集ID、场景和疾病类型信息
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import glob
    import numpy as np
    from matplotlib.patches import Rectangle, Patch
    from collections import defaultdict

    # 创建癌症热图输出目录
    cancer_dir = os.path.join(output_dir, "cancer_heatmap")
    os.makedirs(cancer_dir, exist_ok=True)
    
    # 加载元数据
    try:
        meta_df = pd.read_csv(meta_file, sep='\t')
        # 确保元数据包含所需列
        required_columns = ['all_accession', '分析场景', '疾病类型']
        if not all(col in meta_df.columns for col in required_columns):
            print(f"警告：元数据文件缺少必要的列：{', '.join(required_columns)}")
            return
    except Exception as e:
        print(f"读取元数据文件错误: {e}")
        return
    
    # 创建数据集ID到疾病类型的映射
    dataset_to_disease = {}
    for _, row in meta_df.iterrows():
        if row['分析场景'] == 'Cancer' and not pd.isna(row['疾病类型']):
            dataset_to_disease[row['all_accession']] = row['疾病类型']
    
    # 收集癌症数据集的数据
    cancer_data = []
    
    # 查找Cancer场景目录
    cancer_scene_dir = os.path.join(output_dir, "Cancer")
    if not os.path.isdir(cancer_scene_dir):
        print("未找到Cancer场景目录")
        return
    
    # 在Cancer场景目录中查找数据集目录
    for dataset_dir in glob.glob(os.path.join(cancer_scene_dir, "*")):
        if not os.path.isdir(dataset_dir):
            continue
        
        dataset_id = os.path.basename(dataset_dir)
        if dataset_id.endswith('_human'):
            dataset_id = dataset_id[:-6]  # 去掉_human后缀
        
        # 获取疾病类型，如果在元数据中没有则标记为"Unspecified"
        disease_type = dataset_to_disease.get(dataset_id, "Unspecified")
        
        # 查找site score矩阵文件
        matrix_file = os.path.join(dataset_dir, f"{dataset_id}_site_score_matrix.csv")
        if os.path.exists(matrix_file):
            try:
                df = pd.read_csv(matrix_file, index_col=0)
                
                # 将数据转换为长格式
                df_long = df.reset_index()
                df_long = pd.melt(df_long, id_vars=['index'], var_name='Sample', value_name='Site_Score')
                df_long['Modification'] = df_long['index']
                df_long['Dataset'] = dataset_id
                df_long['Disease'] = disease_type
                
                # 处理无限值
                df_long['Site_Score'] = df_long['Site_Score'].replace([np.inf, -np.inf], np.nan)
                
                cancer_data.append(df_long)
                print(f"  找到癌症数据集 {dataset_id} 的矩阵文件，疾病类型: {disease_type}")
            except Exception as e:
                print(f"  读取数据集 {dataset_id} 的矩阵文件时出错: {e}")
    
    if not cancer_data:
        print("没有找到可用的癌症数据集矩阵文件")
        return
    
    # 合并所有数据
    all_cancer_data = pd.concat(cancer_data, ignore_index=True)
    
    # 计算每个修饰类型的统计信息
    mod_stats = []
    
    for mod in all_cancer_data['Modification'].unique():
        mod_data = all_cancer_data[all_cancer_data['Modification'] == mod]
        
        # 计算总样本数和非空样本数
        total_count = len(mod_data)
        non_null_count = mod_data['Site_Score'].count()
        
        # 计算平均值和标准差
        mean_score = mod_data['Site_Score'].mean()
        std_score = mod_data['Site_Score'].std()
        
        # 计算不同疾病类型的平均值差异
        disease_means = mod_data.groupby('Disease')['Site_Score'].mean()
        max_diff = disease_means.max() - disease_means.min() if len(disease_means) > 1 else 0
        
        mod_stats.append({
            'Modification': mod,
            'Total_Count': total_count,
            'Non_Null_Count': non_null_count,
            'Mean_Score': mean_score,
            'Std_Score': std_score,
            'Disease_Max_Diff': max_diff
        })
    
    # 转换为DataFrame并按差异排序
    mod_stats_df = pd.DataFrame(mod_stats)
    mod_stats_df = mod_stats_df.sort_values('Disease_Max_Diff', ascending=False)
    
    # 选择前30个差异最大的修饰类型
    top_mods = mod_stats_df.head(30)['Modification'].tolist()
    
    # 保存修饰类型统计信息
    mod_stats_file = os.path.join(cancer_dir, "cancer_modification_stats.csv")
    mod_stats_df.to_csv(mod_stats_file, index=False)
    print(f"保存癌症修饰类型统计信息到: {mod_stats_file}")
    
    try:
        # 准备热图数据
        pivot_data = all_cancer_data[all_cancer_data['Modification'].isin(top_mods)].groupby(
            ['Disease', 'Dataset', 'Modification']
        )['Site_Score'].mean().reset_index()
        
        # 获取所有疾病类型和数据集
        diseases = sorted(pivot_data['Disease'].unique())
        disease_datasets = {disease: pivot_data[pivot_data['Disease'] == disease]['Dataset'].unique() 
                           for disease in diseases}
        
        # 创建新的列顺序，按疾病类型分组
        new_columns = []
        for disease in diseases:
            new_columns.extend([(disease, dataset) for dataset in sorted(disease_datasets[disease])])
        
        # 创建透视表并重新排序列
        pivot_data = pivot_data.pivot_table(
            values='Site_Score',
            index='Modification',
            columns=['Disease', 'Dataset'],
            aggfunc='mean'
        )
        
        # 处理数据中的NaN和无限值
        pivot_data = pivot_data.fillna(0)
        pivot_data = pivot_data.replace([np.inf, -np.inf], 0)
        
        # 按修饰类型差异排序
        mod_order_map = {mod: i for i, mod in enumerate(top_mods)}
        mod_order = sorted(
            [mod for mod in pivot_data.index if mod in mod_order_map],
            key=lambda x: mod_order_map.get(x, 9999)
        )
        pivot_data = pivot_data.reindex(mod_order)
        
        # =================== 可视化部分 ===================
        # 设置疾病类型颜色
        disease_colors = sns.color_palette("Dark2", n_colors=len(diseases))
        
        # 创建图形
        fig = plt.figure(figsize=(24, 15))
        
        # 创建网格
        gs = fig.add_gridspec(nrows=1, ncols=1)
        
        # 创建主热图
        ax_main = fig.add_subplot(gs[0])
        
        # 创建热图
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        im = sns.heatmap(
            pivot_data,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={
                'label': 'Site Score', 
                'orientation': 'vertical',
                'pad': 0.02,
                'fraction': 0.02,
                'aspect': 20
            },
            ax=ax_main
        )
        
        # 添加疾病类型分隔线和顶部颜色条
        current_x = 0
        
        # 创建图例元素
        legend_elements = []
        
        for disease_idx, disease in enumerate(diseases):
            n_datasets = len(disease_datasets[disease])
            
            # 在热图顶部添加颜色条
            ax_main.add_patch(Rectangle(
                (current_x, -0.5),  # y坐标改为-0.5，使其紧贴热图顶部
                n_datasets,
                0.5,  # 高度调小
                facecolor=disease_colors[disease_idx],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8,
                clip_on=False  # 确保可以超出轴范围
            ))
            
            # 添加疾病类型分隔线
            if current_x > 0:
                ax_main.axvline(x=current_x, color='white', linewidth=2)
            
            # 为图例添加元素
            legend_elements.append(Patch(
                facecolor=disease_colors[disease_idx],
                edgecolor='black',
                alpha=0.8,
                label=disease
            ))
            
            current_x += n_datasets
        
        # 添加右侧图例
        ax_main.legend(
            handles=legend_elements,
            title="Cancer Types",
            loc='center left',
            bbox_to_anchor=(1.25, 0.5),
            fontsize=10,
            title_fontsize=12,
            frameon=True,
            framealpha=0.95,
            edgecolor='black'
        )
        
        # 设置主图的标签
        ax_main.set_xticklabels(
            ax_main.get_xticklabels(), 
            rotation=45, 
            ha='right',
            fontsize=8
        )
        ax_main.set_yticklabels(
            ax_main.get_yticklabels(),
            fontsize=8
        )
        
        # 设置标题
        plt.suptitle('Top 30 Modifications with Greatest Differences Across Cancer Types',
                    fontsize=16,
                    y=0.98)
        
        # 调整布局以适应图例
        plt.tight_layout()
        
        # 保存图片
        output_file = os.path.join(cancer_dir, "cancer_type_heatmap.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存数据
        heatmap_data_file = os.path.join(cancer_dir, "cancer_type_heatmap_data.csv")
        pivot_data.to_csv(heatmap_data_file)
        
        print("生成癌症类型修饰差异热图完成")
        
    except Exception as e:
        print(f"生成癌症热图时出错: {e}")
        import traceback
        traceback.print_exc()
        # 保存导致错误的数据以供检查
        error_data_file = os.path.join(cancer_dir, "cancer_error_data.csv")
        try:
            if 'pivot_data' in locals():
                pivot_data.to_csv(error_data_file)
                print(f"已保存导致错误的数据到: {error_data_file}")
        except:
            pass
    
    print(f"癌症热图结果已保存到: {cancer_dir}")

def plot_modification_boxplot(output_dir, target_modification=None):
    """
    为特定修饰类型生成箱线图，展示不同数据集的site score分布
    按场景分组，不同场景使用不同颜色，每个场景内按中位数排序
    
    参数:
    output_dir: 输出目录
    target_modification: 目标修饰类型，如 "Oxidation[M]"，如果为None则提示用户输入
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import glob
    import numpy as np
    
    # 创建输出目录
    boxplot_dir = os.path.join(output_dir, "modification_boxplots")
    os.makedirs(boxplot_dir, exist_ok=True)
    
    # 如果没有指定修饰类型，列出可用的修饰类型供选择
    if target_modification is None:
        all_mods = set()
        # 收集所有修饰类型
        for matrix_file in glob.glob(os.path.join(output_dir, "**", "*_site_score_matrix.csv"), recursive=True):
            try:
                df = pd.read_csv(matrix_file, index_col=0)
                all_mods.update(df.index)
            except Exception:
                continue
        
        if not all_mods:
            print("未找到任何修饰类型数据")
            return
            
        print("可用的修饰类型:")
        for i, mod in enumerate(sorted(all_mods), 1):
            print(f"{i}. {mod}")
            
        # 提示用户选择
        choice = input("请输入要分析的修饰类型编号或名称: ")
        try:
            # 尝试解析为索引
            idx = int(choice) - 1
            if 0 <= idx < len(all_mods):
                target_modification = sorted(all_mods)[idx]
            else:
                # 如果输入的是修饰类型名称
                target_modification = choice
                if target_modification not in all_mods:
                    print(f"未找到修饰类型: {target_modification}")
                    return
        except ValueError:
            # 如果输入的是修饰类型名称
            target_modification = choice
            if target_modification not in all_mods:
                print(f"未找到修饰类型: {target_modification}")
                return
    
    print(f"正在为修饰类型 {target_modification} 生成箱线图...")
    
    # 收集所有数据集的数据
    scene_data = []
    for scene_dir in glob.glob(os.path.join(output_dir, "*")):
        if not os.path.isdir(scene_dir) or scene_dir.endswith(('boxplots', 'trend_comparison', 'cross_dataset_analysis')):
            continue
            
        scene_name = os.path.basename(scene_dir)
        
        # 在场景目录中查找数据集目录
        for dataset_dir in glob.glob(os.path.join(scene_dir, "*")):
            if not os.path.isdir(dataset_dir):
                continue
                
            dataset_id = os.path.basename(dataset_dir)
            
            # 查找site score矩阵文件
            matrix_files = glob.glob(os.path.join(dataset_dir, "*_site_score_matrix.csv"))
            
            for matrix_file in matrix_files:
                try:
                    df = pd.read_csv(matrix_file, index_col=0)
                    
                    # 检查是否包含目标修饰类型
                    if target_modification in df.index:
                        # 提取该修饰类型的数据
                        mod_data = df.loc[target_modification].dropna()
                        
                        # 将数据转换为长格式
                        mod_df = pd.DataFrame({
                            'Site_Score': mod_data.values,
                            'Sample': mod_data.index,
                            'Dataset': dataset_id,
                            'Scene': scene_name
                        })
                        
                        scene_data.append(mod_df)
                        print(f"  从数据集 {dataset_id} 中提取了 {len(mod_data)} 个样本的数据")
                        
                except Exception as e:
                    print(f"  处理数据集 {dataset_id} 的矩阵文件时出错: {e}")
                    continue
    
    if not scene_data:
        print(f"未找到修饰类型 {target_modification} 的数据")
        return
        
    # 合并所有数据
    all_data = pd.concat(scene_data, ignore_index=True)
    
    # 计算每个数据集的统计信息
    dataset_stats = all_data.groupby(['Scene', 'Dataset'])['Site_Score'].agg(['median', 'mean', 'count']).reset_index()
    
    # 按场景分组，并在每个场景内按中位数排序
    ordered_datasets = []
    ordered_datasets_stats = []
    
    for scene, group in dataset_stats.groupby('Scene'):
        # 按中位数从高到低排序
        sorted_group = group.sort_values('median', ascending=False)
        ordered_datasets_stats.append(sorted_group)
        
        # 按排序后的顺序创建数据集列表
        for _, row in sorted_group.iterrows():
            ordered_datasets.append((scene, row['Dataset']))
    
    # 合并排序后的统计信息
    ordered_stats = pd.concat(ordered_datasets_stats, ignore_index=True)
    
    # 创建箱线图
    plt.figure(figsize=(max(14, len(ordered_datasets) * 0.4), 8))
    
    # 设置场景颜色
    unique_scenes = all_data['Scene'].unique()
    scene_colors = dict(zip(unique_scenes, sns.color_palette("Set2", n_colors=len(unique_scenes))))
    
    # 准备用于绘图的数据
    plot_data = []
    for scene, dataset in ordered_datasets:
        # 提取该数据集的数据
        dataset_data = all_data[(all_data['Scene'] == scene) & (all_data['Dataset'] == dataset)]
        
        # 添加到绘图数据中
        plot_data.append({
            'x': f"{dataset}",  # 只显示数据集名称
            'scene': scene,     # 保存场景信息用于分组显示
            'y': dataset_data['Site_Score'].tolist(),  # y轴数据
            'color': scene_colors[scene]  # 颜色
        })
    
    # 绘制箱线图
    ax = plt.gca()
    
    # 为每个数据集绘制箱线图
    positions = np.arange(len(plot_data))
    box_width = 0.8
    
    box_plot = ax.boxplot(
        [d['y'] for d in plot_data],
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=True,  # 显示异常值
        medianprops={'color': 'black', 'linewidth': 1.5},
        boxprops={'edgecolor': 'black', 'linewidth': 1},
        whiskerprops={'color': 'black', 'linewidth': 1},
        capprops={'color': 'black', 'linewidth': 1}
    )
    
    # 设置箱子颜色
    for box, data in zip(box_plot['boxes'], plot_data):
        box.set(facecolor=data['color'], alpha=0.7)
    
    # 设置x轴标签和场景分隔线
    plt.xticks(positions, [d['x'] for d in plot_data], rotation=90, ha='right')
    
    # 添加场景分隔线和标签
    current_pos = 0
    scene_boundaries = []
    scene_midpoints = []
    current_scene = None
    
    for i, data in enumerate(plot_data):
        if current_scene != data['scene']:
            if i > 0:
                # 添加场景分隔线
                plt.axvline(x=i-0.5, color='black', linestyle='-', alpha=0.7, linewidth=1.5)
                scene_boundaries.append(i-0.5)
            current_scene = data['scene']
            current_pos = i
            scene_midpoints.append((current_scene, current_pos))
    
    # 添加参考线
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # 设置标题和轴标签
    plt.title(f'Site Score Distribution for {target_modification}', fontsize=14)
    plt.ylabel('Site Score', fontsize=12)
    
    # 在上方添加场景标签
    y_min, y_max = plt.ylim()
    for scene, i in scene_midpoints:
        # 查找该场景的结束位置
        end_idx = next((idx for idx, (s, _) in enumerate(ordered_datasets) if s != scene and idx > i), len(ordered_datasets))
        mid = (i + end_idx - 1) / 2
        plt.text(mid, y_max * 1.05, scene, ha='center', fontsize=11, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 添加图例
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', alpha=0.7, 
                                    label=scene) for scene, color in scene_colors.items()]
    plt.legend(handles=legend_elements, title='Scene', loc='best')
    
    # 添加数据量标注
    for i, (scene, dataset) in enumerate(ordered_datasets):
        count = ordered_stats[(ordered_stats['Scene'] == scene) & 
                             (ordered_stats['Dataset'] == dataset)]['count'].values[0]
        plt.text(i, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.02, f"n={count}", ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 保存图片
    safe_mod_name = target_modification.replace('[', '_').replace(']', '_').replace('/', '_')
    output_file = os.path.join(boxplot_dir, f"{safe_mod_name}_boxplot.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存数据
    output_data_file = os.path.join(boxplot_dir, f"{safe_mod_name}_data.csv")
    all_data.to_csv(output_data_file, index=False)
    
    # 保存统计信息
    output_stats_file = os.path.join(boxplot_dir, f"{safe_mod_name}_stats.csv")
    ordered_stats.to_csv(output_stats_file, index=False)
    
    print(f"生成箱线图完成，保存到: {output_file}")
    print(f"统计数据已保存到: {output_stats_file}")
    
    return output_file, all_data

def main(file_path=None, meta_file=None, data_root_dir=None, output_dir="results", 
         plot_modification=None, plot_boxplot=False, plot_cancer=False):
    """主函数：根据参数处理单个文件或批量处理"""
    # 如果提供了meta_file，先加载MHC类型信息
    if meta_file:
        load_meta_data(meta_file)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    if meta_file:
        # 批量处理模式
        print(f"批量处理元数据文件: {meta_file}")
        process_metadata_file(meta_file, data_root_dir, output_dir)
        
        # 生成趋势比较图
        print("生成趋势比较图...")
        plot_trend_comparison(output_dir)
        
        # 生成癌症数据集热图
        if plot_cancer:
            print("生成癌症数据集热图...")
            plot_cancer_heatmap(output_dir, meta_file)
        
        # 如果指定了要绘制箱线图
        if plot_boxplot or plot_modification:
            plot_modification_boxplot(output_dir, plot_modification)
            
    elif file_path:
        # 单文件处理模式
        print(f"处理单个文件: {file_path}")
        
        # 分析文件
        result = analyze_modifications(file_path)
        
        if result is None:
            print("分析失败，请检查文件格式")
            return
        
        # 保存分析结果
        file_name = os.path.basename(file_path)
        file_prefix = os.path.splitext(file_name)[0]
        result_file = os.path.join(output_dir, f"{file_prefix}_analysis.csv")
        result['data'].to_csv(result_file, index=False)
        
        # 保存site score报告
        report_file = os.path.join(output_dir, f"{file_prefix}_site_scores.txt")
        write_site_scores_report(result, report_file)
        
        print(f"分析结果已保存到: {result_file}")
        print(f"Site Score报告已保存到: {report_file}")
        
        # 打印摘要信息
        print("\n修饰类型Site Score摘要:")
        for mod_type, scores in result['mod_site_scores'].items():
            print(f"{mod_type}: Site Score={scores['site_score']:.4f}, 修饰数量={scores['count']}")
            
        # 如果指定了要绘制箱线图
        if plot_boxplot or plot_modification:
            print("单个文件模式下无法绘制箱线图，需要多个数据集")
            
        # 癌症热图不适用于单文件模式
        if plot_cancer:
            print("单个文件模式下无法生成癌症热图，需要多个数据集和元数据文件")
    else:
        print("错误: 必须提供单个文件路径或元数据文件路径")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="肽段修饰位点分析工具")
    
    # 定义模式参数组
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--file", "-f", help="单个.spectra文件路径")
    mode_group.add_argument("--meta", "-m", help="元数据文件(meta_pass.txt)路径")
    
    # 其他参数
    parser.add_argument("--data_dir", "-d", default=".", help="数据集根目录，用于批处理模式")
    parser.add_argument("--output_dir", "-o", default="site_score_results", help="输出目录")
    parser.add_argument("--plot_boxplot", "-p", action="store_true", help="绘制修饰类型的箱线图")
    parser.add_argument("--modification", "-mod", help="指定要绘制箱线图的修饰类型")
    parser.add_argument("--plot_cancer", "-c", action="store_true", help="生成癌症数据集的热图，按疾病类型分组")
    
    args = parser.parse_args()
    
    # 调用主函数
    main(
        file_path=args.file,
        meta_file=args.meta,
        data_root_dir=args.data_dir,
        output_dir=args.output_dir,
        plot_modification=args.modification,
        plot_boxplot=args.plot_boxplot,
        plot_cancer=args.plot_cancer
    )