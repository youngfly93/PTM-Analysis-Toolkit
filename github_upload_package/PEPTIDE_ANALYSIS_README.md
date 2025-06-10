# 肽段理化性质分析工具

这是一个基于Pyteomics的肽段理化性质分析工具，专门用于分析包含修饰信息的.spectra文件。

## 功能特点

- **支持修饰肽段分析**: 解析unimod格式的修饰信息（如"11,Oxidation[M];"）
- **多种理化性质计算**: 
  - 等电点(pI)
  - 电荷(charge)
  - 疏水性指数（Kyte-Doolittle和Eisenberg标度）
  - 分子量
- **修饰校正**: 基于修饰表对理化性质进行校正
- **批量处理**: 支持单文件和批量文件处理
- **统计报告**: 自动生成详细的分析报告

## 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install pyteomics pandas numpy
```

## 使用方法

### 1. 命令行使用

#### 处理单个文件
```bash
python peptide_properties_analyzer.py "PXD008570_human/pFind-Filtered_pFindTask1_result.spectra" -s -o results
```

#### 批量处理
```bash
python peptide_properties_analyzer.py "*/*.spectra" -o batch_results
```

### 2. Python脚本使用

```python
from peptide_properties_analyzer import PeptidePropertiesAnalyzer

# 创建分析器
analyzer = PeptidePropertiesAnalyzer()

# 处理单个文件
result_df = analyzer.process_spectra_file(
    "your_file.spectra", 
    "output_results.csv"
)

# 批量处理
analyzer.process_multiple_files("*/*.spectra", "results_dir")
```

### 3. 运行示例

```bash
python example_usage.py
```

## 输入文件格式

脚本期望输入的.spectra文件包含以下列：
- `Sequence`: 肽段序列
- `Charge`: 电荷态
- `Modification`: 修饰信息（unimod格式）

示例修饰格式：
- `11,Oxidation[M];` - 第11位甲硫氨酸氧化
- `4,Phospho[S];8,Acetyl[K];` - 第4位丝氨酸磷酸化，第8位赖氨酸乙酰化

## 输出结果

### 主要输出列

- `sequence`: 肽段序列
- `length`: 肽段长度
- `molecular_weight`: 分子量 (Da)
- `base_pi`: 基础等电点
- `corrected_pi_kd`: 修饰校正后等电点
- `charge_at_ph7`: pH 7.0时的电荷
- `base_hydrophobicity_kd`: 基础疏水性（Kyte-Doolittle）
- `corrected_hydrophobicity_kd`: 修饰校正后疏水性
- `modifications`: 修饰信息
- `num_modifications`: 修饰数量

### 输出文件

1. **单文件结果**: `{filename}_properties.csv`
2. **合并结果**: `combined_peptide_properties.csv`
3. **统计报告**: `summary_report.txt`

## 支持的修饰类型

基于您提供的修饰表，目前支持以下修饰：

| 修饰类型 | 目标氨基酸 | ΔpKa | Δ疏水性 |
|---------|-----------|------|---------|
| Oxidation[M] | M | 0 | -0.77 |
| Acetyl[K] | K | -0.32 | -4.06 |
| Phospho[S/T/Y] | S/T/Y | -6.3 | -0.81/-1.5/-1.34 |
| Deamidated[N/Q] | N/Q | 0 | -1.6/-0.39 |
| Methyl[K] | K | 0.05 | -3.76 |
| Dimethyl[K] | K | -0.55 | -3.96 |
| Trimethyl[K] | K | -1.00 | -4.06 |

## 文件结构

```
.
├── peptide_properties_analyzer.py  # 主分析脚本
├── example_usage.py               # 使用示例
├── requirements.txt               # 依赖包列表
├── PEPTIDE_ANALYSIS_README.md     # 说明文档
└── flat_file                     # 修饰信息参考文件
```

## 注意事项

1. **文件编码**: 确保输入文件使用UTF-8编码
2. **内存使用**: 大文件处理时可能需要较多内存
3. **修饰格式**: 修饰信息必须符合unimod格式
4. **依赖版本**: 建议使用最新版本的Pyteomics

## 故障排除

### 常见问题

1. **ImportError: No module named 'pyteomics'**
   ```bash
   pip install pyteomics
   ```

2. **文件读取错误**
   - 检查文件路径是否正确
   - 确认文件格式为tab分隔的文本文件

3. **修饰解析失败**
   - 检查修饰格式是否正确
   - 确认修饰名称在支持列表中

### 日志信息

脚本会输出详细的日志信息，包括：
- 文件读取进度
- 处理进度
- 错误信息
- 结果保存位置

## 扩展功能

如需添加新的修饰类型，请修改`_load_modification_table()`方法中的修饰字典。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。
