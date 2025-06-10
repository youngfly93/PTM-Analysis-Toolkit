# 🧬 PTM分析工具包

## 📋 项目概述

这是一个专业的**蛋白质翻译后修饰(PTM)分析工具包**，包含三个核心分析项目：

### 🎯 三个核心项目

1. **peptide_group_analysis.py** - 肽段分组分析(原版)
   - 针对特定修饰类型的分组逻辑
   - PTM-only vs PTM+WT vs WT-only 三组比较

2. **peptide_group_analysis_modified.py** - 肽段分组分析(修改版)  
   - 考虑任意修饰类型的新分组逻辑
   - 更严格的分组定义

3. **tumor_vs_normal_ptm_analysis.py** - 肿瘤vs正常样本PTM分析
   - 比较PTM在肿瘤vs正常样本中的相对效应
   - 统计分析和可视化

### 🔧 核心依赖模块

- **peptide_properties_analyzer.py** - 理化性质计算器
- **tumor_analysis_core.py** - 核心数据处理引擎

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行分析
```bash
# 运行项目1
python peptide_group_analysis.py

# 运行项目2  
python peptide_group_analysis_modified.py

# 运行项目3
python tumor_vs_normal_ptm_analysis.py
```

## 📊 功能特点

- ✅ **专业的PTM分析**: 支持9种常见修饰类型
- ✅ **统计严谨**: 非参数检验 + 效应量分析
- ✅ **可视化专业**: 符合学术发表标准
- ✅ **断点续传**: 支持中断后继续分析
- ✅ **模块化设计**: 易于扩展和维护

## 📁 项目结构

```
PTM-Analysis-Toolkit/
├── peptide_group_analysis.py           # 项目1核心脚本
├── peptide_group_analysis_modified.py  # 项目2核心脚本  
├── tumor_vs_normal_ptm_analysis.py     # 项目3核心脚本
├── peptide_properties_analyzer.py      # 理化性质计算器
├── tumor_analysis_core.py              # 核心数据处理引擎
├── requirements.txt                    # 依赖包列表
├── README.md                          # 项目说明
└── docs/                              # 详细文档
```

## 🔬 分析原理

### 肽段分组逻辑
- **PTM-only**: 只检测到修饰形态的肽段
- **PTM + WT**: 同序列同时检测到修饰和未修饰肽段  
- **WT-only**: 只检测到未修饰形态的肽段

### 理化性质分析
- 分子量 (Molecular Weight)
- 等电点 (pI)
- 疏水性 (Hydrophobicity)
- 净电荷 (Net Charge)

## 📈 输出结果

每个项目都会生成：
- 📊 统计分析结果 (CSV格式)
- 📈 可视化图表 (PNG格式)
- 📄 详细分析报告 (TXT格式)

## 🆘 技术支持

如有问题，请查看：
1. 详细文档: `docs/` 目录
2. GitHub Issues: 在仓库中提交问题
3. 代码注释: 源码中的技术细节

## 📜 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🏆 引用

如果您在研究中使用了本工具，请引用：
```
PTM Analysis Toolkit: A comprehensive tool for post-translational modification analysis
```

---

**🎉 感谢使用PTM分析工具包！**
