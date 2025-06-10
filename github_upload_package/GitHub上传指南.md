# GitHub 上传指南

## 项目概述
本项目是一个**肽段分组分析工具**，包含以下主要功能：
- 🧬 肽段理化性质分析
- 🔬 修饰类型分组比较
- 📊 统计检验和可视化
- 🔄 断点续传功能
- 📈 数据集级别趋势分析

## 🚀 最新更新内容

### 新增核心功能
1. **肽段分组分析** (`peptide_group_analysis.py`)
   - PTM-only vs PTM+WT vs WT-only 三组比较
   - 每种修饰类型独立分析
   - 自动统计检验和显著性标注

2. **断点续传功能**
   - 自动保存分析进度到JSON文件
   - 缓存每个数据集的分析结果
   - 支持从任意中断点继续分析

3. **可视化改进**
   - 合并图表：4个理化性质在一张2x2图上
   - 更窄的小提琴图：突出组间差异
   - 显著性标注：自动添加p值和***/**/*符号

4. **数据集级别趋势图**
   - 多数据集间的变化趋势
   - 综合对比视图
   - 动态点大小反映肽段数量

## 📋 准备工作

### 1. 安装 Git
如果您的系统还没有安装 Git：

**Windows:**
1. 访问 https://git-scm.com/download/win
2. 下载并安装 Git for Windows
3. 安装时选择默认选项即可

**验证安装:**
```bash
git --version
```

### 2. 配置 Git（首次使用）
```bash
git config --global user.name "您的GitHub用户名"
git config --global user.email "您的GitHub邮箱"
```

## 📤 上传步骤

### 方法一：使用Git命令行（推荐）

#### 1. 检查当前状态
```bash
git status
```

#### 2. 添加新文件
```bash
# 添加所有新文件和更改
git add .

# 或者选择性添加文件
git add peptide_group_analysis.py
git add demo_*.py
git add *.md
```

#### 3. 提交更改
```bash
git commit -m "🧬 添加肽段分组分析功能套件

✨ 新功能:
- 肽段分组分析: PTM-only vs PTM+WT vs WT-only
- 断点续传: 支持中断后继续分析
- 合并图表: 4个理化性质在一张图上
- 显著性标注: 自动p值和统计符号
- 数据集趋势图: 多数据集比较分析

📁 新增文件:
- peptide_group_analysis.py: 核心分析模块
- demo_*.py: 6个演示脚本
- GitHub上传指南.md: 详细使用说明

🔧 技术改进:
- 图表数量减少75%
- 自动统计检验
- 智能缓存机制
- 专业可视化标准"
```

#### 4. 推送到 GitHub
```bash
# 推送到主分支
git push origin main

# 如果是第一次推送到新仓库
git push -u origin main
```

### 方法二：使用GitHub网页界面

#### 1. 创建GitHub仓库
1. 登录 [GitHub](https://github.com)
2. 点击右上角的 "+" 号，选择 "New repository"
3. 填写仓库信息：
   - **Repository name**: `peptide-group-analysis`
   - **Description**: `肽段分组分析工具：PTM检测模式与理化性质关系研究`
   - **Public/Private**: 选择Public（公开）
4. 点击 "Create repository"

#### 2. 上传文件
将以下核心文件拖拽到上传区域：
```
peptide_group_analysis.py
demo_*.py (6个演示脚本)
tumor_analysis_core.py
peptide_properties_analyzer.py
requirements.txt
README.md
LICENSE
.gitignore
```

## 📁 主要新增文件

### 🔬 核心分析模块
- **`peptide_group_analysis.py`** - 主要分析模块，包含所有新功能

### 🎯 演示脚本套件
- **`demo_peptide_group_analysis.py`** - 基础功能演示
- **`demo_single_modification_analysis.py`** - 单修饰分析演示
- **`demo_combined_plots.py`** - 合并图表演示
- **`demo_significance_plots.py`** - 显著性标注演示
- **`demo_dataset_trends.py`** - 趋势图演示
- **`demo_resume_analysis.py`** - 断点续传演示

### 📚 文档文件
- **`GitHub上传指南.md`** - 本文件，详细上传说明

### 📂 推荐的完整文件结构
```
peptide-group-analysis/
├── peptide_group_analysis.py         # 🔬 核心分析模块
├── tumor_analysis_core.py            # 🧬 肿瘤分析引擎
├── peptide_properties_analyzer.py    # ⚗️ 理化性质计算器
├── demo_peptide_group_analysis.py    # 🎯 基础演示
├── demo_single_modification_analysis.py  # 🔬 单修饰演示
├── demo_combined_plots.py            # 📊 合并图表演示
├── demo_significance_plots.py        # 📈 显著性演示
├── demo_dataset_trends.py            # 📉 趋势图演示
├── demo_resume_analysis.py           # 🔄 断点续传演示
├── requirements.txt                  # 📦 依赖包列表
├── README.md                        # 📖 项目说明
├── LICENSE                          # ⚖️ 开源许可证
├── .gitignore                       # 🚫 Git忽略文件
└── GitHub上传指南.md                 # 📋 上传指南
```

## 🎯 功能特点详解

### 1. 🧬 肽段分组分析
```
PTM-only:  只检测到修饰形态的肽段
PTM + WT:  同序列同时检测到修饰和未修饰肽段
WT-only:   只检测到未修饰形态的肽段
```

### 2. 🔄 断点续传功能
- ✅ 自动保存分析进度到 `cache/analysis_progress.json`
- 💾 缓存每个数据集结果到 `cache/{dataset_id}_results.pkl`
- 🚀 智能跳过已完成的数据集
- 🛡️ 容错能力强，程序崩溃后可恢复

### 3. 📊 可视化改进
- **合并图表**: 从36个图表减少到9个图表（75%减少）
- **更窄小提琴图**: width=0.6，突出组间差异
- **显著性标注**: 自动添加 *** (p<0.001), ** (p<0.01), * (p<0.05)
- **专业布局**: 2×2网格，符合学术发表标准

### 4. 📈 统计分析
- **Kruskal-Wallis**: 总体差异检验
- **Mann-Whitney U**: 两两比较检验
- **效应量**: Cliff's delta计算
- **自动标注**: p值和显著性符号

## 🚀 使用方法

### 基础分析
```python
from peptide_group_analysis import PeptideGroupAnalyzer

# 创建分析器（默认启用断点续传）
analyzer = PeptideGroupAnalyzer()

# 运行完整分析
results = analyzer.run_full_analysis()
```

### 演示分析
```python
# 限制数据集数量的演示
results = analyzer.run_full_analysis(max_datasets=3)
```

### 断点续传管理
```python
# 启用断点续传（默认）
analyzer = PeptideGroupAnalyzer(enable_resume=True)

# 禁用断点续传
analyzer = PeptideGroupAnalyzer(enable_resume=False)

# 清除缓存重新开始
analyzer.clear_cache()

# 检查数据集是否已完成
is_done = analyzer.is_dataset_completed('PXD007596')
```

## 📂 输出文件结构

```
peptide_group_analysis_results/
├── group_stats.csv              # 分组统计结果
├── stat_tests.csv               # 统计检验结果
├── analysis_report.txt          # 详细分析报告
├── fig/                         # 图表目录
│   ├── combined_properties_*.png    # 合并理化性质图表
│   ├── dataset_trends_*.png         # 数据集级别趋势图
│   ├── comprehensive_trends_*.png   # 综合趋势图
│   └── global_heatmap_*.png         # 全局热图
└── cache/                       # 缓存目录
    ├── analysis_progress.json       # 分析进度
    └── *_results.pkl               # 缓存的分析结果
```

## 🎯 性能优势

### 图表优化
- **旧版本**: 1个数据集 × 9种修饰 × 4个性质 = **36个图表**
- **新版本**: 1个数据集 × 9种修饰 = **9个合并图表**
- **节省**: **27个图表 (75%减少)**

### 断点续传优势
- ⚡ **避免重复计算**: 节省大量时间
- 🛡️ **容错能力强**: 程序崩溃后可恢复
- 📈 **增量分析**: 支持逐步添加数据集
- 💾 **资源高效**: 充分利用已有结果

## ⚠️ 注意事项

1. **📦 依赖包**: 确保安装所有必需的Python包
   ```bash
   pip install pandas numpy matplotlib seaborn scipy tqdm
   ```

2. **💾 存储空间**: 分析结果可能产生大量图表和缓存文件

3. **🧠 内存使用**: 大数据集分析需要足够内存（建议8GB+）

4. **🔄 首次运行**: 建议启用断点续传功能

5. **📊 数据集要求**: 需要包含Tumor样本的数据集

## 🌟 增加项目可见性

### 1. 添加标签（Topics）
在GitHub仓库页面，点击设置图标，添加相关标签：
- `proteomics`
- `bioinformatics`
- `cancer-research`
- `peptide-analysis`
- `python`
- `mass-spectrometry`
- `ptm-analysis`
- `statistical-analysis`

### 2. 创建Release
1. 在仓库页面点击 "Releases"
2. 点击 "Create a new release"
3. 填写版本号（如 v2.0.0）和发布说明

## 🆘 技术支持

如有问题，请查看：
1. **演示脚本**: `demo_*.py` 中的使用示例
2. **分析报告**: `analysis_report.txt` 中的详细说明
3. **代码注释**: 源码中的技术细节
4. **GitHub Issues**: 在仓库中提交问题

## 🏆 项目亮点

- 🧬 **生物学意义**: 深入理解修饰肽的检测模式
- 📊 **统计严谨**: 非参数检验 + 效应量分析
- 🎨 **可视化专业**: 符合学术发表标准
- 🔧 **工程化**: 断点续传 + 缓存机制
- 📈 **可扩展**: 支持新数据集和修饰类型

---

**🎉 恭喜！您现在拥有了一个功能完整、性能优化、可视化专业的肽段分组分析工具！**
