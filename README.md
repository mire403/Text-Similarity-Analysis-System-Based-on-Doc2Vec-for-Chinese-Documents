# 🚀 基于 Doc2Vec 的中文文本相似度分析系统

Text-Similarity-Analysis-System-Based-on-Doc2Vec-for-Chinese-Documents

**一个可复现、可扩展、带可视化输出的中文文本相似度分析项目。**
尤其适合作业提交、课程实验、科研 Demo、GitHub 项目展示。

## 📌 项目简介 | Project Overview

本项目实现了一个**多维度的中文文本相似度分析系统**，利用 Microsoft Research Segment 中文分词语料库训练**Doc2Vec 文档向量模型**，并结合 TF-IDF、Jaccard、文本长度、向量距离等多个指标，对两篇中文网页文本进行全面比较，并以图表方式进行可视化展示。

- 🧠 不仅提供结果，还提供背后的语义结构解释。
- 📈 不仅计算相似度，还提供多维度雷达图、PCA、t-SNE 可视化。
- 📄 还输出一份文本解释报告（text_explanation.txt），方便上交作业或论文引用。

## ✨ 功能亮点 | Features

| 功能 | 说明 |
|------|-----------|
| 🔍 Doc2Vec 文档向量建模 | 完整训练流程 + 文档向量推断 |
| 📊 多指标相似度计算 | Doc2Vec、TF-IDF、Jaccard、长度相似度、欧氏距离 |
| 🌈 雷达图可视化 | 展示多个指标的相似度构成 |
| 🎨 PCA / t-SNE 可视化 | 拍出“文本语义空间”两张高质量图 |
| 📝 文本解释报告输出 | 自动生成 text_explanation.txt（用于报告/论文） |
| 📁 输出 CSV 指标表 | metrics_table.csv 便于导入 Excel 或论文作图 |
| 🧩 编码自动识别处理 | ANSI + UTF-8 混合文件自动处理 |
| 🎯 语义 + 词汇 + 写作特征全覆盖 | 多角度解释文本接近度 |
	
## 🧪 项目效果展示 | Results

### 📌 文档相似度雷达图（radar.png）

（图：radar.png）

解释：

Doc2Vec 相似度最高 → 语义高度相近

Jaccard 最低 → 词汇差异明显

TF-IDF 中等 → 表达方式不同但关键词分布接近

🎯 PCA 文档向量可视化（pca.png）

（图：pca.png）

解释：

两个点距离表示语义差异程度

整体相对靠近 → 两篇文本主题一致

🌌 t-SNE 文档可视化（tsne.png）

（图：tsne.png）

解释：

t-SNE 会把极少样本尽可能拉开，视觉“空旷”是正常现象

关键在：文档 A 与 B 的相对位置

📝 文本解释报告（text_explanation.txt）

项目自动生成，包含：

Doc2Vec_cosine: 0.868125
TFIDF_cosine: 0.510651
Jaccard: 0.158385
LengthSim: 0.320892
FinalScore: 0.499465 (★★★☆☆ 中度相关)

Top keywords in Text1: ...
Top keywords in Text2: ...
Keywords intersection: ...


可直接用于：

作业报告

论文分析

项目展示

🧠 技术原理 | Technical Principles
1️⃣ Doc2Vec 模型

来自 Gensim，实现 Paragraph Vector (Distributed Memory, DM) 方法。

特点：
✔ 捕获语义信息
✔ 学习上下文
✔ 文本向量维度固定（如 256D）

2️⃣ TF-IDF 词频统计

词袋模型，用于衡量“关键词分布是否相似”。

3️⃣ Jaccard 文本集合重叠度

衡量高频词集合的重叠程度。
较敏感 → 反映文本词汇风格差异。

4️⃣ 文本长度相似度

保持更客观的对比，避免长文对短文造成偏差。

5️⃣ PCA / t-SNE

降维可视化文档向量，让抽象的 256D 向量变成可视图像。

🧱 项目结构 | Project Structure
📂 project
 ├── msr_training.txt          # Doc2Vec 训练语料（ANSI）
 ├── web_text1.txt             # 文本 1（UTF-8）
 ├── web_text2.txt             # 文本 2（UTF-8）
 ├── 使用Doc2Vec计算文本相似度.py
 ├── text_explanation.txt       # 文本解释报告（自动生成）
 ├── metrics_table.csv          # 指标表格（自动生成）
 ├── radar.png                  # 雷达图（自动生成）
 ├── pca.png                    # PCA 降维图（自动生成）
 └── tsne.png                   # t-SNE 图（自动生成）

▶ 使用方式 | How To Use
1️⃣ 安装环境
conda create -n gensim python=3.8
conda activate gensim
pip install gensim jieba numpy scikit-learn matplotlib chardet

2️⃣ 放置数据文件

把三个文件放在同目录：

msr_training.txt

web_text1.txt

web_text2.txt

3️⃣ 运行主程序
python 使用Doc2Vec计算文本相似度.py

4️⃣ 查看输出文件

运行后自动生成：

text_explanation.txt

metrics_table.csv

radar.png

pca.png

tsne.png

📌 项目流程图（Mermaid）
flowchart LR
    A[加载 msr_training 语料] --> B[jieba 分词]
    B --> C[训练 Doc2Vec 文档向量模型]
    C --> D1[推断 Text1 向量]
    C --> D2[推断 Text2 向量]
    D1 --> E[多指标相似度分析]
    D2 --> E
    E --> F[生成雷达图 radar.png]
    E --> G[生成 PCA / t-SNE 可视化]
    E --> H[生成 metrics_table.csv]
    E --> I[生成 text_explanation.txt]

📈 示例输出（metrics_table.csv）
Metric,Value
Doc2VecCosine,0.868125
TFIDFSim,0.510651
Jaccard,0.158385
LengthSim,0.320892
FinalScore,0.499465

🧩 Todo List

 支持 SimCSE / BERT 文本向量

 增加 keyword cloud（词云）

 增加多文本批量评估

 加入 Flask Web 界面展示

 制作可交互 Notebook 版本

⭐ Star Support

如果你觉得这个项目对你有帮助，请给仓库点一个 ⭐ Star！
你的鼓励是我继续优化此项目的最大动力 😊
