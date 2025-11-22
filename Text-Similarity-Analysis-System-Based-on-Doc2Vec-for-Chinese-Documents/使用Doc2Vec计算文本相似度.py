# coding:utf-8
import os
import sys
import gensim
import numpy as np
import codecs
import jieba
import collections
import csv
import platform
import chardet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -----------------------------
# 小说明（你可以跳过）：
# - 保持原始 Doc2Vec 训练逻辑和超参（你最开始的代码）
# - 增加：text_explanation.txt, metrics_table.csv, radar.png, pca.png, tsne.png
# - 处理编码（msr_training.txt 为 ANSI），web_text1/2 为 utf-8
# - 处理 matplotlib 中文显示（尽量加载系统中文字体）
# -----------------------------

# -----------------------------
# 0. 尝试设置中文字体（Windows 常用）
# -----------------------------
def setup_chinese_font():
    try:
        # Windows 常见字体路径
        win_fonts = [
            "C:/Windows/Fonts/simhei.ttf",    # 黑体
            "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
            "C:/Windows/Fonts/msyh.ttf",
            "C:/Windows/Fonts/simsun.ttc"
        ]
        for fp in win_fonts:
            if os.path.exists(fp):
                import matplotlib.font_manager as fm
                fm.fontManager.addfont(fp)
                plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
                plt.rcParams['axes.unicode_minus'] = False
                print("已加载中文字体：", fp)
                return
    except Exception as e:
        print("加载中文字体时出错：", e)
    # fallback
    plt.rcParams['axes.unicode_minus'] = False
    print("未找到常见中文字体，图片可能出现中文缺字警告。")

setup_chinese_font()

# -----------------------------
# 1. 读取 msr_training.txt（考虑 ANSI/GBK 情况）
# -----------------------------
msr_path = "msr_training.txt"
if not os.path.exists(msr_path):
    raise FileNotFoundError(f"找不到 {msr_path}，请把 msr_training.txt 放在脚本同级目录。")

# If the user said 'ANSI', on Windows that's 'mbcs'; fallback to chardet if not Windows
detected_enc = None
if platform.system().lower().startswith("win"):
    # mbcs maps to the ANSI code page on Windows (usually GBK for Chinese Windows)
    try:
        with open(msr_path, "r", encoding="mbcs", errors="strict") as f:
            # try quick read
            _ = f.readline()
        detected_enc = "mbcs"
        print("msr_training.txt 使用 Windows ANSI (mbcs) 编码读取。")
    except Exception:
        detected_enc = None

if detected_enc is None:
    # fallback: use chardet to detect
    with open(msr_path, "rb") as f:
        raw = f.read()
    det = chardet.detect(raw)
    detected_enc = det.get("encoding", "utf-8")
    print(f"检测到 msr_training.txt 编码：{detected_enc}（chardet）")

# 读取语料（用检测到的编码，errors='ignore' 保证不因单个异常字节崩溃）
with open(msr_path, "r", encoding=detected_enc, errors="ignore") as cf:
    docs = cf.readlines()

print("语料行数：", len(docs))
print("示例行（第11行）：", docs[10] if len(docs) > 10 else "(行数不足)")

# -----------------------------
# 2. 为句子做标记（保留你原始的 split 空格逻辑）
#    如果语料不是用空格分词，请把下面注释替换为 jieba.lcut(text)
# -----------------------------
train_data = []
for i, text in enumerate(docs):
    # 如果 msr_training.txt 是已经用空格分词的语料（通常 MSR 训练集是分好词的），保持 split
    word_list = text.split(' ')
    # 若你的语料不是空格分词，使用以下替代： word_list = jieba.lcut(text)
    word_list[-1] = word_list[-1].strip()
    document = gensim.models.doc2vec.TaggedDocument(word_list, tags=[i])
    train_data.append(document)

# -----------------------------
# 3. 定义并训练 Doc2Vec（完全保留你原来的超参）
# -----------------------------
model = gensim.models.doc2vec.Doc2Vec(train_data,
                                     min_count=1,
                                     window=3,
                                     vector_size=256,
                                     negative=10,
                                     workers=4,
                                     alpha=0.001,
                                     min_alpha=0.001)

model.train(train_data, total_examples=model.corpus_count, epochs=10)
model.save('model_msr')
print("Doc2Vec 模型训练并保存为 model_msr。")

# -----------------------------
# 4. 定义相似度函数
# -----------------------------
def sim_cal(vector_1, vector_2):
    v1 = np.linalg.norm(vector_1)
    v2 = np.linalg.norm(vector_2)
    if v1 != 0 and v2 != 0:
        return float(np.dot(vector_1, vector_2) / (v1 * v2))
    else:
        return 0.0

# -----------------------------
# 5. 推断文档向量（web_text1/2 为 UTF-8）
# -----------------------------
def infer_doc_vector(file_name, model, assumed_encoding="utf-8"):
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"找不到 {file_name}")
    # web_text1/2 你已说明是 UTF-8，直接按 UTF-8 打开（若有 BOM 或异常字节，使用 errors='ignore'）
    with open(file_name, "r", encoding=assumed_encoding, errors="ignore") as f:
        lines = f.readlines()
    tokens = [w for x in lines for w in jieba.cut(x.strip())]
    vec = model.infer_vector(tokens)  # 如果想更稳定，可以加 steps=20
    text_joined = " ".join(tokens)
    return vec, text_joined, tokens

p1 = "web_text1.txt"
p2 = "web_text2.txt"

p1_vec, p1_text_joined, p1_tokens = infer_doc_vector(p1, model, assumed_encoding="utf-8")
p2_vec, p2_text_joined, p2_tokens = infer_doc_vector(p2, model, assumed_encoding="utf-8")

doc2vec_score = sim_cal(p1_vec, p2_vec)
print("Doc2Vec 余弦相似度：", doc2vec_score)

# -----------------------------
# 6. 多指标计算：TF-IDF 相似度、Jaccard（关键词重叠率）、长度相似度
# -----------------------------
# TF-IDF 余弦相似度（基于 jieba 切词后的空格拼接文本）
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform([p1_text_joined, p2_text_joined])
tfidf_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

# Jaccard（基于分词集合）
set1 = set([w for w in p1_tokens if len(w) > 0])
set2 = set([w for w in p2_tokens if len(w) > 0])
jaccard = float(len(set1 & set2) / len(set1 | set2)) if len(set1 | set2) > 0 else 0.0

# 文本长度相似度（1 - 长度差占比）
length_sim = 1.0 - abs(len(p1_text_joined) - len(p2_text_joined)) / max(len(p1_text_joined), len(p2_text_joined), 1)

# 额外指标：关键词向量中心距离（用简单方式：两向量欧氏距离的归一化）
vec_dist = np.linalg.norm(p1_vec - p2_vec)
# 归一化到 0..1（用 sigmoid 风格或 max_norm，这里用 1/(1+dist) 作为相似度替代）
vec_dist_sim = 1.0 / (1.0 + vec_dist)

# 综合相似度（简单平均）
metrics_list = {
    "Doc2Vec_cosine": doc2vec_score,
    "TFIDF_cosine": float(tfidf_score),
    "Jaccard": jaccard,
    "LengthSim": length_sim,
    "VecDistSim": vec_dist_sim
}
final_score = float(np.mean(list(metrics_list.values())))

# -----------------------------
# 7. 相似度等级标签（写入解释）
# -----------------------------
def score_label(s):
    if s >= 0.80:
        return "★★★★★（强相关）"
    elif s >= 0.60:
        return "★★★★☆（高度相似）"
    elif s >= 0.40:
        return "★★★☆☆（中度相关）"
    elif s >= 0.20:
        return "★★☆☆☆（弱相关）"
    else:
        return "★☆☆☆☆（无关）"

label = score_label(final_score)
print(f"综合相似度：{final_score:.4f}，等级：{label}")

# -----------------------------
# 8. 关键词分析（TOP10）
# -----------------------------
def top_k_words(tokens, k=10):
    cnt = collections.Counter([w for w in tokens if len(w) > 1])
    return cnt.most_common(k)

top1 = top_k_words(p1_tokens, 10)
top2 = top_k_words(p2_tokens, 10)
overlap_top = list(set([w for w,_ in top1]) & set([w for w,_ in top2]))

# -----------------------------
# 9. 输出文本解释 text_explanation.txt
# -----------------------------
with open("text_explanation.txt", "w", encoding="utf-8") as f:
    f.write("===== 文档相似度分析报告 =====\n\n")
    f.write("文件1: {}\n\n".format(p1))
    f.write("文件2: {}\n\n".format(p2))
    f.write("---- 单项指标 ----\n")
    for k,v in metrics_list.items():
        f.write(f"{k}: {v:.6f}\n")
    f.write(f"\n综合相似度 (平均): {final_score:.6f}\n")
    f.write(f"等级: {label}\n\n")
    f.write("---- 文档1 高频词 (top10) ----\n")
    for w,c in top1:
        f.write(f"{w}\t{c}\n")
    f.write("\n---- 文档2 高频词 (top10) ----\n")
    for w,c in top2:
        f.write(f"{w}\t{c}\n")
    f.write("\n---- 高频词重叠 ----\n")
    for w in overlap_top:
        f.write(w + "\n")
print("已输出 text_explanation.txt")

# -----------------------------
# 10. 多指标对比表 CSV 输出 metrics_table.csv
# -----------------------------
csv_path = "metrics_table.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["metric", "value"])
    writer.writerow(["Doc2Vec_cosine", metrics_list["Doc2Vec_cosine"]])
    writer.writerow(["TFIDF_cosine", metrics_list["TFIDF_cosine"]])
    writer.writerow(["Jaccard", metrics_list["Jaccard"]])
    writer.writerow(["LengthSim", metrics_list["LengthSim"]])
    writer.writerow(["VecDistSim", metrics_list["VecDistSim"]])
    writer.writerow(["FinalScore", final_score])
print("已输出", csv_path)

# -----------------------------
# 11. 雷达图 radar.png（使用 Doc2Vec, TFIDF, Jaccard, LengthSim 四项）
# -----------------------------
try:
    radar_labels = ["Doc2Vec", "TFIDF", "Jaccard", "LengthSim"]
    radar_scores = [metrics_list["Doc2Vec_cosine"], metrics_list["TFIDF_cosine"], metrics_list["Jaccard"], metrics_list["LengthSim"]]
    # close the loop
    radar_scores_loop = radar_scores + [radar_scores[0]]
    angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
    angles_loop = angles + [angles[0]]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles_loop, radar_scores_loop, 'o-', linewidth=2)
    ax.fill(angles_loop, radar_scores_loop, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles), radar_labels)
    ax.set_ylim(0,1)
    plt.title("文档相似度雷达图")
    plt.savefig("radar.png", bbox_inches='tight')
    plt.close()
    print("已输出 radar.png")
except Exception as e:
    print("画雷达图发生错误：", e)

# -----------------------------
# 12. PCA 可视化 pca.png
# -----------------------------
try:
    vecs = np.vstack([p1_vec, p2_vec])
    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(vecs)
    plt.figure(figsize=(6,5))
    plt.scatter(pca_points[:,0], pca_points[:,1])
    plt.text(pca_points[0,0], pca_points[0,1], "Text1")
    plt.text(pca_points[1,0], pca_points[1,1], "Text2")
    plt.title("PCA 文档向量可视化")
    plt.savefig("pca.png", bbox_inches='tight')
    plt.close()
    print("已输出 pca.png")
except Exception as e:
    print("PCA 可视化错误：", e)

# -----------------------------
# 13. t-SNE 可视化 tsne.png（自动调整 perplexity，保证 perplexity < n_samples）
# -----------------------------
try:
    vecs = np.vstack([p1_vec, p2_vec])
    n_samples = vecs.shape[0]
    # perplexity 必须 < n_samples，且一般建议 >=1；设置上限30
    perp = max(1, min(30, n_samples - 1))
    print("t-SNE 使用 perplexity =", perp)
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perp)
    tsne_points = tsne.fit_transform(vecs)
    plt.figure(figsize=(6,5))
    plt.scatter(tsne_points[:,0], tsne_points[:,1])
    plt.text(tsne_points[0,0], tsne_points[0,1], "Text1")
    plt.text(tsne_points[1,0], tsne_points[1,1], "Text2")
    plt.title("t-SNE 文档向量可视化")
    plt.savefig("tsne.png", bbox_inches='tight')
    plt.close()
    print("已输出 tsne.png")
except Exception as e:
    print("t-SNE 可视化错误（可能样本太少或库问题）：", e)

print("\n全部流程完成：已生成 text_explanation.txt, metrics_table.csv, radar.png, pca.png, tsne.png（如无错误）。")
