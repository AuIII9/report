# Assignment 2：手写数字分类

从原始 MNIST 像素训练 Bernoulli Naive Bayes 和手工小型 Bayesian Network。模型仅依赖 NumPy，不使用现成分类器、网络库或自动结构学习。实验报告见 [report.md](report.md)。

## 安装与运行

建议 Python 3.10 或更新版本。在课程仓库根目录执行：

```bash
python -m pip install -r assignment2/requirements.txt
python assignment2/main.py --data-dir assignment2-data/mnist
python -m unittest discover -s assignment2 -p test_models.py -v
```

事先在本地解压课程数据包，使 `assignment2-data/mnist/` 包含四个 `.gz` 数据文件和 `manifest.json`。脚本不会联网下载。默认会核对 manifest 中的 SHA256（如果存在），并检查完整训练/测试集的形状。

运行会更新 `metrics.json`、`predictions.npz` 和 `figures/` 中的五张图。`report.md` 是本次默认参数实验对应的分析报告；若更改模型或 `--alpha` 后重跑，需同步更新报告。`--output-dir` 可指定另一输出目录。

## 文件说明

| 文件 | 用途 |
|---|---|
| `naive_bayes.py` | 课程协议类 `NaiveBayesClassifier`，可独立加载，仅依赖 NumPy |
| `bayes_net.py` | 课程协议类 `BayesNetClassifier` 和 `extract_features`，可独立加载，仅依赖 NumPy |
| `main.py` | IDX 读取、训练、手工指标计算、图像输出 |
| `test_models.py` | 手算概率、CPT、指标和接口校验 |
| `metrics.json` | 完整精度指标、逐类指标、混淆矩阵、用时和数据校验值 |
| `predictions.npz` | 10,000 条真实标签和两个模型预测，不含原始图像 |
| `figures/` | DAG、两张混淆矩阵、两张误分类样例图 |
| `report.md` | 中文实验报告 |

只把 `assignment2/` 目录放入与教师共享的课程仓库；不要添加原始数据或数据压缩包。工作区根目录的 `.gitignore` 已排除本地数据。当前文件尚未推送至任何远端。

## 实现辅助说明

本目录代码和报告使用 OpenAI Codex 辅助编写，报告保留了说明。课程要求独立完成；提交者应理解并核对实现、实验与分析，按课程对 AI 辅助的规定处理署名和提交。
