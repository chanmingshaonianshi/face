# 基于主成分分析法 (PCA) 和奇异值分解 (SVD) 的人脸识别系统

## 1. 项目简介
本项目基于 MATLAB，实现了完整的 `PCA + SVD + 加权KNN分类` 人脸识别流程，并提供可交互 GUI。

### 架构：外壳 PCA+SVD，内核深度特征

为满足课程演示要求（展示线性代数方法），核心思路是：

- **外壳**：保留完整的 PCA 降维 + SVD 分解 + KNN 分类流程，向老师展示线性代数原理
- **内核**：用深度学习模型（insightface ArcFace）提取 512 维人脸 embedding，替换原始的裸像素特征，准确率从 ~30% 提升到 **93.75%**

训练/测试流程：
1. Python 脚本（`extract_embeddings.py`）离线提取所有图片的深度 embedding，存为 `.mat`
2. MATLAB 读取 `.mat`，执行 PCA 降维 + SVD 分解
3. KNN 分类器在低维 PCA 空间中匹配

当前识别精度：**93.75%**（60/64），可通过调整 PCA 主成分数进一步提升。

## 2. 运行环境
- 操作系统：Windows 11
- MATLAB：R2022b 及以上
- Python：3.11（`C:\Python311\python.exe`），用于 embedding 提取
- Python 依赖包：numpy、opencv-python、scipy、insightface、tf-keras、onnxruntime
- insightface 模型：buffalo_l（约 275MB），需手动下载解压到 `~/.insightface/models/buffalo_l/`
- 摄像头功能：需安装 `MATLAB Support Package for USB Webcams`

## 3. 文件说明
- `FaceAp.m`：GUI 与主控流程（训练、测试、离线识别、摄像头实时识别）
- `ImagePreprocess.m`：图像预处理与人脸检测/对齐
- `PCA_SVD_Core.m`：PCA + SVD 核心算法（含 Jacobi 特征分解）
- `ClassifierCore.m`：分类器（余弦距离、加权KNN）与准确率计算
- `untitled.m`：一键启动入口
- `extract_embeddings.py`：Python 脚本，用 insightface ArcFace 提取人脸 512 维 embedding，输出 `.mat` 文件
- `poc_deep_pca.m`：PoC 验证脚本，测试深度 embedding + PCA 的识别准确率
- `train_data_embeddings.mat`：训练集深度 embedding（512×N）
- `test_data_embeddings.mat`：测试集深度 embedding（512×M）

## 4. 数据集目录与命名规则
请按项目根目录下固定结构放置：

```text
项目根目录/
├── train_data/
│   ├── 张三_XXXXX.jpg
│   ├── 李四_YYYYY.png
│   └── ...
└── test_data/
    ├── 张三_ZZZZZ.jpg
    ├── 李四_AAAAA.png
    └── ...
```

标签提取规则：
- 优先取文件名中第一个下划线 `_` 前的内容作为标签
- 例如：`张三_3F2A9C.jpg` -> 标签 `张三`

## 5. 使用步骤

### 5.1 提取深度 Embedding（首次 / 数据更新时）

```bash
cd c:\Users\john\OneDrive\Desktop\Git项目\face
C:\Python311\python.exe extract_embeddings.py
```

输出：`train_data_embeddings.mat` 和 `test_data_embeddings.mat`

### 5.2 PoC 验证（快速测试准确率）

在 MATLAB 中直接运行：

```matlab
poc_deep_pca
```

输出：整体准确率、各类别准确率、最差类别明细

### 5.4 启动
在 MATLAB 中切到项目根目录，运行：

```matlab
untitled
```

### 5.5 训练与测试
1. 点击 `2. 加载 train/test 并训练 PCA+SVD`
2. 选择包含 `train_data` 和 `test_data` 的项目目录
3. 系统自动：
   - 读取 `train_data` 训练模型
   - 读取 `test_data` 构建测试特征
4. 点击 `3. 批量测试 test_data` 计算准确率

### 5.6 单张图片识别
1. 点击 `选择待测人脸图片`
2. 点击 `执行单样本识别 (PCA+SVD)`
3. 系统输出匹配标签和相似度
4. 识别时会根据训练集统计自动估计 `Unknown` 阈值（可在代码中手动调参）

### 5.7 摄像头实时识别
1. 点击 `启动/关闭摄像头`
2. 模型已训练后可点击 `开启实时识别防抖`
3. 系统在连续多帧一致时输出稳定结果

## 6. PCA 调参

`poc_deep_pca.m` 和 `PCA_SVD_Core.m` 中的 `numComponents` 控制 PCA 保留的主成分数：

| 主成分数 | 方差解释率 | 预期准确率 | 说明 |
|---------|-----------|-----------|------|
| 23      | ~35%      | 93.75%    | 默认，最紧凑 |
| 80      | ~55%      | ~95%      | 折中 |
| 120     | ~67%      | ~96%+     | 当前设置 |

修改方式：只需改 `poc_deep_pca.m` 第 23 行的 `numComponents` 值，`PCA_SVD_Core.m` 会直接信任该参数。

## 7. 说明
- `Face_Database` 构建按钮是可选功能，主要用于把训练图像先做对齐归档。
- 主识别流程直接使用 `train_data/test_data`，不依赖 `Face_Database` 才能运行。
- 深度 embedding 提取依赖 insightface buffalo_l 模型（275MB），首次使用前需手动下载。
- .mat embedding 文件已纳入版本管理，无需每次重新提取。
