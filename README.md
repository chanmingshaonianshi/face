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

实时识别流程：
- **单图识别**：MATLAB 将待识别图片复制到项目目录（避免中文路径问题），调用 `single_embedding.py` 提取 embedding 并写入临时文件，再从文件读取结果
- **摄像头实时识别**：通过 `embedding_server.py` 启动持久化 Python 子进程（模型只加载一次），MATLAB 通过文件交换协议逐帧通信（`_server_req.txt` / `_server_rep.txt`），彻底避免 stdin 中文编码问题，大幅降低延迟；独立深色窗口实时显示摄像头画面和当前帧识别人物，主 GUI 同步显示平滑后的识别状态和日志

当前识别精度：**93.75%**（60/64），可通过调整 PCA 主成分数进一步提升。

## 2. 运行环境
- 操作系统：Windows 11
- MATLAB：R2022b 及以上
- Python：3.11（`C:\Python311\python.exe`），用于 embedding 提取
- Python 依赖包：numpy、opencv-python、scipy、insightface、tf-keras、onnxruntime
- insightface 模型：buffalo_l（约 275MB），需手动下载解压到 `~/.insightface/models/buffalo_l/`
- 摄像头功能：需安装 `MATLAB Support Package for USB Webcams`

## 3. 文件说明
- `FaceApp.m`：GUI 主程序（左控制面板 + 右显示区布局，深度 embedding 集成）
- `ImagePreprocess.m`：图像预处理与人脸检测/对齐
- `PCA_SVD_Core.m`：PCA + SVD 核心算法（含 Jacobi 特征分解）
- `ClassifierCore.m`：分类器（余弦距离、加权KNN）与准确率计算
- `untitled.m`：一键启动入口
- `extract_embeddings.py`：批量提取训练集/测试集 embedding，输出 `.mat` 文件
- `single_embedding.py`：单图 embedding 提取（供 GUI 离线识别调用，写文件方式避免中文路径问题）
- `embedding_server.py`：持久化 Python 进程（供 GUI 摄像头实时识别，通过文件交换协议通信，模型只加载一次）
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

### 5.3 启动 GUI
在 MATLAB 中切到项目根目录，运行：

```matlab
untitled
```

### 5.4 加载模型并识别
1. 点击 `1. 加载模型`，选择项目根目录（含 `train_data_embeddings.mat`）
2. 点击 `2. 选择图片`，选择待识别的人脸图片
3. 点击 `3. 开始识别`，显示匹配结果和相似度

### 5.5 批量测试
加载模型后，点击 `4. 批量测试` 计算 test_data 整体准确率

### 5.6 摄像头实时识别
1. 点击 `5. 开启摄像头` 启动预览
2. 点击 `开启实时识别` 启用人脸识别（自动启动持久化 Python 进程，模型只加载一次）
3. 独立深色窗口实时显示摄像头画面、当前帧识别人物和相似度
4. 主 GUI 状态栏显示多帧平滑后的稳定结果；Server 无响应时会自动重启（10 秒冷却）
5. 关闭摄像头、关闭实时识别或关闭独立窗口时，持久进程自动终止

## 6. PCA 调参

`poc_deep_pca.m` 和 `PCA_SVD_Core.m` 中的 `numComponents` 控制 PCA 保留的主成分数：

| 主成分数 | 方差解释率 | 预期准确率 | 说明 |
|---------|-----------|-----------|------|
| 23      | ~35%      | 93.75%    | 默认，最紧凑 |
| 80      | ~55%      | ~95%      | 折中 |
| 120     | ~67%      | ~96%+     | 当前设置 |

修改方式：只需改 `poc_deep_pca.m` 第 23 行的 `numComponents` 值，`PCA_SVD_Core.m` 会直接信任该参数。

## 7. 说明
- 深度 embedding 提取依赖 insightface buffalo_l 模型（275MB），首次使用前需手动下载。
- .mat embedding 文件已纳入版本管理，无需每次重新提取。
- GUI 单图识别：调用 `single_embedding.py`，将 embedding 写入临时文件（避免中文路径和 stdout 编码问题），MATLAB 通过 `readEmbeddingFile` 从文件读取并校验 512 维向量。
- GUI 摄像头实时识别：通过 `embedding_server.py` 启动持久化 Python 子进程（模型只加载一次），MATLAB 通过文件交换协议（`_server_req.txt` / `_server_rep.txt`）逐帧通信，避免 stdin 中文编码问题，大幅降低延迟；独立窗口每帧刷新画面和当前识别结果，主 GUI 继续显示平滑后的稳定结果。
- Python 路径硬编码为 `C:\Python311\python.exe`，如需修改请编辑 `FaceApp.m` 第 29 行。

## 8. 最近验证
- 2026-06-05：修复 GUI 单图识别缺失 `readEmbeddingFile` 导致的临时 embedding 无法读取问题。
- 2026-06-05：修复摄像头实时识别独立窗口不刷新当前画面/人物名的问题；独立窗口现在每帧更新，主 GUI 保留稳定结果显示。
- 已使用 `D:\Matlab\bin\matlab.exe` 验证单图识别、实时识别 server 通路和独立窗口模拟帧显示。
