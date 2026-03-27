# 基于主成分分析法 (PCA) 和奇异值分解 (SVD) 的人脸识别系统

## 1. 项目简介
本项目基于 MATLAB，实现了完整的 `PCA + SVD + 最近邻分类` 人脸识别流程，并提供可交互 GUI。  
当前代码主流程不依赖深度学习模型，核心识别路径为：

1. 图像预处理（灰度化、去噪、增强、缩放）
2. PCA+SVD 训练特征空间（训练集）
3. 特征投影与最近邻匹配（测试/离线/实时）

## 2. 运行环境
- 操作系统：Windows / macOS / Linux
- MATLAB：R2022b 及以上
- 摄像头功能：需安装 `MATLAB Support Package for USB Webcams`

## 3. 文件说明
- `FaceApp.m`：GUI 与主控流程（训练、测试、离线识别、摄像头实时识别）
- `ImagePreprocess.m`：图像预处理与人脸检测/对齐
- `PCA_SVD_Core.m`：PCA + SVD 核心算法（含 Jacobi 特征分解）
- `ClassifierCore.m`：最近邻分类与准确率计算
- `untitled.m`：一键启动入口

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

### 5.1 启动
在 MATLAB 中切到项目根目录，运行：

```matlab
untitled
```

### 5.2 训练与测试
1. 点击 `2. 加载 train/test 并训练 PCA+SVD`
2. 选择包含 `train_data` 和 `test_data` 的项目目录
3. 系统自动：
   - 读取 `train_data` 训练模型
   - 读取 `test_data` 构建测试特征
4. 点击 `3. 批量测试 test_data` 计算准确率

### 5.3 单张图片识别
1. 点击 `选择待测人脸图片`
2. 点击 `执行单样本识别 (PCA+SVD)`
3. 系统输出匹配标签和相似度

### 5.4 摄像头实时识别
1. 点击 `启动/关闭摄像头`
2. 模型已训练后可点击 `开启实时识别防抖`
3. 系统在连续多帧一致时输出稳定结果

## 6. 说明
- `Face_Database` 构建按钮是可选功能，主要用于把训练图像先做对齐归档。
- 主识别流程直接使用 `train_data/test_data`，不依赖 `Face_Database` 才能运行。
