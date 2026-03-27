classdef FaceApp < handle
    % FaceApp 人脸识别交互界面 (GUI) 与主控业务逻辑
    % 当前主流程：PCA + SVD 特征提取 + 最近邻(余弦距离)分类

    properties
        % UI 界面元素
        UIFigure
        PanelData
        PanelVisual
        PanelOffline
        PanelCamera
        PanelLog

        LogTextArea
        AxesOriginal
        AxesPreprocess
        AxesResult
        AxesCamera

        BtnBuildDB
        BtnLoadDB
        BtnTestBatch
        BtnSelectOffline
        BtnRecognize
        BtnToggleCam
        BtnToggleRealTime

        LabelAcc
        LabelResult
        LabelStatus

        % 数据路径
        DatasetPath = ''
        TrainFolder = ''
        TestFolder = ''

        % PCA + SVD 模型数据
        MeanFace
        EigenFaces
        DBFeatures   % K x N 训练特征矩阵
        DBLabels     % 1 x N 标签
        DBPaths      % N x 1 图片路径
        FeatureScale % K x 1 投影特征标准化尺度

        TestDataFeatures % K x M 测试特征矩阵
        TestLabels
        TestPaths

        TargetRows = 112
        TargetCols = 92
        NumComponents = 0
        KnnK = 5

        % 摄像头对象与实时防抖
        CamObj = []
        CamTimer = []
        IsCamRunning = false
        IsRealTimeEnabled = false

        HistoryLabels = {} % 存储最近 N 帧的识别结果
        SmoothFrames = 5   % 需要连续多少帧一致
        SimThreshold = 0.60 % 置信度阈值（余弦相似度）

        % 当前选中的待测图像
        CurrentTestImage = []
    end

    methods
        function obj = FaceApp()
            % 构造函数，初始化 GUI
            obj.createUI();
            obj.logMsg('系统初始化完成。请先加载 train_data / test_data 并训练 PCA+SVD 模型。');
        end

        function createUI(obj)
            % 创建主窗口
            obj.UIFigure = uifigure('Name', '人脸识别系统 (PCA+SVD + 实时防抖)', 'Position', [100, 100, 1200, 800]);

            % 数据集与训练操作区
            obj.PanelData = uipanel(obj.UIFigure, 'Title', '1. 数据集与模型管理', 'Position', [20, 600, 300, 180]);
            obj.BtnBuildDB = uibutton(obj.PanelData, 'Position', [20, 120, 260, 30], ...
                'Text', '1. 从 train_data 构建 Face_Database (可选)', ...
                'ButtonPushedFcn', @(~, ~) obj.buildDatabase());
            obj.BtnLoadDB = uibutton(obj.PanelData, 'Position', [20, 80, 260, 30], ...
                'Text', '2. 加载 train/test 并训练 PCA+SVD', ...
                'ButtonPushedFcn', @(~, ~) obj.loadDatabase());
            obj.BtnTestBatch = uibutton(obj.PanelData, 'Position', [20, 40, 260, 30], ...
                'Text', '3. 批量测试 test_data', ...
                'ButtonPushedFcn', @(~, ~) obj.batchTest(), 'Enable', 'off');
            obj.LabelAcc = uilabel(obj.PanelData, 'Position', [20, 10, 260, 22], ...
                'Text', '模型准确率: 等待测试...', 'FontWeight', 'bold');

            % 图像处理中间结果可视化区
            obj.PanelVisual = uipanel(obj.UIFigure, 'Title', '2. 图像预处理可视化', 'Position', [340, 480, 840, 300]);
            obj.AxesOriginal = uiaxes(obj.PanelVisual, 'Position', [10, 30, 250, 240]); title(obj.AxesOriginal, '原始图像');
            obj.AxesPreprocess = uiaxes(obj.PanelVisual, 'Position', [280, 30, 250, 240]); title(obj.AxesPreprocess, '预处理结果');

            % 离线人脸点选识别区
            obj.PanelOffline = uipanel(obj.UIFigure, 'Title', '3. 离线样本识别区', 'Position', [20, 330, 300, 250]);
            obj.BtnSelectOffline = uibutton(obj.PanelOffline, 'Position', [20, 190, 260, 30], ...
                'Text', '选择待测人脸图片', ...
                'ButtonPushedFcn', @(~, ~) obj.selectOfflineImage(), 'Enable', 'off');
            obj.BtnRecognize = uibutton(obj.PanelOffline, 'Position', [20, 150, 260, 30], ...
                'Text', '执行单样本识别 (PCA+SVD)', ...
                'ButtonPushedFcn', @(~, ~) obj.recognizeSingle(), 'Enable', 'off');
            obj.AxesResult = uiaxes(obj.PanelOffline, 'Position', [20, 40, 120, 100]); title(obj.AxesResult, '匹配结果');
            obj.LabelResult = uilabel(obj.PanelOffline, 'Position', [150, 40, 130, 100], ...
                'Text', '识别结果待定', 'WordWrap', 'on');

            % 摄像头实时人脸识别区
            obj.PanelCamera = uipanel(obj.UIFigure, 'Title', '4. 摄像头实时防抖识别区', 'Position', [340, 20, 400, 440]);
            obj.BtnToggleCam = uibutton(obj.PanelCamera, 'Position', [20, 380, 170, 30], ...
                'Text', '启动/关闭摄像头', 'ButtonPushedFcn', @(~, ~) obj.toggleCamera());
            obj.BtnToggleRealTime = uibutton(obj.PanelCamera, 'Position', [210, 380, 170, 30], ...
                'Text', '开启实时识别防抖', ...
                'ButtonPushedFcn', @(~, ~) obj.toggleRealTime(), 'Enable', 'off');
            obj.AxesCamera = uiaxes(obj.PanelCamera, 'Position', [20, 60, 360, 310]); title(obj.AxesCamera, '实时画面预览');
            obj.LabelStatus = uilabel(obj.PanelCamera, 'Position', [20, 20, 360, 30], ...
                'Text', '摄像头状态: 已关闭', 'FontColor', 'blue', 'FontWeight', 'bold');

            % 结果展示与日志区
            obj.PanelLog = uipanel(obj.UIFigure, 'Title', '5. 系统操作日志', 'Position', [760, 20, 420, 440]);
            obj.LogTextArea = uitextarea(obj.PanelLog, 'Position', [10, 10, 400, 400], 'Editable', 'off');
        end

        function logMsg(obj, msg)
            % 记录操作日志
            timestamp = datestr(now, 'HH:MM:SS');
            newMsg = sprintf('[%s] %s', timestamp, msg);
            currentText = obj.LogTextArea.Value;
            if isempty(currentText) || (iscell(currentText) && isempty(currentText{1}))
                obj.LogTextArea.Value = {newMsg};
            else
                if ischar(currentText)
                    currentText = {currentText};
                end
                obj.LogTextArea.Value = [currentText; {newMsg}];
            end
            scroll(obj.LogTextArea, 'bottom');
            drawnow;
        end

        function buildDatabase(obj)
            % 可选步骤：将 train_data 人脸对齐后导出到 Face_Database
            rootDir = uigetdir(pwd, '选择包含 train_data 的项目文件夹');
            if rootDir == 0
                return;
            end

            trainDir = fullfile(rootDir, 'train_data');
            if ~exist(trainDir, 'dir')
                obj.logMsg('未找到 train_data 文件夹，已取消构建。');
                return;
            end

            targetDir = fullfile(rootDir, 'Face_Database');
            obj.logMsg('正在检测人脸、对齐并构建 Face_Database...');
            obj.logMsg(['源路径: ', trainDir]);
            obj.logMsg(['目标路径: ', targetDir]);
            drawnow;

            ImagePreprocess.buildFaceDatabase(trainDir, targetDir);
            obj.logMsg('Face_Database 构建完成（可选步骤，不影响 PCA+SVD 主流程）。');
        end

        function loadDatabase(obj)
            % 加载 train_data / test_data，并训练 PCA + SVD 主流程
            rootDir = uigetdir(pwd, '选择包含 train_data / test_data 的项目文件夹');
            if rootDir == 0
                return;
            end

            trainDir = fullfile(rootDir, 'train_data');
            testDir = fullfile(rootDir, 'test_data');

            if ~exist(trainDir, 'dir')
                obj.logMsg('未找到 train_data 文件夹，请检查路径。');
                return;
            end

            obj.DatasetPath = rootDir;
            obj.TrainFolder = trainDir;
            if exist(testDir, 'dir')
                obj.TestFolder = testDir;
            else
                obj.TestFolder = '';
            end

            trainFiles = obj.listImageFiles(trainDir);
            if isempty(trainFiles)
                obj.logMsg('train_data 中未找到图片。');
                return;
            end

            N = length(trainFiles);
            d = obj.TargetRows * obj.TargetCols;

            obj.logMsg(sprintf('检测到训练样本 %d 张，开始预处理...', N));
            Xtrain = zeros(d, N);
            trainLabels = cell(1, N);
            trainPaths = cell(N, 1);

            for i = 1:N
                imgPath = trainFiles{i};
                img = imread(imgPath);
                [procImg, vec] = obj.preprocessForPCA(img, false);

                Xtrain(:, i) = vec;
                trainLabels{i} = obj.extractLabelFromPath(imgPath);
                trainPaths{i} = imgPath;

                if i == 1
                    imshow(img, 'Parent', obj.AxesOriginal);
                    imshow(procImg, 'Parent', obj.AxesPreprocess);
                end

                if mod(i, 20) == 0 || i == N
                    obj.logMsg(sprintf('训练样本预处理进度: %d/%d', i, N));
                end
            end

            % 维度上限采用较保守设置，减少高维噪声对泛化的影响
            numComponentsHint = min(max(20, round(N * 0.25)), max(1, N - 1));
            obj.logMsg('开始执行 PCA+SVD 特征训练...');
            [obj.MeanFace, obj.EigenFaces, ~, ~] = PCA_SVD_Core.computePCA_SVD(Xtrain, numComponentsHint);
            obj.DBFeatures = PCA_SVD_Core.project(Xtrain, obj.MeanFace, obj.EigenFaces);
            obj.FeatureScale = std(obj.DBFeatures, 0, 2);
            obj.FeatureScale(obj.FeatureScale < 1e-6) = 1;
            obj.DBFeatures = obj.DBFeatures ./ obj.FeatureScale;
            obj.DBFeatures = obj.l2NormalizeColumns(obj.DBFeatures);
            obj.DBLabels = trainLabels;
            obj.DBPaths = trainPaths;
            obj.NumComponents = size(obj.EigenFaces, 2);
            obj.logMsg(sprintf('模型训练完成。主成分个数: %d, KNN(k=%d)', obj.NumComponents, obj.KnnK));

            % 基于训练集内同类最近邻相似度估计 Unknown 阈值
            obj.SimThreshold = obj.estimateSimThresholdFromTrain();
            obj.logMsg(sprintf('自动阈值估计完成。SimThreshold=%.3f', obj.SimThreshold));

            if ~isempty(obj.TestFolder)
                testFiles = obj.listImageFiles(obj.TestFolder);
            else
                testFiles = {};
            end

            M = length(testFiles);
            if M > 0
                obj.logMsg(sprintf('检测到测试样本 %d 张，开始构建测试特征...', M));
                Xtest = zeros(d, M);
                obj.TestLabels = cell(1, M);
                obj.TestPaths = cell(M, 1);

                for i = 1:M
                    imgPath = testFiles{i};
                    img = imread(imgPath);
                    [~, vec] = obj.preprocessForPCA(img, false);

                    Xtest(:, i) = vec;
                    obj.TestLabels{i} = obj.extractLabelFromPath(imgPath);
                    obj.TestPaths{i} = imgPath;
                end

                obj.TestDataFeatures = PCA_SVD_Core.project(Xtest, obj.MeanFace, obj.EigenFaces);
                obj.TestDataFeatures = obj.TestDataFeatures ./ obj.FeatureScale;
                obj.TestDataFeatures = obj.l2NormalizeColumns(obj.TestDataFeatures);
                obj.logMsg('测试特征构建完成。');
                obj.BtnTestBatch.Enable = 'on';
            else
                obj.TestDataFeatures = [];
                obj.TestLabels = {};
                obj.TestPaths = {};
                obj.BtnTestBatch.Enable = 'off';
                obj.logMsg('未检测到 test_data，已跳过批量测试集构建。');
            end

            obj.logMsg('系统已就绪：可进行单图识别、批量测试或摄像头识别。');
            obj.BtnSelectOffline.Enable = 'on';
            if obj.IsCamRunning
                obj.BtnToggleRealTime.Enable = 'on';
            end
        end

        function batchTest(obj)
            % 批量测试 test_data 样本
            obj.logMsg('开始批量识别 test_data...');
            tic;

            M = size(obj.TestDataFeatures, 2);
            if M == 0
                obj.logMsg('测试集为空。');
                return;
            end

            predictedLabels = cell(1, M);
            for i = 1:M
                [bestLabel, ~, ~] = ClassifierCore.classifyKNNByLabel( ...
                    obj.TestDataFeatures(:, i), obj.DBFeatures, obj.DBLabels, obj.KnnK);
                predictedLabels{i} = char(bestLabel);
            end

            testStrLabels = string(obj.TestLabels);
            predStrLabels = string(predictedLabels);
            [overallAcc, ~] = ClassifierCore.calcAccuracy(testStrLabels, predStrLabels);

            elapsedTime = toc;
            obj.logMsg(sprintf('批量测试完成！耗时: %.2f 秒', elapsedTime));
            obj.logMsg(sprintf('整体识别正确率: %.2f%%', overallAcc));

            obj.LabelAcc.Text = sprintf('模型准确率: %.2f%%', overallAcc);
            obj.LabelAcc.FontColor = 'red';
        end

        function selectOfflineImage(obj)
            [fileName, pathName] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', '图像文件 (*.jpg, *.jpeg, *.png, *.bmp)'}, '选择待测人脸图片');
            if fileName == 0
                return;
            end
            imgPath = fullfile(pathName, fileName);
            obj.CurrentTestImage = imread(imgPath);
            imshow(obj.CurrentTestImage, 'Parent', obj.AxesOriginal);
            obj.logMsg(['已导入待测图像: ', fileName]);

            if ~isempty(obj.DBFeatures)
                obj.BtnRecognize.Enable = 'on';
            end
        end

        function recognizeSingle(obj)
            if isempty(obj.CurrentTestImage) || isempty(obj.DBFeatures)
                obj.logMsg('请先训练 PCA+SVD 模型并导入待测图像。');
                return;
            end

            obj.logMsg('正在进行单样本识别...');
            tic;

            [procImg, vec] = obj.preprocessForPCA(obj.CurrentTestImage, true);
            imshow(procImg, 'Parent', obj.AxesPreprocess);

            testFeature = PCA_SVD_Core.project(vec, obj.MeanFace, obj.EigenFaces);
            testFeature = testFeature ./ obj.FeatureScale;
            testFeature = obj.l2NormalizeColumns(testFeature);
            [bestLabel, bestMatchIndex, simScore] = ClassifierCore.classifyKNNByLabel( ...
                testFeature, obj.DBFeatures, obj.DBLabels, obj.KnnK);

            elapsedTime = toc;
            matchLabel = char(bestLabel);
            matchPath = obj.DBPaths{bestMatchIndex};
            simScore = max(0, simScore);

            obj.logMsg(sprintf('识别完成！耗时: %.2f 秒', elapsedTime));
            if simScore < obj.SimThreshold
                obj.logMsg(sprintf('置信度较低 (%.2f)，判定为 Unknown', simScore));
                displayLabel = 'Unknown';
            else
                displayLabel = matchLabel;
                obj.logMsg(sprintf('匹配人员身份: %s (相似度: %.2f)', displayLabel, simScore));
            end

            if exist(matchPath, 'file')
                matchImg = imread(matchPath);
                imshow(matchImg, 'Parent', obj.AxesResult);
                resultText = sprintf('匹配身份:\n%s\n\n相似度: %.2f', displayLabel, simScore);
            else
                resultText = sprintf('匹配身份:\n%s', displayLabel);
            end

            obj.LabelResult.Text = resultText;
            obj.LabelResult.FontColor = 'blue';
            obj.LabelResult.FontWeight = 'bold';
        end

        function toggleCamera(obj)
            if obj.IsCamRunning
                % 关闭
                if ~isempty(obj.CamTimer)
                    stop(obj.CamTimer);
                    delete(obj.CamTimer);
                    obj.CamTimer = [];
                end
                if ~isempty(obj.CamObj)
                    camTmp = obj.CamObj; %#ok<NASGU>
                    obj.CamObj = [];
                    clear camTmp;
                end
                cla(obj.AxesCamera);
                obj.LabelStatus.Text = '摄像头状态: 已关闭';
                obj.IsCamRunning = false;
                obj.IsRealTimeEnabled = false;
                obj.BtnToggleRealTime.Enable = 'off';
                obj.BtnToggleRealTime.Text = '开启实时识别防抖';
                obj.logMsg('已关闭摄像头。');
            else
                % 启动
                if exist('webcam', 'file') ~= 2
                    obj.logMsg('未检测到 webcam。请安装 MATLAB Support Package for USB Webcams。');
                    return;
                end
                try
                    obj.CamObj = webcam(1);
                    obj.IsCamRunning = true;
                    obj.LabelStatus.Text = '摄像头状态: 运行中';
                    if ~isempty(obj.DBFeatures)
                        obj.BtnToggleRealTime.Enable = 'on';
                    end
                    obj.logMsg('摄像头启动成功，正在实时预览...');

                    % 定时器频率 0.2s，兼顾计算与流畅度
                    obj.CamTimer = timer('ExecutionMode', 'fixedRate', 'Period', 0.2, ...
                        'TimerFcn', @(~, ~) obj.updateCameraPreview());
                    start(obj.CamTimer);
                catch e
                    obj.logMsg(['无法启动摄像头: ', e.message]);
                end
            end
        end

        function toggleRealTime(obj)
            if obj.IsRealTimeEnabled
                obj.IsRealTimeEnabled = false;
                obj.BtnToggleRealTime.Text = '开启实时识别防抖';
                obj.logMsg('已关闭实时识别防抖。');
            else
                obj.IsRealTimeEnabled = true;
                obj.BtnToggleRealTime.Text = '关闭实时识别防抖';
                obj.HistoryLabels = {};
                obj.logMsg('已开启实时识别防抖。请面向摄像头...');
            end
        end

        function updateCameraPreview(obj)
            if obj.IsCamRunning && ~isempty(obj.CamObj)
                try
                    img = snapshot(obj.CamObj);
                    imshow(img, 'Parent', obj.AxesCamera);

                    if obj.IsRealTimeEnabled && ~isempty(obj.DBFeatures)
                        obj.processRealTimeFrame(img);
                    end
                catch
                    % 忽略偶发掉帧
                end
            end
        end

        function processRealTimeFrame(obj, img)
            [procImg, vec] = obj.preprocessForPCA(img, true);
            imshow(procImg, 'Parent', obj.AxesPreprocess);

            testFeature = PCA_SVD_Core.project(vec, obj.MeanFace, obj.EigenFaces);
            testFeature = testFeature ./ obj.FeatureScale;
            testFeature = obj.l2NormalizeColumns(testFeature);
            [bestLabel, ~, simScore] = ClassifierCore.classifyKNNByLabel( ...
                testFeature, obj.DBFeatures, obj.DBLabels, obj.KnnK);

            bestLabel = char(bestLabel);
            simScore = max(0, simScore);

            if simScore < obj.SimThreshold
                currentResult = 'Unknown';
            else
                currentResult = bestLabel;
            end

            obj.HistoryLabels{end+1} = currentResult;
            if length(obj.HistoryLabels) > obj.SmoothFrames
                obj.HistoryLabels(1) = [];
            end

            if length(obj.HistoryLabels) == obj.SmoothFrames
                uniqueLabels = unique(obj.HistoryLabels);
                counts = cellfun(@(x) sum(strcmp(obj.HistoryLabels, x)), uniqueLabels);
                [maxCount, idx] = max(counts);

                if maxCount >= ceil(obj.SmoothFrames * 0.8)
                    finalLabel = uniqueLabels{idx};
                    obj.LabelStatus.Text = sprintf('实时结果: %s (相似度: %.2f)', finalLabel, simScore);
                    obj.LabelStatus.FontColor = 'red';
                else
                    obj.LabelStatus.Text = '识别中...';
                    obj.LabelStatus.FontColor = 'blue';
                end
            else
                obj.LabelStatus.Text = '收集防抖数据...';
                obj.LabelStatus.FontColor = 'blue';
            end
        end
    end

    methods (Access = private)
        function files = listImageFiles(~, folderPath)
            patterns = {'*.jpg', '*.jpeg', '*.png', '*.bmp'};
            files = {};
            for i = 1:length(patterns)
                tempFiles = dir(fullfile(folderPath, '**', patterns{i}));
                for j = 1:length(tempFiles)
                    files{end+1, 1} = fullfile(tempFiles(j).folder, tempFiles(j).name); %#ok<AGROW>
                end
            end

            if ~isempty(files)
                files = sort(files);
            end
        end

        function label = extractLabelFromPath(~, imgPath)
            [~, baseName, ~] = fileparts(imgPath);

            token = regexp(baseName, '^(.+?)_', 'tokens', 'once');
            if ~isempty(token)
                label = token{1};
                return;
            end

            % 兼容无下划线命名：去除末尾数字和连接符
            label = regexprep(baseName, '[\d\s_\-]+$', '');
            if isempty(label)
                label = 'Unknown';
            end
        end

        function [procImg, vec] = preprocessForPCA(obj, img, useFaceDetect)
            % 训练/测试集使用稳定统一预处理；待识别图像优先做人脸检测与对齐
            if nargin < 3
                useFaceDetect = true;
            end

            if ~useFaceDetect
                procImg = ImagePreprocess.fullProcess(img, obj.TargetRows, obj.TargetCols);
            else
                aligned = [];
                success = false;
                try
                    [aligned, success] = ImagePreprocess.detectAndAlignFace(img);
                catch
                    success = false;
                end

                if success
                    grayImg = aligned;
                else
                    grayImg = ImagePreprocess.toGray(img);
                    grayImg = ImagePreprocess.cropCenter(grayImg, 0.70);
                end

                grayImg = ImagePreprocess.denoise(grayImg);
                grayImg = ImagePreprocess.histEq(grayImg);
                resized = ImagePreprocess.resizeImg(grayImg, obj.TargetRows, obj.TargetCols);
                procImg = ImagePreprocess.normalizePixels(resized);
            end

            vec = ImagePreprocess.vectorizeImg(procImg);
            vec = obj.normalizeVector(vec);
        end

        function outVec = normalizeVector(~, vec)
            % 向量标准化：去均值 + 标准差归一 + L2 归一，提升光照鲁棒性
            outVec = double(vec);
            m = mean(outVec);
            outVec = outVec - m;

            s = std(outVec);
            if s > 1e-8
                outVec = outVec / s;
            end

            n = norm(outVec);
            if n > 0
                outVec = outVec / n;
            end
        end

        function F = l2NormalizeColumns(~, F)
            % 对特征矩阵每一列做 L2 归一，稳定余弦相似度
            [~, M] = size(F);
            for i = 1:M
                n = norm(F(:, i));
                if n > 0
                    F(:, i) = F(:, i) / n;
                end
            end
        end

        function threshold = estimateSimThresholdFromTrain(obj)
            % 根据训练集“同类最近邻相似度”自适应估计 Unknown 阈值
            N = size(obj.DBFeatures, 2);
            if N < 2
                threshold = 0.55;
                return;
            end

            sims = [];
            for i = 1:N
                thisLabel = string(obj.DBLabels{i});
                bestSim = -inf;
                for j = 1:N
                    if j == i
                        continue;
                    end
                    if string(obj.DBLabels{j}) ~= thisLabel
                        continue;
                    end
                    sim = obj.DBFeatures(:, i)' * obj.DBFeatures(:, j);
                    if sim > bestSim
                        bestSim = sim;
                    end
                end
                if bestSim > -inf
                    sims(end+1) = bestSim; %#ok<AGROW>
                end
            end

            if isempty(sims)
                threshold = 0.55;
                return;
            end

            mu = mean(sims);
            sigma = std(sims);
            threshold = mu - 2.2 * sigma;
            threshold = min(max(threshold, 0.40), 0.85);
        end
    end
end
