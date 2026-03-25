classdef FaceApp < handle
    % FaceApp 人脸识别交互界面 (GUI) 与主控业务逻辑
    % 升级版：支持自动构建人脸库、深度学习特征提取(ResNet-50)、实时防抖识别
    
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
        
        % 核心数据
        DatasetPath = ''
        
        % 深度学习特征数据
        DBFeatures   % 2048 x N 特征向量矩阵
        DBLabels     % 1 x N 标签数组
        DBPaths      % N x 1 路径数组
        
        TestDataFeatures % 2048 x M
        TestLabels
        TestPaths
        
        % 摄像头对象与实时防抖
        CamObj = []
        CamTimer = []
        IsCamRunning = false
        IsRealTimeEnabled = false
        
        HistoryLabels = {} % 存储最近 N 帧的识别结果
        SmoothFrames = 5   % 需要连续多少帧一致
        SimThreshold = 0.75 % 置信度阈值
        
        % 当前选中的待测图像
        CurrentTestImage = []
    end
    
    methods
        function obj = FaceApp()
            % 构造函数，初始化 GUI
            obj.createUI();
            obj.logMsg('系统初始化完成。请先构建人脸库，或直接加载已有的 Face_Database。');
        end
        
        function createUI(obj)
            % 创建主窗口
            obj.UIFigure = uifigure('Name', '高精度人脸识别系统 (ResNet-50 + 实时防抖)', 'Position', [100, 100, 1200, 800]);
            
            % 数据集与训练操作区
            obj.PanelData = uipanel(obj.UIFigure, 'Title', '1. 数据集与特征库管理', 'Position', [20, 600, 300, 180]);
            obj.BtnBuildDB = uibutton(obj.PanelData, 'Position', [20, 120, 260, 30], 'Text', '1. 从练习图片构建 Face_Database', 'ButtonPushedFcn', @(~, ~) obj.buildDatabase());
            obj.BtnLoadDB = uibutton(obj.PanelData, 'Position', [20, 80, 260, 30], 'Text', '2. 加载基准库并提取深度特征', 'ButtonPushedFcn', @(~, ~) obj.loadDatabase());
            obj.BtnTestBatch = uibutton(obj.PanelData, 'Position', [20, 40, 260, 30], 'Text', '3. 批量测试(如果划分了测试集)', 'ButtonPushedFcn', @(~, ~) obj.batchTest(), 'Enable', 'off');
            obj.LabelAcc = uilabel(obj.PanelData, 'Position', [20, 10, 260, 22], 'Text', '模型准确率: 等待测试...', 'FontWeight', 'bold');
            
            % 图像处理中间结果可视化区
            obj.PanelVisual = uipanel(obj.UIFigure, 'Title', '2. 图像对齐可视化', 'Position', [340, 480, 840, 300]);
            obj.AxesOriginal = uiaxes(obj.PanelVisual, 'Position', [10, 30, 250, 240]); title(obj.AxesOriginal, '原始图像');
            obj.AxesPreprocess = uiaxes(obj.PanelVisual, 'Position', [280, 30, 250, 240]); title(obj.AxesPreprocess, '对齐与裁剪 (224x224)');
            
            % 离线人脸点选识别区
            obj.PanelOffline = uipanel(obj.UIFigure, 'Title', '3. 离线样本识别区', 'Position', [20, 330, 300, 250]);
            obj.BtnSelectOffline = uibutton(obj.PanelOffline, 'Position', [20, 190, 260, 30], 'Text', '选择待测人脸图片', 'ButtonPushedFcn', @(~, ~) obj.selectOfflineImage(), 'Enable', 'off');
            obj.BtnRecognize = uibutton(obj.PanelOffline, 'Position', [20, 150, 260, 30], 'Text', '执行单样本深度识别', 'ButtonPushedFcn', @(~, ~) obj.recognizeSingle(), 'Enable', 'off');
            obj.AxesResult = uiaxes(obj.PanelOffline, 'Position', [20, 40, 120, 100]); title(obj.AxesResult, '匹配结果');
            obj.LabelResult = uilabel(obj.PanelOffline, 'Position', [150, 40, 130, 100], 'Text', '识别结果待定', 'WordWrap', 'on');
            
            % 摄像头实时人脸识别区
            obj.PanelCamera = uipanel(obj.UIFigure, 'Title', '4. 摄像头实时防抖识别区', 'Position', [340, 20, 400, 440]);
            obj.BtnToggleCam = uibutton(obj.PanelCamera, 'Position', [20, 380, 170, 30], 'Text', '启动/关闭摄像头', 'ButtonPushedFcn', @(~, ~) obj.toggleCamera());
            obj.BtnToggleRealTime = uibutton(obj.PanelCamera, 'Position', [210, 380, 170, 30], 'Text', '开启实时识别防抖', 'ButtonPushedFcn', @(~, ~) obj.toggleRealTime(), 'Enable', 'off');
            obj.AxesCamera = uiaxes(obj.PanelCamera, 'Position', [20, 60, 360, 310]); title(obj.AxesCamera, '实时画面预览');
            obj.LabelStatus = uilabel(obj.PanelCamera, 'Position', [20, 20, 360, 30], 'Text', '摄像头状态: 已关闭', 'FontColor', 'blue', 'FontWeight', 'bold');
            
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
            sourceDir = uigetdir(pwd, '选择包含练习图片的原始文件夹');
            if sourceDir == 0
                return;
            end
            targetDir = fullfile(pwd, 'Face_Database');
            obj.logMsg(['正在检测人脸、对齐并构建 Face_Database... 这可能需要一些时间。']);
            obj.logMsg(['目标路径: ', targetDir]);
            drawnow;
            
            ImagePreprocess.buildFaceDatabase(sourceDir, targetDir);
            obj.logMsg('Face_Database 构建完成！请点击第二步加载基准库。');
        end
        
        function loadDatabase(obj)
            % 加载 Face_Database，提取所有深度特征
            folderPath = uigetdir(pwd, '选择 Face_Database 文件夹 (或按人员分类的已对齐人脸库)');
            if folderPath == 0
                return;
            end
            obj.DatasetPath = folderPath;
            obj.logMsg(['已选择基准库路径: ', folderPath]);
            
            files = [dir(fullfile(folderPath, '**', '*.jpg')); dir(fullfile(folderPath, '**', '*.png'))];
            if isempty(files)
                obj.logMsg('未在选定文件夹中找到图片！');
                return;
            end
            
            labels = cell(length(files), 1);
            paths = cell(length(files), 1);
            
            for i = 1:length(files)
                [folder, ~, ~] = fileparts(fullfile(files(i).folder, files(i).name));
                [~, parentFolder] = fileparts(folder);
                
                % 智能提取标签：如果文件夹就是根目录或Face_Database，则从文件名提取人名
                [~, rootName] = fileparts(obj.DatasetPath);
                if strcmp(parentFolder, rootName) || strcmp(parentFolder, 'Face_Database') || strcmp(parentFolder, 'date_train')
                    [baseName, ~] = fileparts(files(i).name);
                    personName = regexprep(baseName, '[\d\s_\-]+$', '');
                    if isempty(personName)
                        personName = 'Unknown';
                    end
                    labels{i} = personName;
                else
                    labels{i} = parentFolder; 
                end
                
                paths{i} = fullfile(files(i).folder, files(i).name);
            end
            
            % 划分：由于是基准库，如果不做批量测试，就全部作为 DB。这里为了兼容批量测试，做简单划分 7:3
            uniqueLabels = unique(labels);
            numClasses = length(uniqueLabels);
            
            trainIdx = [];
            testIdx = [];
            
            for i = 1:numClasses
                clsIdx = find(strcmp(labels, uniqueLabels{i}));
                numSamples = length(clsIdx);
                if numSamples >= 2
                    splitPoint = round(numSamples * 0.7);
                    trainIdx = [trainIdx; clsIdx(1:splitPoint)];
                    testIdx = [testIdx; clsIdx(splitPoint+1:end)];
                else
                    trainIdx = [trainIdx; clsIdx]; % 只有1张则只作训练
                end
            end
            
            obj.logMsg(sprintf('共发现 %d 张图片，划分: %d 作为基准特征库, %d 作为测试集。', length(files), length(trainIdx), length(testIdx)));
            obj.logMsg('正在加载 ResNet-50 并提取深度特征，请稍候...');
            drawnow;
            
            N = length(trainIdx);
            obj.DBFeatures = zeros(2048, N);
            obj.DBLabels = cell(1, N);
            obj.DBPaths = cell(N, 1);
            
            for i = 1:N
                idx = trainIdx(i);
                imgPath = paths{idx};
                img = imread(imgPath);
                
                % 如果图片不是 224x224，需要 resize
                if size(img,1) ~= 224 || size(img,2) ~= 224
                    img = imresize(img, [224, 224]);
                end
                
                feat = ClassifierCore.extractDeepFeature(img);
                obj.DBFeatures(:, i) = feat;
                obj.DBLabels{i} = labels{idx};
                obj.DBPaths{i} = imgPath;
                
                if mod(i, 10) == 0
                    obj.logMsg(sprintf('基准库特征提取进度: %d/%d', i, N));
                end
            end
            
            M = length(testIdx);
            obj.TestDataFeatures = zeros(2048, M);
            obj.TestLabels = cell(1, M);
            obj.TestPaths = cell(M, 1);
            
            for i = 1:M
                idx = testIdx(i);
                imgPath = paths{idx};
                img = imread(imgPath);
                if size(img,1) ~= 224 || size(img,2) ~= 224
                    img = imresize(img, [224, 224]);
                end
                feat = ClassifierCore.extractDeepFeature(img);
                obj.TestDataFeatures(:, i) = feat;
                obj.TestLabels{i} = labels{idx};
                obj.TestPaths{i} = imgPath;
            end
            
            obj.logMsg('特征提取完成！系统已准备好进行识别。');
            
            % 展示第一张图
            if ~isempty(obj.DBPaths)
                img = imread(obj.DBPaths{1});
                imshow(img, 'Parent', obj.AxesOriginal);
                imshow(imresize(img, [224,224]), 'Parent', obj.AxesPreprocess);
            end
            
            obj.BtnTestBatch.Enable = 'on';
            obj.BtnSelectOffline.Enable = 'on';
            if obj.IsCamRunning
                obj.BtnToggleRealTime.Enable = 'on';
            end
        end
        
        function batchTest(obj)
            % 批量测试离线样本
            obj.logMsg('开始批量识别测试集...');
            tic;
            
            M = size(obj.TestDataFeatures, 2);
            if M == 0
                obj.logMsg('测试集为空。');
                return;
            end
            
            predictedLabels = cell(1, M);
            for i = 1:M
                [bestMatchIndex, ~] = ClassifierCore.classify(obj.TestDataFeatures(:, i), obj.DBFeatures);
                predictedLabels{i} = obj.DBLabels{bestMatchIndex};
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
            [fileName, pathName] = uigetfile({'*.jpg;*.png', '图像文件 (*.jpg, *.png)'}, '选择待测人脸图片');
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
                obj.logMsg('请先加载基准库并导入待测图像。');
                return;
            end
            
            obj.logMsg('正在进行检测、对齐与深度特征识别...');
            tic;
            
            % 检测与对齐
            [alignedFace, success] = ImagePreprocess.detectAndAlignFace(obj.CurrentTestImage);
            if ~success
                obj.logMsg('未在图像中检测到人脸！');
                return;
            end
            imshow(alignedFace, 'Parent', obj.AxesPreprocess);
            
            % 提取特征与分类
            testFeature = ClassifierCore.extractDeepFeature(alignedFace);
            [bestMatchIndex, minDistance] = ClassifierCore.classify(testFeature, obj.DBFeatures);
            
            elapsedTime = toc;
            
            matchLabel = obj.DBLabels{bestMatchIndex};
            matchPath = obj.DBPaths{bestMatchIndex};
            
            % 余弦距离转换为相似度 (可选，仅用于显示)
            simScore = 1 - minDistance;
            
            obj.logMsg(sprintf('识别完成！耗时: %.2f 秒', elapsedTime));
            if simScore < obj.SimThreshold
                obj.logMsg(sprintf('置信度较低 (%.2f)，可能为 Unknown', simScore));
                matchLabel = 'Unknown';
            else
                obj.logMsg(sprintf('匹配人员身份: %s (相似度: %.2f)', matchLabel, simScore));
            end
            
            % 在界面上显示结果与匹配图
            if exist(matchPath, 'file')
                matchImg = imread(matchPath);
                imshow(matchImg, 'Parent', obj.AxesResult);
                [~, matchName, matchExt] = fileparts(matchPath);
                resultText = sprintf('匹配身份:\n%s\n\n相似度: %.2f', matchLabel, simScore);
            else
                resultText = sprintf('匹配身份:\n%s', matchLabel);
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
                    clear obj.CamObj;
                    obj.CamObj = [];
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
                    
                    % 降低定时器频率，给深度学习预测留出时间 (如 0.2s)
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
                obj.HistoryLabels = {}; % 清空历史队列
                obj.logMsg('已开启实时识别防抖。请面向摄像头...');
            end
        end
        
        function updateCameraPreview(obj)
            if obj.IsCamRunning && ~isempty(obj.CamObj)
                try
                    img = snapshot(obj.CamObj);
                    imshow(img, 'Parent', obj.AxesCamera);
                    
                    % 如果开启了实时识别
                    if obj.IsRealTimeEnabled && ~isempty(obj.DBFeatures)
                        obj.processRealTimeFrame(img);
                    end
                catch
                    % 忽略偶尔的掉帧
                end
            end
        end
        
        function processRealTimeFrame(obj, img)
            % 1. 人脸检测与对齐
            [alignedFace, success] = ImagePreprocess.detectAndAlignFace(img);
            if ~success
                obj.LabelStatus.Text = '未检测到人脸';
                return;
            end
            
            % 可选：在原图上更新小视图
            imshow(alignedFace, 'Parent', obj.AxesPreprocess);
            
            % 2. 特征提取与识别
            testFeature = ClassifierCore.extractDeepFeature(alignedFace);
            [bestMatchIndex, minDistance] = ClassifierCore.classify(testFeature, obj.DBFeatures);
            
            bestLabel = obj.DBLabels{bestMatchIndex};
            simScore = 1 - minDistance;
            
            % 3. 置信度判定
            if simScore < obj.SimThreshold
                currentResult = 'Unknown';
            else
                currentResult = bestLabel;
            end
            
            % 4. 时间平滑防抖机制
            obj.HistoryLabels{end+1} = currentResult;
            if length(obj.HistoryLabels) > obj.SmoothFrames
                obj.HistoryLabels(1) = []; % 保持队列长度
            end
            
            % 统计众数
            if length(obj.HistoryLabels) == obj.SmoothFrames
                uniqueLabels = unique(obj.HistoryLabels);
                counts = cellfun(@(x) sum(strcmp(obj.HistoryLabels, x)), uniqueLabels);
                [maxCount, idx] = max(counts);
                
                % 如果 80% 以上一致 (如 5 帧中有 4 帧相同)
                if maxCount >= ceil(obj.SmoothFrames * 0.8)
                    finalLabel = uniqueLabels{idx};
                    obj.LabelStatus.Text = sprintf('实时结果: %s (置信度: %.2f)', finalLabel, simScore);
                    obj.LabelStatus.FontColor = 'red';
                else
                    obj.LabelStatus.Text = '识别中...';
                    obj.LabelStatus.FontColor = 'blue';
                end
            else
                obj.LabelStatus.Text = '收集防抖数据...';
            end
        end
    end
end