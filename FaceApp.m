classdef FaceApp < handle
    % FaceApp 人脸识别交互界面 (GUI) 与主控业务逻辑
    % 实现数据集导入、PCA+SVD 训练、结果可视化、离线点选识别与摄像头实时识别
    
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
        AxesMeanFace
        AxesEigenFace
        AxesResult
        AxesCamera
        
        BtnSelectDir
        BtnTrain
        BtnTestBatch
        BtnSelectOffline
        BtnRecognize
        BtnToggleCam
        BtnCapture
        
        LabelAcc
        LabelResult
        LabelStatus
        
        % 核心数据
        DatasetPath = ''
        ImgRows = 353
        ImgCols = 353
        NumComponents = 100 % 优化：提高保留的主成分个数上限，增强特征表达能力
        
        TrainDataMatrix % d x N 矩阵
        TrainLabels     % 1 x N 数组
        TrainPaths      % N x 1 路径数组
        
        TestDataMatrix  % d x M 矩阵
        TestLabels      % 1 x M 数组
        TestPaths       % M x 1 路径数组
        
        % PCA 模型数据
        MeanFace
        EigenFaces
        TrainFeatures   % K x N 特征向量矩阵
        
        % 摄像头对象
        CamObj = []
        CamTimer = []
        IsCamRunning = false
        
        % 当前选中的待测图像
        CurrentTestImage = []
    end
    
    methods
        function obj = FaceApp()
            % 构造函数，初始化 GUI
            obj.createUI();
            obj.logMsg('系统初始化完成，请选择人脸数据库文件夹。');
        end
        
        function createUI(obj)
            % 创建主窗口
            obj.UIFigure = uifigure('Name', '基于PCA与SVD的人脸识别系统', 'Position', [100, 100, 1200, 800]);
            
            % 数据集与训练操作区
            obj.PanelData = uipanel(obj.UIFigure, 'Title', '1. 数据集与训练操作区', 'Position', [20, 600, 300, 180]);
            obj.BtnSelectDir = uibutton(obj.PanelData, 'Position', [20, 120, 260, 30], 'Text', '导入人脸数据集文件夹', 'ButtonPushedFcn', @(~, ~) obj.importDataset());
            obj.BtnTrain = uibutton(obj.PanelData, 'Position', [20, 80, 260, 30], 'Text', '启动模型训练 (PCA+SVD)', 'ButtonPushedFcn', @(~, ~) obj.startTraining(), 'Enable', 'off');
            obj.BtnTestBatch = uibutton(obj.PanelData, 'Position', [20, 40, 260, 30], 'Text', '批量测试并计算准确率', 'ButtonPushedFcn', @(~, ~) obj.batchTest(), 'Enable', 'off');
            obj.LabelAcc = uilabel(obj.PanelData, 'Position', [20, 10, 260, 22], 'Text', '模型准确率: 等待训练...', 'FontWeight', 'bold');
            
            % 图像处理中间结果可视化区
            obj.PanelVisual = uipanel(obj.UIFigure, 'Title', '2. 中间结果可视化区', 'Position', [340, 480, 840, 300]);
            obj.AxesOriginal = uiaxes(obj.PanelVisual, 'Position', [10, 30, 190, 240]); title(obj.AxesOriginal, '原始图像');
            obj.AxesPreprocess = uiaxes(obj.PanelVisual, 'Position', [210, 30, 190, 240]); title(obj.AxesPreprocess, '预处理图像');
            obj.AxesMeanFace = uiaxes(obj.PanelVisual, 'Position', [410, 30, 190, 240]); title(obj.AxesMeanFace, '平均脸');
            obj.AxesEigenFace = uiaxes(obj.PanelVisual, 'Position', [610, 30, 190, 240]); title(obj.AxesEigenFace, '特征脸示例');
            
            % 离线人脸点选识别区
            obj.PanelOffline = uipanel(obj.UIFigure, 'Title', '3. 离线样本识别区', 'Position', [20, 330, 300, 250]);
            obj.BtnSelectOffline = uibutton(obj.PanelOffline, 'Position', [20, 190, 260, 30], 'Text', '选择待测人脸图片', 'ButtonPushedFcn', @(~, ~) obj.selectOfflineImage());
            obj.BtnRecognize = uibutton(obj.PanelOffline, 'Position', [20, 150, 260, 30], 'Text', '执行单样本识别', 'ButtonPushedFcn', @(~, ~) obj.recognizeSingle(), 'Enable', 'off');
            obj.AxesResult = uiaxes(obj.PanelOffline, 'Position', [20, 40, 120, 100]); title(obj.AxesResult, '匹配结果');
            obj.LabelResult = uilabel(obj.PanelOffline, 'Position', [150, 40, 130, 100], 'Text', '识别结果待定', 'WordWrap', 'on');
            
            % 摄像头实时人脸识别区
            obj.PanelCamera = uipanel(obj.UIFigure, 'Title', '4. 摄像头实时识别区', 'Position', [340, 20, 400, 440]);
            obj.BtnToggleCam = uibutton(obj.PanelCamera, 'Position', [20, 380, 170, 30], 'Text', '启动/关闭摄像头', 'ButtonPushedFcn', @(~, ~) obj.toggleCamera());
            obj.BtnCapture = uibutton(obj.PanelCamera, 'Position', [210, 380, 170, 30], 'Text', '采集当前帧并识别', 'ButtonPushedFcn', @(~, ~) obj.captureAndRecognize(), 'Enable', 'off');
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
            % 滚动到底部
            scroll(obj.LogTextArea, 'bottom');
            drawnow;
        end
        
        function importDataset(obj)
            % 选择数据集文件夹并自动划分训练集/测试集
            folderPath = uigetdir(pwd, '选择人脸数据库文件夹 (按人员分文件夹或统一存放命名有规律均可)');
            if folderPath == 0
                return;
            end
            obj.DatasetPath = folderPath;
            obj.logMsg(['已选择数据集路径: ', folderPath]);
            
            % 解析文件夹结构 (这里假定每个同学有自己的文件夹，或者文件名带编号)
            % 根据项目要求，数据集为本班同学照片，每人10张。
            % 简化处理：递归读取所有jpg/png，尝试从文件名或文件夹名提取标签
            files = dir(fullfile(folderPath, '**', '*.jpg'));
            pngs = dir(fullfile(folderPath, '**', '*.png'));
            files = [files; pngs];
            
            if isempty(files)
                obj.logMsg('未在选定文件夹中找到图片！');
                return;
            end
            
            obj.logMsg(sprintf('共发现 %d 张图片，正在解析标签与预处理...', length(files)));
            
            % 简单划分策略：每人前 7 张作为训练集，后 3 张作为测试集
            % 假设通过上级文件夹名称作为人员标签 (或文件名的前缀)
            labels = cell(length(files), 1);
            paths = cell(length(files), 1);
            
            for i = 1:length(files)
                % 取上级文件夹名作为标签
                [folder, name, ext] = fileparts(fullfile(files(i).folder, files(i).name));
                [~, parentFolder] = fileparts(folder);
                labels{i} = parentFolder; % 使用文件夹名作为类别
                paths{i} = fullfile(files(i).folder, files(i).name);
            end
            
            % 按类别划分训练集与测试集
            uniqueLabels = unique(labels);
            numClasses = length(uniqueLabels);
            
            trainIdx = [];
            testIdx = [];
            
            for i = 1:numClasses
                clsIdx = find(strcmp(labels, uniqueLabels{i}));
                numSamples = length(clsIdx);
                % 取 70% 训练，30% 测试
                splitPoint = max(1, round(numSamples * 0.7));
                trainIdx = [trainIdx; clsIdx(1:splitPoint)];
                testIdx = [testIdx; clsIdx(splitPoint+1:end)];
            end
            
            obj.logMsg(sprintf('共提取 %d 个类别。划分完成: %d 训练样本, %d 测试样本。', numClasses, length(trainIdx), length(testIdx)));
            
            % 加载并向量化训练数据
            obj.logMsg('正在加载并预处理训练数据，请稍候...');
            d = obj.ImgRows * obj.ImgCols;
            N = length(trainIdx);
            obj.TrainDataMatrix = zeros(d, N);
            obj.TrainLabels = cell(1, N);
            obj.TrainPaths = cell(N, 1);
            
            for i = 1:N
                idx = trainIdx(i);
                imgPath = paths{idx};
                img = imread(imgPath);
                
                % 预处理: 灰度、去噪、归一化、向量化
                processedImg = ImagePreprocess.fullProcess(img, obj.ImgRows, obj.ImgCols);
                vec = ImagePreprocess.vectorizeImg(processedImg);
                
                obj.TrainDataMatrix(:, i) = vec;
                obj.TrainLabels{i} = labels{idx};
                obj.TrainPaths{i} = imgPath;
                
                if mod(i, 20) == 0
                    obj.logMsg(sprintf('预处理训练集进度: %d/%d', i, N));
                end
            end
            
            % 加载并向量化测试数据
            obj.logMsg('正在加载并预处理测试数据，请稍候...');
            M = length(testIdx);
            obj.TestDataMatrix = zeros(d, M);
            obj.TestLabels = cell(1, M);
            obj.TestPaths = cell(M, 1);
            
            for i = 1:M
                idx = testIdx(i);
                imgPath = paths{idx};
                img = imread(imgPath);
                
                processedImg = ImagePreprocess.fullProcess(img, obj.ImgRows, obj.ImgCols);
                vec = ImagePreprocess.vectorizeImg(processedImg);
                
                obj.TestDataMatrix(:, i) = vec;
                obj.TestLabels{i} = labels{idx};
                obj.TestPaths{i} = imgPath;
            end
            
            obj.logMsg('数据集加载与预处理完成！可启动模型训练。');
            obj.BtnTrain.Enable = 'on';
            obj.BtnSelectOffline.Enable = 'on';
        end
        
        function startTraining(obj)
            % 启动 PCA+SVD 模型训练
            obj.logMsg('==============================');
            obj.logMsg('启动 PCA+SVD 模型训练 (自主核心算法)...');
            tic;
            
            % 计算平均脸、协方差、特征值与特征向量、特征脸
            [obj.MeanFace, obj.EigenFaces, ~, ~] = PCA_SVD_Core.computePCA_SVD(obj.TrainDataMatrix, obj.NumComponents);
            
            % 生成训练样本的投影特征
            obj.TrainFeatures = PCA_SVD_Core.project(obj.TrainDataMatrix, obj.MeanFace, obj.EigenFaces);
            
            elapsedTime = toc;
            obj.logMsg(sprintf('训练完成！耗时: %.2f 秒', elapsedTime));
            
            % 可视化中间结果
            obj.visualizeIntermediate();
            
            obj.BtnTestBatch.Enable = 'on';
            obj.BtnRecognize.Enable = 'on';
        end
        
        function visualizeIntermediate(obj)
            % 可视化：原始图、预处理图、平均脸、特征脸
            if isempty(obj.TrainPaths)
                return;
            end
            % 取训练集第一张图作为展示
            samplePath = obj.TrainPaths{1};
            img = imread(samplePath);
            imshow(img, 'Parent', obj.AxesOriginal);
            
            processedImg = ImagePreprocess.fullProcess(img, obj.ImgRows, obj.ImgCols);
            imshow(processedImg, [], 'Parent', obj.AxesPreprocess);
            
            % 显示平均脸
            meanFaceImg = reshape(obj.MeanFace, [obj.ImgRows, obj.ImgCols]);
            imshow(meanFaceImg, [], 'Parent', obj.AxesMeanFace);
            
            % 显示第一主成分特征脸
            if size(obj.EigenFaces, 2) >= 1
                ef1 = obj.EigenFaces(:, 1);
                % 归一化到 0-1 显示
                ef1 = ImagePreprocess.normalizePixels(ef1);
                ef1Img = reshape(ef1, [obj.ImgRows, obj.ImgCols]);
                imshow(ef1Img, [], 'Parent', obj.AxesEigenFace);
            end
            
            obj.logMsg('中间结果已在界面可视化展示。');
        end
        
        function batchTest(obj)
            % 批量测试离线样本，计算准确率
            obj.logMsg('==============================');
            obj.logMsg('开始批量识别测试集并计算准确率...');
            tic;
            
            M = size(obj.TestDataMatrix, 2);
            if M == 0
                obj.logMsg('测试集为空，无法计算准确率。');
                return;
            end
            
            % 投影测试集
            testFeatures = PCA_SVD_Core.project(obj.TestDataMatrix, obj.MeanFace, obj.EigenFaces);
            
            % 批量分类匹配
            predictedIndices = ClassifierCore.classifyBatch(testFeatures, obj.TrainFeatures);
            
            % 获取预测标签
            predictedLabels = cell(1, M);
            for i = 1:M
                predictedLabels{i} = obj.TrainLabels{predictedIndices(i)};
            end
            
            % 计算准确率 (为兼容自主编写的 calcAccuracy，传入数值或字符串数组，这里转为字符串)
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
            % 离线识别：选择单张图片
            [fileName, pathName] = uigetfile({'*.jpg;*.png', '图像文件 (*.jpg, *.png)'}, '选择待测人脸图片');
            if fileName == 0
                return;
            end
            
            imgPath = fullfile(pathName, fileName);
            obj.CurrentTestImage = imread(imgPath);
            imshow(obj.CurrentTestImage, 'Parent', obj.AxesOriginal);
            obj.logMsg(['已导入待测图像: ', fileName]);
            
            if ~isempty(obj.MeanFace)
                obj.BtnRecognize.Enable = 'on';
            end
        end
        
        function recognizeSingle(obj)
            % 执行离线单样本识别
            if isempty(obj.CurrentTestImage) || isempty(obj.MeanFace)
                obj.logMsg('请先完成模型训练并导入待测图像。');
                return;
            end
            
            obj.logMsg('正在进行单样本识别...');
            tic;
            
            % 预处理与向量化
            processedImg = ImagePreprocess.fullProcess(obj.CurrentTestImage, obj.ImgRows, obj.ImgCols);
            imshow(processedImg, [], 'Parent', obj.AxesPreprocess);
            vec = ImagePreprocess.vectorizeImg(processedImg);
            
            % 投影提取特征
            testFeature = PCA_SVD_Core.project(vec, obj.MeanFace, obj.EigenFaces);
            
            % 欧式距离分类匹配
            [bestMatchIndex, minDistance] = ClassifierCore.classify(testFeature, obj.TrainFeatures);
            
            elapsedTime = toc;
            
            % 显示结果
            matchLabel = obj.TrainLabels{bestMatchIndex};
            matchPath = obj.TrainPaths{bestMatchIndex};
            
            obj.logMsg(sprintf('识别完成！耗时: %.2f 秒', elapsedTime));
            obj.logMsg(sprintf('匹配人员身份: %s', matchLabel));
            obj.logMsg(sprintf('最小欧式距离: %.4f', minDistance));
            
            % 在界面上显示结果与匹配图
            matchImg = imread(matchPath);
            imshow(matchImg, 'Parent', obj.AxesResult);
            [~, matchName, matchExt] = fileparts(matchPath);
            
            resultText = sprintf('匹配身份:\n%s\n\n对应数据库照片:\n%s', matchLabel, [matchName, matchExt]);
            obj.LabelResult.Text = resultText;
            obj.LabelResult.FontColor = 'blue';
            obj.LabelResult.FontWeight = 'bold';
        end
        
        function toggleCamera(obj)
            % 启动/关闭摄像头
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
                obj.BtnCapture.Enable = 'off';
                obj.logMsg('已关闭摄像头。');
            else
                % 启动
                try
                    % 自主调用本地摄像头 (依赖 Image Acquisition Toolbox 
                    % 或 webcam 支持包，项目未强制禁用底层摄像头采集命令)
                    obj.CamObj = webcam(); 
                    obj.IsCamRunning = true;
                    obj.LabelStatus.Text = '摄像头状态: 运行中';
                    obj.BtnCapture.Enable = 'on';
                    obj.logMsg('摄像头启动成功，正在实时预览...');
                    
                    % 开启定时器刷新画面
                    obj.CamTimer = timer('ExecutionMode', 'fixedRate', 'Period', 0.1, ...
                        'TimerFcn', @(~, ~) obj.updateCameraPreview());
                    start(obj.CamTimer);
                catch e
                    obj.logMsg(['无法启动摄像头: ', e.message]);
                end
            end
        end
        
        function updateCameraPreview(obj)
            % 定时更新摄像头画面
            if obj.IsCamRunning && ~isempty(obj.CamObj)
                try
                    img = snapshot(obj.CamObj);
                    imshow(img, 'Parent', obj.AxesCamera);
                catch
                    % 忽略单帧错误
                end
            end
        end
        
        function captureAndRecognize(obj)
            % 采集摄像头当前帧并进行识别
            if ~obj.IsCamRunning || isempty(obj.CamObj)
                obj.logMsg('摄像头未运行，无法采集。');
                return;
            end
            
            if isempty(obj.MeanFace)
                obj.logMsg('模型尚未训练，无法进行识别！');
                return;
            end
            
            obj.logMsg('已采集当前帧，正在识别...');
            % 捕获图像
            img = snapshot(obj.CamObj);
            
            % 直接全图预处理识别 (如果需要框选人脸，可通过自主计算中心区域简单截取)
            % 为保证只有头部，可以取图像中心正方形区域
            [h, w, ~] = size(img);
            side = min(h, w);
            cropImg = img(round((h-side)/2)+1 : round((h+side)/2), round((w-side)/2)+1 : round((w+side)/2), :);
            
            % 在界面显示截取的当前人脸图像
            imshow(cropImg, 'Parent', obj.AxesOriginal);
            
            % 预处理与特征提取
            processedImg = ImagePreprocess.fullProcess(cropImg, obj.ImgRows, obj.ImgCols);
            vec = ImagePreprocess.vectorizeImg(processedImg);
            testFeature = PCA_SVD_Core.project(vec, obj.MeanFace, obj.EigenFaces);
            
            % 分类匹配
            [bestMatchIndex, minDistance] = ClassifierCore.classify(testFeature, obj.TrainFeatures);
            
            matchLabel = obj.TrainLabels{bestMatchIndex};
            
            % 在画面上叠加文字结果
            obj.LabelStatus.Text = sprintf('实时识别结果: %s (距离: %.2f)', matchLabel, minDistance);
            obj.LabelStatus.FontColor = 'red';
            obj.logMsg(sprintf('实时采集识别结果: 身份 %s', matchLabel));
        end
    end
end