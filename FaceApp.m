classdef FaceApp < handle
    % FaceApp 人脸识别 GUI — 深度 embedding + PCA/SVD + KNN

    properties
        UIFigure
        PanelControl
        PanelDisplay
        PanelCamera
        PanelLog

        BtnLoadModel
        BtnSelectImage
        BtnRecognize
        BtnBatchTest
        BtnToggleCam
        BtnToggleRealTime

        LabelModelStatus
        LabelAccuracy
        LabelResult
        LabelCamStatus

        AxesOriginal
        AxesMatch
        AxesCamera
        LogTextArea

        % Python
        PythonExe = 'C:\Python311\python.exe'
        SingleEmbScript   % single_embedding.py
        ServerScript      % embedding_server.py
        ProjectDir        % 项目根目录

        % 持久 Python 进程（摄像头用）
        ServerProc = []   % java.lang.Process

        % 数据
        DatasetPath = ''

        % PCA 模型
        MeanFace
        EigenFaces
        DBFeatures   % K x N
        DBLabels     % 1 x N cell
        DBFilenames  % N x 1 cell
        FeatureScale % K x 1
        NumComponents = 120

        TestFeatures % K x M
        TestLabels
        TestFilenames

        KnnK = 5
        SimThreshold = 0.60

        % 摄像头
        CamObj = []
        CamTimer = []
        IsCamRunning = false
        IsRealTimeEnabled = false
        HistoryLabels = {}
        SmoothFrames = 5

        % 实时识别独立窗口
        RealTimeFig = []
        RealTimeAxes = []
        RealTimeLabel = []
        RealTimeInfo = []
        LastServerRestart = 0

        CurrentTestImage = []
        MaxLogLines = 30
    end

    methods
        function obj = FaceApp()
            obj.ProjectDir = fileparts(mfilename('fullpath'));
            obj.SingleEmbScript = fullfile(obj.ProjectDir, 'single_embedding.py');
            obj.ServerScript = fullfile(obj.ProjectDir, 'embedding_server.py');
            obj.createUI();
            obj.logMsg('系统就绪。请点击「1. 加载模型」。');
        end

        function createUI(obj)
            obj.UIFigure = uifigure('Name', '基于PCA+SVD的人脸识别系统', ...
                'Position', [80, 80, 1280, 720], 'Color', [0.94 0.94 0.94]);

            % ── 左侧控制面板 ──
            obj.PanelControl = uipanel(obj.UIFigure, ...
                'Title', '', 'BorderType', 'none', ...
                'Position', [0, 0, 260, 720], 'BackgroundColor', [0.92 0.92 0.92]);

            uilabel(obj.PanelControl, 'Text', '人脸识别系统', ...
                'Position', [0, 670, 260, 40], ...
                'FontSize', 16, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center');

            uilabel(obj.PanelControl, 'Text', '—— PCA + SVD ——', ...
                'Position', [0, 648, 260, 22], ...
                'FontSize', 11, 'FontColor', [0.5 0.5 0.5], ...
                'HorizontalAlignment', 'center');

            y = 580;
            obj.BtnLoadModel = uibutton(obj.PanelControl, 'push', ...
                'Text', '1. 加载模型', 'Position', [20, y, 220, 38], ...
                'FontSize', 13, 'BackgroundColor', [0.3 0.6 0.9], ...
                'FontColor', 'white', ...
                'ButtonPushedFcn', @(~,~) obj.loadModel());

            y = y - 50;
            obj.BtnSelectImage = uibutton(obj.PanelControl, 'push', ...
                'Text', '2. 选择图片', 'Position', [20, y, 220, 38], ...
                'FontSize', 13, 'Enable', 'off', ...
                'ButtonPushedFcn', @(~,~) obj.selectImage());

            y = y - 50;
            obj.BtnRecognize = uibutton(obj.PanelControl, 'push', ...
                'Text', '3. 开始识别', 'Position', [20, y, 220, 38], ...
                'FontSize', 13, 'Enable', 'off', ...
                'ButtonPushedFcn', @(~,~) obj.recognizeSingle());

            y = y - 50;
            obj.BtnBatchTest = uibutton(obj.PanelControl, 'push', ...
                'Text', '4. 批量测试', 'Position', [20, y, 220, 38], ...
                'FontSize', 13, 'Enable', 'off', ...
                'ButtonPushedFcn', @(~,~) obj.batchTest());

            y = y - 50;
            obj.BtnToggleCam = uibutton(obj.PanelControl, 'push', ...
                'Text', '5. 开启摄像头', 'Position', [20, y, 220, 38], ...
                'FontSize', 13, ...
                'ButtonPushedFcn', @(~,~) obj.toggleCamera());

            y = y - 50;
            obj.BtnToggleRealTime = uibutton(obj.PanelControl, 'push', ...
                'Text', '开启实时识别', 'Position', [20, y, 220, 38], ...
                'FontSize', 13, 'Enable', 'off', ...
                'ButtonPushedFcn', @(~,~) obj.toggleRealTime());

            % 分隔线
            y = y - 30;
            uilabel(obj.PanelControl, 'Text', repmat('_', 1, 30), ...
                'Position', [20, y, 220, 16], 'FontColor', [0.75 0.75 0.75]);

            y = y - 30;
            obj.LabelAccuracy = uilabel(obj.PanelControl, ...
                'Text', '准确率: --', ...
                'Position', [20, y, 220, 24], ...
                'FontSize', 14, 'FontWeight', 'bold', 'FontColor', [0.1 0.1 0.1]);

            y = y - 28;
            obj.LabelModelStatus = uilabel(obj.PanelControl, ...
                'Text', '模型: 未加载', ...
                'Position', [20, y, 220, 22], ...
                'FontSize', 11, 'FontColor', [0.5 0.5 0.5]);

            % ── 右上：主显示区 ──
            obj.PanelDisplay = uipanel(obj.UIFigure, ...
                'Title', '识别结果', 'FontSize', 12, ...
                'Position', [270, 300, 1000, 400]);

            obj.AxesOriginal = uiaxes(obj.PanelDisplay, ...
                'Position', [30, 30, 420, 330]);
            title(obj.AxesOriginal, '原始图像', 'FontSize', 13);
            axis(obj.AxesOriginal, 'off');

            obj.AxesMatch = uiaxes(obj.PanelDisplay, ...
                'Position', [510, 30, 420, 330]);
            title(obj.AxesMatch, '匹配结果', 'FontSize', 13);
            axis(obj.AxesMatch, 'off');

            obj.LabelResult = uilabel(obj.PanelDisplay, ...
                'Text', '选择图片后点击「开始识别」', ...
                'Position', [460, 0, 520, 30], ...
                'FontSize', 15, 'FontWeight', 'bold', ...
                'FontColor', [0.2 0.2 0.2], ...
                'HorizontalAlignment', 'center');

            % ── 右中：摄像头预览区 ──
            obj.PanelCamera = uipanel(obj.UIFigure, ...
                'Title', '摄像头预览', 'FontSize', 12, ...
                'Position', [270, 40, 1000, 250]);

            obj.AxesCamera = uiaxes(obj.PanelCamera, ...
                'Position', [250, 10, 500, 220]);
            title(obj.AxesCamera, '点击「开启摄像头」启动', 'FontSize', 12);
            axis(obj.AxesCamera, 'off');

            obj.LabelCamStatus = uilabel(obj.PanelCamera, ...
                'Text', '摄像头: 关闭', ...
                'Position', [10, 10, 230, 22], ...
                'FontSize', 11, 'FontWeight', 'bold', ...
                'FontColor', [0.4 0.4 0.4]);

            % ── 右下：日志区 ──
            obj.PanelLog = uipanel(obj.UIFigure, ...
                'Title', '日志', 'FontSize', 11, ...
                'Position', [270, 0, 1000, 35]);

            obj.LogTextArea = uitextarea(obj.PanelLog, ...
                'Position', [5, 2, 990, 28], ...
                'Editable', 'off', 'FontSize', 10, ...
                'BackgroundColor', [0.97 0.97 0.97]);
        end

        % ────────── 日志 ──────────
        function logMsg(obj, msg)
            ts = datestr(now, 'HH:MM:SS');
            line = sprintf('[%s] %s', ts, msg);
            val = obj.LogTextArea.Value;
            if ischar(val), val = {val}; end
            if isempty(val) || (iscell(val) && isempty(val{1}))
                val = {line};
            else
                val{end+1} = line; %#ok<AGROW>
            end
            if length(val) > obj.MaxLogLines
                val = val(end - obj.MaxLogLines + 1 : end);
            end
            obj.LogTextArea.Value = val;
            scroll(obj.LogTextArea, 'bottom');
            drawnow;
        end

        % ────────── 加载模型 ──────────
        function loadModel(obj)
            rootDir = uigetdir(pwd, '选择包含 train_data_embeddings.mat 的文件夹');
            if isequal(rootDir, 0), return; end

            trainMatPath = fullfile(rootDir, 'train_data_embeddings.mat');
            if ~isfile(trainMatPath)
                obj.logMsg('错误: 未找到 train_data_embeddings.mat，请先运行 extract_embeddings.py');
                return;
            end

            obj.DatasetPath = rootDir;
            obj.logMsg('正在加载训练集 embedding...');
            drawnow;

            data = load(trainMatPath);
            Xtrain = data.embeddings'; % 512 x N
            obj.DBLabels = data.labels;
            obj.DBFilenames = data.filenames;

            [d, N] = size(Xtrain);
            obj.logMsg(sprintf('训练集: %d 样本, 维度 %d', N, d));

            obj.logMsg(sprintf('PCA 训练中 (主成分数=%d)...', obj.NumComponents));
            drawnow;
            [obj.MeanFace, obj.EigenFaces, ~, ~] = PCA_SVD_Core.computePCA_SVD(Xtrain, obj.NumComponents);
            obj.NumComponents = size(obj.EigenFaces, 2);
            obj.logMsg(sprintf('PCA 完成: %d 个主成分', obj.NumComponents));

            obj.DBFeatures = PCA_SVD_Core.project(Xtrain, obj.MeanFace, obj.EigenFaces);
            obj.FeatureScale = std(obj.DBFeatures, 0, 2);
            obj.FeatureScale(obj.FeatureScale < 1e-6) = 1;
            obj.DBFeatures = obj.DBFeatures ./ obj.FeatureScale;
            obj.DBFeatures = obj.l2normalize(obj.DBFeatures);

            % 测试集
            testMatPath = fullfile(rootDir, 'test_data_embeddings.mat');
            if isfile(testMatPath)
                tdata = load(testMatPath);
                Xtest = tdata.embeddings';
                obj.TestLabels = tdata.labels;
                obj.TestFilenames = tdata.filenames;
                obj.TestFeatures = PCA_SVD_Core.project(Xtest, obj.MeanFace, obj.EigenFaces);
                obj.TestFeatures = obj.TestFeatures ./ obj.FeatureScale;
                obj.TestFeatures = obj.l2normalize(obj.TestFeatures);
                obj.logMsg(sprintf('测试集: %d 样本已加载', size(Xtest, 2)));
                obj.BtnBatchTest.Enable = 'on';
            else
                obj.TestFeatures = [];
                obj.TestLabels = {};
                obj.TestFilenames = {};
                obj.logMsg('未找到 test_data_embeddings.mat，跳过测试集');
            end

            obj.SimThreshold = obj.estimateThreshold();
            obj.logMsg(sprintf('阈值: %.3f', obj.SimThreshold));

            obj.LabelModelStatus.Text = sprintf('模型: 就绪 (%d主成分)', obj.NumComponents);
            obj.LabelModelStatus.FontColor = [0.1 0.6 0.1];
            obj.BtnSelectImage.Enable = 'on';
            obj.BtnRecognize.Enable = 'off';
            obj.logMsg('模型加载完成！可进行识别。');
        end

        % ────────── 选择图片 ──────────
        function selectImage(obj)
            [fname, fpath] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', '图像文件'}, '选择待识别图片');
            if isequal(fname, 0), return; end
            imgPath = fullfile(fpath, fname);
            obj.CurrentTestImage = imread(imgPath);
            imshow(obj.CurrentTestImage, 'Parent', obj.AxesOriginal);
            title(obj.AxesOriginal, '原始图像');
            cla(obj.AxesMatch);
            title(obj.AxesMatch, '匹配结果');
            obj.LabelResult.Text = '点击「开始识别」';
            obj.LabelResult.FontColor = [0.2 0.2 0.2];
            obj.BtnRecognize.Enable = 'on';
            obj.logMsg(sprintf('已选择: %s', fname));
        end

        % ────────── 单图识别（文件方式） ──────────
        function recognizeSingle(obj)
            if isempty(obj.CurrentTestImage) || isempty(obj.DBFeatures)
                obj.logMsg('请先加载模型并选择图片');
                return;
            end

            obj.logMsg('正在提取 embedding...');
            drawnow;
            tic;

            % 把选中的图片复制到项目目录下固定名称（避免中文路径）
            tmpSrc = fullfile(tempdir, 'face_rec_src.jpg');
            tmpEmb = fullfile(obj.ProjectDir, '_tmp_emb.txt');
            imwrite(obj.CurrentTestImage, tmpSrc);

            % 复制到项目目录（ASCII 路径）
            tmpImg = fullfile(obj.ProjectDir, '_tmp_rec.jpg');
            copyfile(tmpSrc, tmpImg);

            % 调 Python：图片路径 + 输出文件路径
            cmd = sprintf('"%s" "%s" "%s" "%s"', ...
                obj.PythonExe, obj.SingleEmbScript, tmpImg, tmpEmb);
            [~, ~] = system(cmd);

            % 从文件读取 embedding
            embVec = obj.readEmbeddingFile(tmpEmb);

            % 清理临时文件
            if isfile(tmpImg), delete(tmpImg); end
            if isfile(tmpEmb), delete(tmpEmb); end
            if isfile(tmpSrc), delete(tmpSrc); end

            if isempty(embVec)
                obj.logMsg('Embedding 提取失败');
                return;
            end

            % PCA 投影
            testFeat = PCA_SVD_Core.project(embVec, obj.MeanFace, obj.EigenFaces);
            testFeat = testFeat ./ obj.FeatureScale;
            testFeat = obj.l2normalize(testFeat);

            % KNN 分类
            [bestLabel, bestIdx, simScore] = ClassifierCore.classifyKNNByLabel(...
                testFeat, obj.DBFeatures, obj.DBLabels, obj.KnnK);

            elapsed = toc;
            simScore = max(0, simScore);

            if simScore < obj.SimThreshold
                displayLabel = 'Unknown';
                obj.LabelResult.FontColor = [0.6 0.6 0.6];
            else
                displayLabel = char(bestLabel);
                obj.LabelResult.FontColor = [0.1 0.1 0.8];
            end

            % 显示匹配图片
            matchFile = obj.DBFilenames{bestIdx};
            matchPath = fullfile(obj.DatasetPath, 'train_data', matchFile);
            if isfile(matchPath)
                matchImg = imread(matchPath);
                imshow(matchImg, 'Parent', obj.AxesMatch);
                title(obj.AxesMatch, '匹配结果');
            end

            obj.LabelResult.Text = sprintf('%s  (相似度: %.2f, %.2fs)', displayLabel, simScore, elapsed);
            obj.logMsg(sprintf('识别: %s | 相似度: %.2f | 耗时: %.2fs', displayLabel, simScore, elapsed));
        end

        % ────────── 批量测试 ──────────
        function batchTest(obj)
            if isempty(obj.TestFeatures)
                obj.logMsg('测试集为空');
                return;
            end

            obj.logMsg('批量测试中...');
            drawnow;
            tic;

            M = size(obj.TestFeatures, 2);
            pred = cell(1, M);
            for i = 1:M
                [bl, ~, ~] = ClassifierCore.classifyKNNByLabel(...
                    obj.TestFeatures(:, i), obj.DBFeatures, obj.DBLabels, obj.KnnK);
                pred{i} = char(bl);
            end

            [acc, ~] = ClassifierCore.calcAccuracy(string(obj.TestLabels), string(pred));
            elapsed = toc;

            obj.LabelAccuracy.Text = sprintf('准确率: %.2f%%', acc);
            obj.LabelAccuracy.FontColor = [0.8 0.1 0.1];
            obj.logMsg(sprintf('批量测试完成: %.2f%% (%d/%d), 耗时 %.2fs', acc, round(M*acc/100), M, elapsed));
        end

        % ────────── 摄像头 ──────────
        function toggleCamera(obj)
            if obj.IsCamRunning
                % 关闭摄像头
                obj.stopCamera();
            else
                % 打开摄像头
                if exist('webcam', 'file') ~= 2
                    obj.logMsg('未检测到 webcam 支持包');
                    return;
                end
                try
                    obj.CamObj = webcam(1);
                    obj.IsCamRunning = true;
                    obj.BtnToggleCam.Text = '关闭摄像头';
                    obj.LabelCamStatus.Text = '摄像头: 运行中';
                    obj.LabelCamStatus.FontColor = [0.1 0.6 0.1];
                    if ~isempty(obj.DBFeatures)
                        obj.BtnToggleRealTime.Enable = 'on';
                    end
                    obj.logMsg('摄像头已启动');
                    obj.CamTimer = timer('ExecutionMode', 'fixedRate', 'Period', 0.3, ...
                        'TimerFcn', @(~,~) obj.camTick());
                    start(obj.CamTimer);
                catch e
                    obj.logMsg(sprintf('摄像头启动失败: %s', e.message));
                end
            end
        end

        function stopCamera(obj)
            % 关闭实时识别 + 持久进程 + 独立窗口
            obj.IsRealTimeEnabled = false;
            obj.BtnToggleRealTime.Enable = 'off';
            obj.BtnToggleRealTime.Text = '开启实时识别';
            obj.stopServer();
            obj.closeRealTimeWindow();

            % 关闭摄像头
            if ~isempty(obj.CamTimer), stop(obj.CamTimer); delete(obj.CamTimer); obj.CamTimer = []; end
            if ~isempty(obj.CamObj), tmp = obj.CamObj; obj.CamObj = []; clear tmp; end
            cla(obj.AxesCamera);
            title(obj.AxesCamera, '点击「开启摄像头」启动');
            obj.LabelCamStatus.Text = '摄像头: 关闭';
            obj.LabelCamStatus.FontColor = [0.4 0.4 0.4];
            obj.IsCamRunning = false;
            obj.BtnToggleCam.Text = '5. 开启摄像头';
            obj.logMsg('摄像头已关闭');
        end

        function toggleRealTime(obj)
            if obj.IsRealTimeEnabled
                % 关闭实时识别
                obj.IsRealTimeEnabled = false;
                obj.BtnToggleRealTime.Text = '开启实时识别';
                obj.stopServer();
                obj.closeRealTimeWindow();
                obj.logMsg('实时识别已关闭');
            else
                % 启动持久 Python 进程
                obj.logMsg('正在启动 embedding server...');
                drawnow;
                if ~obj.startServer()
                    obj.logMsg('embedding server 启动失败');
                    return;
                end
                obj.IsRealTimeEnabled = true;
                obj.BtnToggleRealTime.Text = '关闭实时识别';
                obj.HistoryLabels = {};
                obj.openRealTimeWindow();
                obj.logMsg('实时识别已开启，摄像头已就绪');
            end
        end

        function camTick(obj)
            if ~obj.IsCamRunning || isempty(obj.CamObj), return; end
            try
                img = snapshot(obj.CamObj);

                if obj.IsRealTimeEnabled
                    obj.processFrame(img);
                else
                    % 非实时模式：画面显示在原 GUI 预览区
                    imshow(img, 'Parent', obj.AxesCamera);
                end
            catch
            end
        end

        function processFrame(obj, img)
            if isempty(obj.DBFeatures), return; end

            obj.showRealTimeFrame(img);

            % 保存帧到临时文件
            tmpImg = fullfile(obj.ProjectDir, '_tmp_cam.jpg');
            imwrite(img, tmpImg);

            % 向持久进程发送路径，读取回复
            reply = obj.serverQuery(tmpImg);
            if isfile(tmpImg), delete(tmpImg); end

            % 如果 server 无响应，尝试自动重启
            if isempty(reply)
                now_t = posixtime(datetime('now'));
                if (now_t - obj.LastServerRestart) > 10
                    obj.logMsg('Server 无响应，正在重启...');
                    obj.LastServerRestart = now_t;
                    obj.stopServer();
                    if obj.startServer()
                        obj.logMsg('Server 重启成功');
                    else
                        obj.logMsg('Server 重启失败');
                    end
                end
                return;
            end

            if startsWith(reply, 'error')
                obj.logMsg(sprintf('Server 错误: %s', reply));
                return;
            end

            % 解析 "ok,0.0012,0.0034,..."
            parts = split(reply, ',');
            if length(parts) ~= 513
                obj.logMsg(sprintf('Embedding 解析失败: 部分数=%d, 期望513', length(parts)));
                return;
            end

            vals = zeros(512, 1);
            for k = 1:512
                vals(k) = str2double(parts{k + 1});
            end
            if ~all(isfinite(vals))
                obj.logMsg('Embedding 包含 NaN/Inf');
                return;
            end

            vec = vals(:);
            testFeat = PCA_SVD_Core.project(vec, obj.MeanFace, obj.EigenFaces);
            testFeat = testFeat ./ obj.FeatureScale;
            testFeat = obj.l2normalize(testFeat);

            [bestLabel, ~, simScore] = ClassifierCore.classifyKNNByLabel(...
                testFeat, obj.DBFeatures, obj.DBLabels, obj.KnnK);
            simScore = max(0, simScore);
            bestLabel = char(bestLabel);

            if simScore < obj.SimThreshold
                curResult = 'Unknown';
            else
                curResult = bestLabel;
            end

            obj.updateRealTimeResult(curResult, simScore);

            % 平滑结果
            obj.HistoryLabels{end+1} = curResult;
            if length(obj.HistoryLabels) > obj.SmoothFrames
                obj.HistoryLabels(1) = [];
            end

            % 更新原 GUI 状态标签
            if length(obj.HistoryLabels) >= obj.SmoothFrames
                [uniqueL, ~, ic] = unique(obj.HistoryLabels);
                counts = accumarray(ic, 1);
                [maxCount, idx] = max(counts);
                if maxCount >= ceil(obj.SmoothFrames * 0.8)
                    stableResult = uniqueL{idx};
                    obj.LabelCamStatus.Text = sprintf('识别: %s (%.2f)', stableResult, simScore);
                    obj.LabelCamStatus.FontColor = [0.8 0.1 0.1];
                else
                    obj.LabelCamStatus.Text = '识别中...';
                    obj.LabelCamStatus.FontColor = [0.1 0.1 0.8];
                end
            else
                obj.LabelCamStatus.Text = '采集中...';
                obj.LabelCamStatus.FontColor = [0.1 0.1 0.8];
            end

            % 更新独立窗口的实时信息
            if ~isempty(obj.RealTimeFig) && isvalid(obj.RealTimeFig)
                obj.RealTimeInfo.Text = sprintf('相似度: %.4f | 阈值: %.4f | 当前帧: %s', ...
                    simScore, obj.SimThreshold, curResult);
            end
        end
    end

    methods (Access = private)
        % ────── 实时识别独立窗口 ──────
        function openRealTimeWindow(obj)
            obj.closeRealTimeWindow();

            obj.RealTimeFig = uifigure('Name', '实时人脸识别', ...
                'Position', [200, 100, 800, 650], 'Color', [0.15 0.15 0.15], ...
                'CloseRequestFcn', @(~,~) obj.onRealTimeWindowClosed());

            obj.RealTimeAxes = uiaxes(obj.RealTimeFig, ...
                'Position', [10, 70, 780, 540], 'BackgroundColor', [0.1 0.1 0.1]);
            title(obj.RealTimeAxes, '摄像头画面', 'FontSize', 14, 'Color', 'white');
            axis(obj.RealTimeAxes, 'off');

            obj.RealTimeLabel = uilabel(obj.RealTimeFig, ...
                'Text', '等待识别...', ...
                'Position', [10, 35, 780, 30], ...
                'FontSize', 18, 'FontWeight', 'bold', ...
                'FontColor', [0.2 1.0 0.4], ...
                'HorizontalAlignment', 'center');

            obj.RealTimeInfo = uilabel(obj.RealTimeFig, ...
                'Text', '就绪', ...
                'Position', [10, 8, 780, 22], ...
                'FontSize', 11, 'FontColor', [0.7 0.7 0.7], ...
                'HorizontalAlignment', 'center');
        end

        function showRealTimeFrame(obj, img)
            if isempty(obj.RealTimeFig) || ~isvalid(obj.RealTimeFig)
                return;
            end

            try
                imshow(img, 'Parent', obj.RealTimeAxes);
                drawnow limitrate;
            catch
            end
        end

        function closeRealTimeWindow(obj)
            if ~isempty(obj.RealTimeFig)
                try
                    if isvalid(obj.RealTimeFig)
                        delete(obj.RealTimeFig);
                    end
                catch
                end
                obj.RealTimeFig = [];
                obj.RealTimeAxes = [];
                obj.RealTimeLabel = [];
                obj.RealTimeInfo = [];
            end
        end

        function onRealTimeWindowClosed(obj)
            % 用户手动关闭独立窗口 → 关闭实时识别
            if obj.IsRealTimeEnabled
                obj.IsRealTimeEnabled = false;
                obj.BtnToggleRealTime.Text = '开启实时识别';
                obj.stopServer();
                obj.RealTimeFig = [];
                obj.RealTimeAxes = [];
                obj.RealTimeLabel = [];
                obj.RealTimeInfo = [];
                obj.logMsg('实时识别窗口已关闭');
            end
        end

        function updateRealTimeResult(obj, label, simScore)
            if isempty(obj.RealTimeFig) || ~isvalid(obj.RealTimeFig), return; end
            if strcmp(label, 'Unknown')
                obj.RealTimeLabel.Text = sprintf('? Unknown  (%.2f)', simScore);
                obj.RealTimeLabel.FontColor = [0.7 0.7 0.7];
            else
                obj.RealTimeLabel.Text = sprintf('%s  (%.2f)', label, simScore);
                obj.RealTimeLabel.FontColor = [0.2 1.0 0.4];
            end
        end

        % ────── 持久 Python 进程管理（文件交换协议） ──────
        function ok = startServer(obj)
            % 启动 embedding_server.py 子进程，通过文件交换中文路径
            ok = false;
            obj.stopServer();

            % 清理旧的交换文件
            for f = {'_server_req.txt','_server_rep.txt','_server_quit.lock','_server_ready.lock'}
                fp = fullfile(obj.ProjectDir, f{1});
                if isfile(fp), delete(fp); end
            end

            try
                cmdList = java.util.Arrays.asList({obj.PythonExe, obj.ServerScript});
                pb = java.lang.ProcessBuilder(cmdList);
                pb.directory(java.io.File(obj.ProjectDir));
                obj.ServerProc = pb.start();

                % 等待 _server_ready.lock 出现（最多 30 秒）
                readyFile = fullfile(obj.ProjectDir, '_server_ready.lock');
                t0 = now;
                while ~isfile(readyFile)
                    pause(0.1);
                    if (now - t0) * 86400 > 30, break; end
                end
                if isfile(readyFile)
                    ok = true;
                    delete(readyFile);
                end
            catch
                obj.ServerProc = [];
            end
        end

        function stopServer(obj)
            % 发送退出信号
            quitLock = fullfile(obj.ProjectDir, '_server_quit.lock');
            try
                fid = fopen(quitLock, 'w'); fclose(fid);
            catch
            end

            if ~isempty(obj.ServerProc)
                try
                    obj.ServerProc.waitFor();
                catch
                end
                try
                    obj.ServerProc.destroy();
                catch
                end
                obj.ServerProc = [];
            end

            % 清理交换文件
            for f = {'_server_req.txt','_server_rep.txt','_server_quit.lock','_server_ready.lock'}
                fp = fullfile(obj.ProjectDir, f{1});
                if isfile(fp), delete(fp); end
            end
        end

        function reply = serverQuery(obj, imgPath)
            % 通过文件交换方式向 Python 进程发送路径并获取 embedding
            reply = '';
            if isempty(obj.ServerProc), return; end

            reqFile = fullfile(obj.ProjectDir, '_server_req.txt');
            repFile = fullfile(obj.ProjectDir, '_server_rep.txt');

            try
                % 写入请求文件
                fid = fopen(reqFile, 'w', 'n', 'UTF-8');
                fwrite(fid, imgPath, 'char');
                fclose(fid);

                % 等待回复文件出现（最多 5 秒）
                t0 = now;
                while ~isfile(repFile)
                    pause(0.01);
                    if (now - t0) * 86400 > 5
                        obj.logMsg('Server 回复超时');
                        return;
                    end
                end

                pause(0.02);
                reply = strtrim(fileread(repFile));

                if isfile(repFile), delete(repFile); end
            catch
                reply = '';
            end
        end

        function vec = readEmbeddingFile(~, filePath)
            vec = [];
            if ~isfile(filePath), return; end

            try
                raw = strtrim(fileread(filePath));
                if isempty(raw) || startsWith(raw, 'ERROR')
                    return;
                end

                parts = strsplit(raw, ',');
                if numel(parts) ~= 512
                    return;
                end

                vals = zeros(512, 1);
                for k = 1:512
                    vals(k) = str2double(strtrim(parts{k}));
                end

                if all(isfinite(vals))
                    vec = vals;
                end
            catch
                vec = [];
            end
        end

        function F = l2normalize(~, F)
            for i = 1:size(F, 2)
                n = norm(F(:, i));
                if n > 0, F(:, i) = F(:, i) / n; end
            end
        end

        function thr = estimateThreshold(obj)
            N = size(obj.DBFeatures, 2);
            if N < 2, thr = 0.55; return; end
            sims = [];
            for i = 1:N
                lbl = string(obj.DBLabels{i});
                bestSim = -inf;
                for j = 1:N
                    if j == i || string(obj.DBLabels{j}) ~= lbl, continue; end
                    s = obj.DBFeatures(:, i)' * obj.DBFeatures(:, j);
                    if s > bestSim, bestSim = s; end
                end
                if bestSim > -inf, sims(end+1) = bestSim; end %#ok<AGROW>
            end
            if isempty(sims), thr = 0.55; return; end
            thr = mean(sims) - 2.2 * std(sims);
            thr = min(max(thr, 0.40), 0.85);
        end
    end
end
