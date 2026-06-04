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

        % Python 路径
        PythonExe = 'C:\Python311\python.exe'
        EmbeddingScript

        % 数据
        DatasetPath = ''
        TrainMat
        TestMat

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

        CurrentTestImage = []
        LogCount = 0
        MaxLogLines = 30
    end

    methods
        function obj = FaceApp()
            obj.EmbeddingScript = fullfile(fileparts(mfilename('fullpath')), 'single_embedding.py');
            obj.createUI();
            obj.logMsg('系统就绪。请先点击「加载模型」。');
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

            % 按钮组
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

            % 状态信息
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

            % ── 右上：主显示区（原始图 + 匹配结果） ──
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

            % ── 右下：日志区（窄条） ──
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

            obj.TrainMat = load(trainMatPath);
            Xtrain = obj.TrainMat.embeddings'; % 512 x N
            obj.DBLabels = obj.TrainMat.labels;
            obj.DBFilenames = obj.TrainMat.filenames;

            [d, N] = size(Xtrain);
            obj.logMsg(sprintf('训练集: %d 样本, 维度 %d', N, d));

            % PCA + SVD 训练
            obj.logMsg(sprintf('PCA 训练中 (主成分数=%d)...', obj.NumComponents));
            drawnow;
            [obj.MeanFace, obj.EigenFaces, ~, ~] = PCA_SVD_Core.computePCA_SVD(Xtrain, obj.NumComponents);
            obj.NumComponents = size(obj.EigenFaces, 2);
            obj.logMsg(sprintf('PCA 完成: %d 个主成分', obj.NumComponents));

            % 投影 + 标准化
            obj.DBFeatures = PCA_SVD_Core.project(Xtrain, obj.MeanFace, obj.EigenFaces);
            obj.FeatureScale = std(obj.DBFeatures, 0, 2);
            obj.FeatureScale(obj.FeatureScale < 1e-6) = 1;
            obj.DBFeatures = obj.DBFeatures ./ obj.FeatureScale;
            obj.DBFeatures = obj.l2normalize(obj.DBFeatures);

            % 加载测试集
            testMatPath = fullfile(rootDir, 'test_data_embeddings.mat');
            if isfile(testMatPath)
                obj.TestMat = load(testMatPath);
                Xtest = obj.TestMat.embeddings';
                obj.TestLabels = obj.TestMat.labels;
                obj.TestFilenames = obj.TestMat.filenames;
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

            % 估计 Unknown 阈值
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

        % ────────── 单图识别 ──────────
        function recognizeSingle(obj)
            if isempty(obj.CurrentTestImage) || isempty(obj.DBFeatures)
                obj.logMsg('请先加载模型并选择图片');
                return;
            end

            obj.logMsg('正在提取 embedding...');
            drawnow;
            tic;

            % 调 Python 获取 embedding
            tmpImg = fullfile(tempdir, 'face_rec_tmp.jpg');
            imwrite(obj.CurrentTestImage, tmpImg);

            cmd = sprintf('"%s" "%s" "%s"', obj.PythonExe, obj.EmbeddingScript, tmpImg);
            [status, result] = system(cmd);

            if status ~= 0
                obj.logMsg(sprintf('Python 调用失败: %s', result));
                return;
            end

            emb = str2num(strtrim(result)); %#ok<ST2NM>
            if isempty(emb) || length(emb) ~= 512
                obj.logMsg('Embedding 维度错误');
                return;
            end

            vec = emb(:); % 512 x 1

            % PCA 投影
            testFeat = PCA_SVD_Core.project(vec, obj.MeanFace, obj.EigenFaces);
            testFeat = testFeat ./ obj.FeatureScale;
            testFeat = obj.l2normalize(testFeat);

            % KNN 分类
            [bestLabel, bestIdx, simScore] = ClassifierCore.classifyKNNByLabel(...
                testFeat, obj.DBFeatures, obj.DBLabels, obj.KnnK);

            elapsed = toc;
            simScore = max(0, simScore);
            matchLabel = char(bestLabel);

            if simScore < obj.SimThreshold
                displayLabel = 'Unknown';
                obj.LabelResult.FontColor = [0.6 0.6 0.6];
            else
                displayLabel = matchLabel;
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

            if isfile(tmpImg), delete(tmpImg); end
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
                if ~isempty(obj.CamTimer), stop(obj.CamTimer); delete(obj.CamTimer); obj.CamTimer = []; end
                if ~isempty(obj.CamObj), tmp = obj.CamObj; obj.CamObj = []; clear tmp; end
                cla(obj.AxesCamera);
                title(obj.AxesCamera, '点击「开启摄像头」启动');
                obj.LabelCamStatus.Text = '摄像头: 关闭';
                obj.LabelCamStatus.FontColor = [0.4 0.4 0.4];
                obj.IsCamRunning = false;
                obj.IsRealTimeEnabled = false;
                obj.BtnToggleRealTime.Enable = 'off';
                obj.BtnToggleCam.Text = '5. 开启摄像头';
                obj.logMsg('摄像头已关闭');
            else
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

        function toggleRealTime(obj)
            if obj.IsRealTimeEnabled
                obj.IsRealTimeEnabled = false;
                obj.BtnToggleRealTime.Text = '开启实时识别';
                obj.logMsg('实时识别已关闭');
            else
                obj.IsRealTimeEnabled = true;
                obj.BtnToggleRealTime.Text = '关闭实时识别';
                obj.HistoryLabels = {};
                obj.logMsg('实时识别已开启');
            end
        end

        function camTick(obj)
            if ~obj.IsCamRunning || isempty(obj.CamObj), return; end
            try
                img = snapshot(obj.CamObj);
                imshow(img, 'Parent', obj.AxesCamera);

                if obj.IsRealTimeEnabled && ~isempty(obj.DBFeatures)
                    obj.processFrame(img);
                end
            catch
            end
        end

        function processFrame(obj, img)
            tmpImg = fullfile(tempdir, 'face_cam_tmp.jpg');
            imwrite(img, tmpImg);
            cmd = sprintf('"%s" "%s" "%s"', obj.PythonExe, obj.EmbeddingScript, tmpImg);
            [status, result] = system(cmd);
            if isfile(tmpImg), delete(tmpImg); end
            if status ~= 0, return; end

            emb = str2num(strtrim(result)); %#ok<ST2NM>
            if isempty(emb) || length(emb) ~= 512, return; end

            vec = emb(:);
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

            obj.HistoryLabels{end+1} = curResult;
            if length(obj.HistoryLabels) > obj.SmoothFrames
                obj.HistoryLabels(1) = [];
            end

            if length(obj.HistoryLabels) >= obj.SmoothFrames
                [uniqueL, ~, ic] = unique(obj.HistoryLabels);
                counts = accumarray(ic, 1);
                [maxCount, idx] = max(counts);
                if maxCount >= ceil(obj.SmoothFrames * 0.8)
                    obj.LabelCamStatus.Text = sprintf('识别: %s (%.2f)', uniqueL{idx}, simScore);
                    obj.LabelCamStatus.FontColor = [0.8 0.1 0.1];
                else
                    obj.LabelCamStatus.Text = '识别中...';
                    obj.LabelCamStatus.FontColor = [0.1 0.1 0.8];
                end
            else
                obj.LabelCamStatus.Text = '采集中...';
                obj.LabelCamStatus.FontColor = [0.1 0.1 0.8];
            end
        end
    end

    methods (Access = private)
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
