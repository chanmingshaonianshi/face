classdef ImagePreprocess
    % ImagePreprocess 自主实现的图像预处理类
    % 包含灰度化、尺寸归一化、去噪、像素值归一化、向量化等标准化处理函数
    
    methods(Static)
        function grayImg = toGray(img)
            % 自主实现灰度化：将彩色图像转换为灰度图像
            % 公式：Gray = 0.2989*R + 0.5870*G + 0.1140*B
            [~, ~, channels] = size(img);
            if channels == 3
                imgD = double(img);
                grayImg = 0.2989 * imgD(:,:,1) + 0.5870 * imgD(:,:,2) + 0.1140 * imgD(:,:,3);
                grayImg = uint8(grayImg);
            else
                grayImg = img;
            end
        end
        
        function resizedImg = resizeImg(img, targetRows, targetCols)
            % 自主实现尺寸归一化（最近邻插值）
            [rows, cols, channels] = size(img);
            rowScale = rows / targetRows;
            colScale = cols / targetCols;
            
            if isa(img, 'uint8')
                resizedImg = zeros(targetRows, targetCols, channels, 'uint8');
            else
                resizedImg = zeros(targetRows, targetCols, channels, class(img));
            end
            
            % 预先计算坐标映射
            rMap = round((1:targetRows) * rowScale);
            rMap(rMap < 1) = 1; rMap(rMap > rows) = rows;
            
            cMap = round((1:targetCols) * colScale);
            cMap(cMap < 1) = 1; cMap(cMap > cols) = cols;
            
            for c = 1:channels
                resizedImg(:,:,c) = img(rMap, cMap, c);
            end
        end
        
        function normImg = normalizePixels(img)
            % 自主实现像素值归一化到 [0, 1] 范围
            imgD = double(img);
            minVal = min(imgD(:));
            maxVal = max(imgD(:));
            if maxVal > minVal
                normImg = (imgD - minVal) / (maxVal - minVal);
            else
                normImg = imgD - minVal;
            end
        end
        
        function vec = vectorizeImg(img)
            % 自主实现图像向量化：将二维矩阵转换为一维列向量
            vec = img(:);
        end
        
        function outImg = denoise(img)
            % 自主实现去噪（3x3简单均值滤波）
            [rows, cols] = size(img);
            outImg = double(img);
            imgD = double(img);
            
            % 边缘填充
            paddedImg = zeros(rows+2, cols+2);
            paddedImg(2:end-1, 2:end-1) = imgD;
            % 复制边缘
            paddedImg(1, 2:end-1) = imgD(1, :);
            paddedImg(end, 2:end-1) = imgD(end, :);
            paddedImg(2:end-1, 1) = imgD(:, 1);
            paddedImg(2:end-1, end) = imgD(:, end);
            paddedImg(1, 1) = imgD(1, 1);
            paddedImg(1, end) = imgD(1, end);
            paddedImg(end, 1) = imgD(end, 1);
            paddedImg(end, end) = imgD(end, end);
            
            for i = 1:rows
                for j = 1:cols
                    window = paddedImg(i:i+2, j:j+2);
                    outImg(i,j) = sum(window(:)) / 9;
                end
            end
            
            if isa(img, 'uint8')
                outImg = uint8(outImg);
            end
        end
        
        function eqImg = histEq(img)
            % 自主实现直方图均衡化，增强图像对比度，减弱光照影响
            [rows, cols] = size(img);
            imgD = double(img);
            
            % 计算直方图 (假设输入范围在 0-255)
            hist = zeros(256, 1);
            for i = 1:rows
                for j = 1:cols
                    val = round(imgD(i,j));
                    if val < 0; val = 0; end
                    if val > 255; val = 255; end
                    hist(val + 1) = hist(val + 1) + 1;
                end
            end
            
            % 计算累积分布函数 (CDF)
            cdf = zeros(256, 1);
            cdf(1) = hist(1);
            for i = 2:256
                cdf(i) = cdf(i-1) + hist(i);
            end
            
            % 构建映射表
            totalPixels = rows * cols;
            minCdf = min(cdf(cdf > 0)); % 找到非零的最小CDF值
            if isempty(minCdf)
                minCdf = 0;
            end
            
            map = round((cdf - minCdf) / (totalPixels - minCdf) * 255);
            map(map < 0) = 0;
            map(map > 255) = 255;
            
            % 映射回原图
            if isa(img, 'uint8')
                eqImg = zeros(rows, cols, 'uint8');
            else
                eqImg = zeros(rows, cols, class(img));
            end
            
            for i = 1:rows
                for j = 1:cols
                    val = round(imgD(i,j));
                    if val < 0; val = 0; end
                    if val > 255; val = 255; end
                    eqImg(i,j) = map(val + 1);
                end
            end
        end
        
        function croppedImg = cropCenter(img, cropRatio)
            % 自主实现中心裁剪：去除边缘的衣服和背景，尽量只保留人脸部分
            % 输入:
            %   img: 原始图像
            %   cropRatio: 裁剪比例 (如 0.6 表示保留中心 60% 的区域)
            if nargin < 2
                cropRatio = 0.6; % 默认保留中心 60%
            end
            
            [rows, cols, channels] = size(img);
            
            % 计算裁剪区域
            newRows = round(rows * cropRatio);
            newCols = round(cols * cropRatio);
            
            startRow = round((rows - newRows) / 2) + 1;
            startCol = round((cols - newCols) / 2) + 1;
            
            endRow = startRow + newRows - 1;
            endCol = startCol + newCols - 1;
            
            % 边界保护
            startRow = max(1, startRow);
            startCol = max(1, startCol);
            endRow = min(rows, endRow);
            endCol = min(cols, endCol);
            
            croppedImg = img(startRow:endRow, startCol:endCol, :);
        end
        
        function processImg = fullProcess(img, targetRows, targetCols)
            % 组合全套预处理流程
            % 1. 中心裁剪 (优化点：去除背景和衣服，只留人脸中心)
            img = ImagePreprocess.cropCenter(img, 0.55); % 保留中心 55% 区域
            % 2. 灰度化
            img = ImagePreprocess.toGray(img);
            % 3. 去噪
            img = ImagePreprocess.denoise(img);
            % 4. 直方图均衡化 (优化点：增强对比度)
            img = ImagePreprocess.histEq(img);
            % 5. 尺寸归一化
            img = ImagePreprocess.resizeImg(img, targetRows, targetCols);
            % 6. 像素值归一化
            processImg = ImagePreprocess.normalizePixels(img);
        end
        
        %% 新增：核心需求 1 & 2 - 人脸检测与对齐 (Face Alignment)
        function [alignedFace, success] = detectAndAlignFace(img)
            % detectAndAlignFace: 检测人脸，进行关键点对齐(基于双眼)，并裁剪缩放为 224x224 (适配 ResNet-50)
            success = false;
            alignedFace = [];
            
            % 1. 检测人脸
            persistent faceDetector;
            if isempty(faceDetector)
                faceDetector = vision.CascadeObjectDetector('MergeThreshold', 4);
            end
            
            bbox = step(faceDetector, img);
            if isempty(bbox)
                return;
            end
            
            % 取最大的一个人脸框
            areas = bbox(:,3) .* bbox(:,4);
            [~, maxIdx] = max(areas);
            faceBBox = bbox(maxIdx, :);
            
            % 提取人脸区域以进行后续对齐
            faceImg = imcrop(img, faceBBox);
            
            % 2. 人脸对齐 (基于双眼检测进行仿射变换)
            persistent eyeDetector;
            if isempty(eyeDetector)
                eyeDetector = vision.CascadeObjectDetector('EyePairBig');
            end
            
            eyeBBox = step(eyeDetector, faceImg);
            if ~isempty(eyeBBox)
                % 取最大的双眼框
                eyeAreas = eyeBBox(:,3) .* eyeBBox(:,4);
                [~, maxEyeIdx] = max(eyeAreas);
                eBox = eyeBBox(maxEyeIdx, :);
                
                % 估算左右眼中心位置
                leftEyeCenter = [eBox(1) + eBox(3)*0.25, eBox(2) + eBox(4)*0.5];
                rightEyeCenter = [eBox(1) + eBox(3)*0.75, eBox(2) + eBox(4)*0.5];
                
                % 计算倾斜角度
                dY = rightEyeCenter(2) - leftEyeCenter(2);
                dX = rightEyeCenter(1) - leftEyeCenter(1);
                angle = atan2d(dY, dX);
                
                % 旋转原始图像进行对齐 (只旋转人脸部分)
                alignedFaceImg = imrotate(faceImg, angle, 'bicubic', 'crop');
            else
                % 如果未检测到双眼，则跳过旋转
                alignedFaceImg = faceImg;
            end
            
            % 3. 裁剪并缩放为 ResNet-50 的标准输入尺寸 224x224
            alignedFace = imresize(alignedFaceImg, [224, 224]);
            
            % 根据需求：将裁剪后的人脸转换为灰白图像
            alignedFace = ImagePreprocess.toGray(alignedFace);
            
            success = true;
        end
        
        function buildFaceDatabase(sourceDir, targetDir)
            % 建立独立的物理基准数据库 (核心需求 1)
            % 遍历 sourceDir，检测人脸并对齐后保存到 targetDir
            if ~exist(targetDir, 'dir')
                mkdir(targetDir);
            end
            
            files = [dir(fullfile(sourceDir, '**', '*.jpg')); dir(fullfile(sourceDir, '**', '*.png'))];
            if isempty(files)
                disp('未找到图片文件');
                return;
            end
            
            disp(['开始构建人脸数据库，共 ', num2str(length(files)), ' 张图片...']);
            for i = 1:length(files)
                imgPath = fullfile(files(i).folder, files(i).name);
                img = imread(imgPath);
                
                [alignedFace, success] = ImagePreprocess.detectAndAlignFace(img);
                if success
                    % 提取标签 (从文件名提取人名，自动去除末尾的数字和符号)
                    [baseName, ~] = fileparts(files(i).name);
                    personName = regexprep(baseName, '[\d\s_\-]+$', '');
                    if isempty(personName)
                        personName = 'Unknown';
                    end
                    
                    labelDir = fullfile(targetDir, personName);
                    if ~exist(labelDir, 'dir')
                        mkdir(labelDir);
                    end
                    
                    savePath = fullfile(labelDir, files(i).name);
                    imwrite(alignedFace, savePath);
                else
                    disp(['未能在图片中检测到人脸: ', imgPath]);
                end
            end
            disp('人脸数据库构建完成！');
        end
    end
end