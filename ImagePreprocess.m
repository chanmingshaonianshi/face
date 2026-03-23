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
    end
end