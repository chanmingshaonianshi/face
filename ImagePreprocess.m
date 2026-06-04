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

        %% 统一人脸预处理：训练/测试/单图/摄像头共用同一条流程
        function [procImg, success] = prepareFace(img, targetRows, targetCols)
            % prepareFace: 检测人脸 -> 双眼相似变换对齐(消除旋转/尺度差) -> 直方图均衡 -> 归一化
            % gallery 与 probe 必须走同一条流程，PCA 特征空间才能对齐
            [faceCanonical, success] = ImagePreprocess.alignCanonical(img, targetRows, targetCols);
            eqImg = ImagePreprocess.histEq(faceCanonical);
            procImg = ImagePreprocess.normalizePixels(eqImg);
        end

        function [faceCanonical, success] = alignCanonical(img, targetRows, targetCols)
            % alignCanonical: 返回对齐后的灰度规范脸(直方图均衡之前)
            % 拆出来便于不同光照归一化方案(直方图均衡 / Tan-Triggs / LBP)复用同一套对齐脸
            success = false;
            grayFull = ImagePreprocess.toGray(img);
            [faceBBox, found] = ImagePreprocess.detectFaceBBox(img);

            faceCanonical = [];
            if found
                % 首选：基于双眼的相似变换，把眼睛归一化到固定像素位置
                faceCanonical = ImagePreprocess.alignByEyes(grayFull, faceBBox, targetRows, targetCols);
                if ~isempty(faceCanonical)
                    success = true;
                else
                    % 次选：按目标长宽比扩框裁剪(无对齐)
                    roi = ImagePreprocess.expandBBoxToAspect(faceBBox, targetCols, targetRows, ...
                        size(grayFull, 1), size(grayFull, 2));
                    faceGray = grayFull(roi(2):roi(2)+roi(4)-1, roi(1):roi(1)+roi(3)-1);
                    faceCanonical = ImagePreprocess.resizeBilinear(faceGray, targetRows, targetCols);
                    success = true;
                end
            else
                % 兜底：按目标长宽比中心裁剪，仍然避免拉伸畸变
                faceGray = ImagePreprocess.cropToAspect(grayFull, targetCols, targetRows);
                faceCanonical = ImagePreprocess.resizeBilinear(faceGray, targetRows, targetCols);
            end
        end

        function out = tanTriggs(img)
            % tanTriggs: Tan-Triggs 光照归一化 (gamma 校正 -> DoG 带通 -> 鲁棒对比度均衡)
            % 对方向性阴影/过曝远比全局直方图均衡鲁棒，输出 [0,255] uint8 便于复用现有流程
            I = double(img);
            % 1. gamma 压缩高光、提升暗部
            I = I .^ 0.2;
            % 2. 高斯差分(DoG)带通，抑制低频光照、保留五官边缘
            g0 = ImagePreprocess.gaussianBlur(I, 1.0);
            g1 = ImagePreprocess.gaussianBlur(I, 2.0);
            D = g0 - g1;
            % 3. 两段式鲁棒对比度均衡
            a = 0.1; tau = 10.0;
            meanAbs = mean(abs(D(:)) .^ a) ^ (1/a);
            if meanAbs < 1e-8; meanAbs = 1; end
            D = D / meanAbs;
            meanAbs2 = mean(min(abs(D(:)), tau) .^ a) ^ (1/a);
            if meanAbs2 < 1e-8; meanAbs2 = 1; end
            D = D / meanAbs2;
            D = tau * tanh(D / tau);
            % 归一化到 [0,255]
            mn = min(D(:)); mx = max(D(:));
            if mx > mn
                D = (D - mn) / (mx - mn) * 255;
            else
                D = zeros(size(D));
            end
            out = uint8(D);
        end

        function out = gaussianBlur(img, sigma)
            % gaussianBlur: 自主实现可分离高斯模糊(先行后列)，供 Tan-Triggs 的 DoG 使用
            I = double(img);
            radius = max(1, ceil(3 * sigma));
            x = -radius:radius;
            kern = exp(-(x.^2) / (2 * sigma^2));
            kern = kern / sum(kern);
            [rows, cols] = size(I);
            % 行方向卷积(带边界复制)
            tmp = zeros(rows, cols);
            for c = 1:cols
                acc = zeros(rows, 1);
                for t = -radius:radius
                    cc = min(max(c + t, 1), cols);
                    acc = acc + kern(t + radius + 1) * I(:, cc);
                end
                tmp(:, c) = acc;
            end
            % 列方向卷积
            out = zeros(rows, cols);
            for r = 1:rows
                acc = zeros(1, cols);
                for t = -radius:radius
                    rr = min(max(r + t, 1), rows);
                    acc = acc + kern(t + radius + 1) * tmp(rr, :);
                end
                out(r, :) = acc;
            end
        end

        function feat = lbpFeature(img, gridR, gridC)
            % lbpFeature: 分块均匀LBP直方图特征，对单调光照变化天然不变
            % 3x3 基础LBP -> 59维uniform映射 -> gridR x gridC 分块直方图拼接
            if nargin < 2; gridR = 7; end
            if nargin < 3; gridC = 6; end
            I = double(img);
            [rows, cols] = size(I);
            % 1. 计算 LBP 编码图(内部像素)
            uniMap = ImagePreprocess.lbpUniformMap();
            codeImg = zeros(rows, cols);
            offR = [-1 -1 -1 0 1 1 1 0];
            offC = [-1 0 1 1 1 0 -1 -1];
            for r = 2:rows-1
                for c = 2:cols-1
                    center = I(r, c);
                    code = 0;
                    for b = 1:8
                        if I(r + offR(b), c + offC(b)) >= center
                            code = code + 2^(b-1);
                        end
                    end
                    codeImg(r, c) = uniMap(code + 1);
                end
            end
            % 2. 分块直方图(59 bins/块)
            nBins = 59;
            feat = zeros(gridR * gridC * nBins, 1);
            rEdges = round(linspace(2, rows-1, gridR + 1));
            cEdges = round(linspace(2, cols-1, gridC + 1));
            idx = 0;
            for gr = 1:gridR
                for gc = 1:gridC
                    block = codeImg(rEdges(gr):rEdges(gr+1), cEdges(gc):cEdges(gc+1));
                    h = zeros(nBins, 1);
                    bv = block(:);
                    for t = 1:numel(bv)
                        h(bv(t) + 1) = h(bv(t) + 1) + 1;
                    end
                    s = sum(h); if s > 0; h = h / s; end % 块内归一
                    feat(idx*nBins + (1:nBins)) = h;
                    idx = idx + 1;
                end
            end
        end

        function uniMap = lbpUniformMap()
            % lbpUniformMap: 构建 8 位 LBP 的 uniform 映射表(256->59)
            % 跳变<=2 的模式各占一类(0..57)，其余非uniform归入第58类(索引58)
            persistent M;
            if ~isempty(M); uniMap = M; return; end
            M = zeros(256, 1);
            next = 0;
            for v = 0:255
                bits = bitget(v, 1:8);
                trans = sum(bits ~= bits([2:8 1]));
                if trans <= 2
                    M(v + 1) = next;
                    next = next + 1;
                else
                    M(v + 1) = 58; % 非uniform统一桶
                end
            end
            uniMap = M;
        end

        function out = alignByEyes(grayFull, faceBBox, targetRows, targetCols)
            % alignByEyes: 在人脸框内检测双眼，用相似变换(旋转+缩放+平移)把双眼
            % 映射到画布固定位置，逆映射 + 手写双线性采样输出 targetRows x targetCols
            % 检测不到双眼则返回空，由上层回退到扩框裁剪
            out = [];
            persistent eyeDet;
            if isempty(eyeDet)
                eyeDet = vision.CascadeObjectDetector('EyePairBig', 'MergeThreshold', 4);
            end

            [imH, imW] = size(grayFull);
            x = max(1, round(faceBBox(1)));
            y = max(1, round(faceBBox(2)));
            x2 = min(imW, round(faceBBox(1) + faceBBox(3) - 1));
            y2 = min(imH, round(faceBBox(2) + faceBBox(4) - 1));
            if x2 <= x || y2 <= y
                return;
            end
            faceCrop = grayFull(y:y2, x:x2);

            eb = step(eyeDet, faceCrop);
            if isempty(eb)
                return;
            end
            areas = eb(:,3) .* eb(:,4);
            [~, mi] = max(areas);
            e = eb(mi, :);

            % 双眼中心(全图坐标系)
            leftEye  = [x + e(1) + e(3)*0.25, y + e(2) + e(4)*0.5];
            rightEye = [x + e(1) + e(3)*0.75, y + e(2) + e(4)*0.5];

            % 画布上双眼的目标位置(列, 行)
            desiredLeft  = [0.30 * targetCols, 0.40 * targetRows];
            desiredRight = [0.70 * targetCols, 0.40 * targetRows];

            % 求相似变换 [a -b; b a] 及平移，使 src 双眼 -> dst 双眼
            dx  = rightEye(1) - leftEye(1);
            dy  = rightEye(2) - leftEye(2);
            dxp = desiredRight(1) - desiredLeft(1);
            dyp = desiredRight(2) - desiredLeft(2);
            det = dx*dx + dy*dy;
            if det < 1e-6
                return;
            end
            a = (dx*dxp + dy*dyp) / det;
            b = (dx*dyp - dy*dxp) / det;
            tx = desiredLeft(1) - (a*leftEye(1) - b*leftEye(2));
            ty = desiredLeft(2) - (b*leftEye(1) + a*leftEye(2));

            % 逆映射 dst->src 用的系数
            denom = a*a + b*b;
            if denom < 1e-9
                return;
            end

            out = zeros(targetRows, targetCols);
            gD = double(grayFull);
            for ro = 1:targetRows
                for co = 1:targetCols
                    px = co - tx;
                    py = ro - ty;
                    xs = ( a*px + b*py) / denom;
                    ys = (-b*px + a*py) / denom;
                    % 手写双线性采样
                    x0 = floor(xs); y0 = floor(ys);
                    fx = xs - x0;   fy = ys - y0;
                    x1 = x0 + 1;    y1 = y0 + 1;
                    if x0 < 1 || y0 < 1 || x1 > imW || y1 > imH
                        out(ro, co) = 0;
                        continue;
                    end
                    v00 = gD(y0, x0); v01 = gD(y0, x1);
                    v10 = gD(y1, x0); v11 = gD(y1, x1);
                    top = v00 + (v01 - v00) * fx;
                    bot = v10 + (v11 - v10) * fx;
                    out(ro, co) = top + (bot - top) * fy;
                end
            end
            out = uint8(out);
        end

        function [faceBBox, found] = detectFaceBBox(img)
            % detectFaceBBox: 级联检测人脸(CART->LBP->Profile)，返回最大的人脸框
            found = false;
            faceBBox = [];
            persistent dCART dLBP dProfile;
            if isempty(dCART)
                minFaceSize = [60, 60];
                dCART = vision.CascadeObjectDetector('FrontalFaceCART', ...
                    'MergeThreshold', 4, 'MinSize', minFaceSize);
                dLBP = vision.CascadeObjectDetector('FrontalFaceLBP', ...
                    'MergeThreshold', 4, 'MinSize', minFaceSize);
                dProfile = vision.CascadeObjectDetector('ProfileFace', ...
                    'MergeThreshold', 4, 'MinSize', minFaceSize);
            end

            grayDetect = ImagePreprocess.toGray(img);
            bbox = step(dCART, grayDetect);
            if isempty(bbox)
                bbox = step(dLBP, grayDetect);
            end
            if isempty(bbox)
                bbox = step(dProfile, grayDetect);
            end
            if isempty(bbox)
                return;
            end

            areas = bbox(:,3) .* bbox(:,4);
            [~, mi] = max(areas);
            faceBBox = bbox(mi, :);
            found = true;
        end

        function roi = expandBBoxToAspect(bbox, targetCols, targetRows, imH, imW)
            % expandBBoxToAspect: 以人脸框中心为基准，加边距并调整到目标长宽比，裁剪框不超出图像
            x = bbox(1); y = bbox(2); w = bbox(3); h = bbox(4);
            cx = x + w/2; cy = y + h/2;

            % 加边距，纳入额头与下巴
            marginScale = 1.35;
            w = w * marginScale;
            h = h * marginScale;

            % 调整到目标长宽比 (宽:高 = targetCols:targetRows)
            aspect = targetCols / targetRows;
            if (w / h) > aspect
                h = w / aspect;
            else
                w = h * aspect;
            end

            % 若超出边界，保持长宽比向内收缩
            maxW = 2 * min(cx - 0.5, imW - cx + 0.5);
            maxH = 2 * min(cy - 0.5, imH - cy + 0.5);
            if w > maxW
                w = maxW; h = w / aspect;
            end
            if h > maxH
                h = maxH; w = h * aspect;
            end

            x = round(cx - w/2); y = round(cy - h/2);
            w = floor(w); h = floor(h);
            x = max(1, x); y = max(1, y);
            if x + w - 1 > imW; w = imW - x; end
            if y + h - 1 > imH; h = imH - y; end
            roi = [x, y, w, h];
        end

        function out = cropToAspect(img, targetCols, targetRows)
            % cropToAspect: 按目标长宽比做中心裁剪(不缩放)，避免后续硬缩放拉伸
            [rows, cols, ~] = size(img);
            aspect = targetCols / targetRows;
            if (cols / rows) > aspect
                newW = round(rows * aspect); newH = rows;
            else
                newH = round(cols / aspect); newW = cols;
            end
            sc = max(1, round((cols - newW)/2) + 1);
            sr = max(1, round((rows - newH)/2) + 1);
            ec = min(cols, sc + newW - 1);
            er = min(rows, sr + newH - 1);
            out = img(sr:er, sc:ec);
        end

        function out = resizeBilinear(img, targetRows, targetCols)
            % resizeBilinear: 自主实现双线性插值缩放(像素中心对齐)，比最近邻更平滑
            imgD = double(img);
            [rows, cols] = size(imgD);
            out = zeros(targetRows, targetCols);
            rowScale = rows / targetRows;
            colScale = cols / targetCols;

            for r = 1:targetRows
                fr = (r - 0.5) * rowScale + 0.5;
                r0 = floor(fr); dr = fr - r0; r1 = r0 + 1;
                r0 = min(max(r0, 1), rows); r1 = min(max(r1, 1), rows);
                for c = 1:targetCols
                    fc = (c - 0.5) * colScale + 0.5;
                    c0 = floor(fc); dc = fc - c0; c1 = c0 + 1;
                    c0 = min(max(c0, 1), cols); c1 = min(max(c1, 1), cols);

                    v00 = imgD(r0, c0); v01 = imgD(r0, c1);
                    v10 = imgD(r1, c0); v11 = imgD(r1, c1);
                    top = v00 + (v01 - v00) * dc;
                    bot = v10 + (v11 - v10) * dc;
                    out(r, c) = top + (bot - top) * dr;
                end
            end
            out = uint8(out);
        end

        %% 核心：人脸检测与对齐 (Face Alignment)
        function [alignedFace, success] = detectAndAlignFace(img)
            % detectAndAlignFace: 检测人脸，进行关键点对齐(基于双眼)，并裁剪缩放为统一尺寸
            success = false;
            alignedFace = [];
            
            % 1. 多检测器级联检测人脸（CART -> LBP -> Profile）
            persistent faceDetectorCART faceDetectorLBP faceDetectorProfile;
            if isempty(faceDetectorCART)
                minFaceSize = [80, 80];
                faceDetectorCART = vision.CascadeObjectDetector('FrontalFaceCART', ...
                    'MergeThreshold', 4, 'MinSize', minFaceSize);
                faceDetectorLBP = vision.CascadeObjectDetector('FrontalFaceLBP', ...
                    'MergeThreshold', 4, 'MinSize', minFaceSize);
                faceDetectorProfile = vision.CascadeObjectDetector('ProfileFace', ...
                    'MergeThreshold', 4, 'MinSize', minFaceSize);
            end

            grayDetect = ImagePreprocess.toGray(img);

            bbox = step(faceDetectorCART, grayDetect);
            if isempty(bbox)
                bbox = step(faceDetectorLBP, grayDetect);
            end
            if isempty(bbox)
                bbox = step(faceDetectorProfile, grayDetect);
            end
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
            
            % 3. 裁剪并缩放为统一尺寸 224x224（后续流程可再缩放）
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
                    % 提取标签：优先使用“下划线前”的人名，如 张三_XXXXX.jpg -> 张三
                    [baseName, ~] = fileparts(files(i).name);
                    token = regexp(baseName, '^(.+?)_', 'tokens', 'once');
                    if ~isempty(token)
                        personName = token{1};
                    else
                        % 兼容无下划线命名：去除末尾数字和连接符
                        personName = regexprep(baseName, '[\d\s_\-]+$', '');
                    end
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
