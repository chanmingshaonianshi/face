classdef ClassifierCore
    % ClassifierCore 自主实现的分类器核心类
    % 包含最近邻分类器(余弦距离匹配)及准确率计算功能
    
    methods(Static)
        function [bestMatchIndex, minDistance] = classify(testFeature, trainFeatures)
            % classify: 自主编写最近邻分类器(优化：余弦距离匹配分类器)
            % 相比欧式距离，余弦距离对光照变化和特征缩放更鲁棒
            % 输入:
            %   testFeature: K x 1 的单样本待测特征向量
            %   trainFeatures: K x N 的训练集特征向量矩阵
            % 输出:
            %   bestMatchIndex: 最佳匹配的训练样本索引
            %   minDistance: 最小余弦距离
            
            [~, N] = size(trainFeatures);
            minDistance = inf;
            bestMatchIndex = -1;
            
            % 计算待测样本的范数 (模长)
            normTest = sqrt(sum(testFeature.^2));
            if normTest == 0
                normTest = 1; % 防止除零
            end
            
            for i = 1:N
                trainFeat = trainFeatures(:, i);
                
                % 计算训练样本的范数
                normTrain = sqrt(sum(trainFeat.^2));
                if normTrain == 0
                    normTrain = 1;
                end
                
                % 计算余弦相似度
                cosSim = sum(testFeature .* trainFeat) / (normTest * normTrain);
                
                % 将相似度转换为距离度量 (1 - cosSim)
                dist = 1 - cosSim;
                
                % 更新最小距离及最佳匹配的数据库索引
                if dist < minDistance
                    minDistance = dist;
                    bestMatchIndex = i;
                end
            end
        end
        
        function predictedIndices = classifyBatch(testFeatures, trainFeatures)
            % 批量测试样本识别
            % 输入:
            %   testFeatures: K x M 测试特征矩阵
            %   trainFeatures: K x N 训练特征矩阵
            % 输出:
            %   predictedIndices: 1 x M 的最佳匹配索引数组
            
            M = size(testFeatures, 2);
            predictedIndices = zeros(1, M);
            for i = 1:M
                [bestMatchIndex, ~] = ClassifierCore.classify(testFeatures(:, i), trainFeatures);
                predictedIndices(i) = bestMatchIndex;
            end
        end
        
        function [bestLabel, bestIndex, minGroupDistance] = classifyByGroup(testFeature, trainFeatures, trainLabels)
            % 基于类别聚合的最近邻分类：与每个类别的所有样本比较，取该类别的最小距离作为该类别分数
            % 输入:
            %   testFeature: K x 1 待测特征
            %   trainFeatures: K x N 训练特征
            %   trainLabels: 1 x N 训练标签（cell 或 string）
            % 输出:
            %   bestLabel: 最匹配的人员标签
            %   bestIndex: 该人员中与测试样本距离最近的训练样本索引
            %   minGroupDistance: 对应的最小组内距离（余弦距离）
            
            if iscell(trainLabels)
                lbls = strings(1, numel(trainLabels));
                for i = 1:numel(trainLabels)
                    lbls(i) = string(trainLabels{i});
                end
            else
                lbls = string(trainLabels);
            end
            
            [~, N] = size(trainFeatures);
            normTest = sqrt(sum(testFeature.^2));
            if normTest == 0
                normTest = 1;
            end
            
            uniqueLabels = lbls(1);
            groupMinDist = [inf];
            groupBestIdx = [-1];
            
            for i = 1:N
                trainFeat = trainFeatures(:, i);
                normTrain = sqrt(sum(trainFeat.^2));
                if normTrain == 0
                    normTrain = 1;
                end
                cosSim = sum(testFeature .* trainFeat) / (normTest * normTrain);
                dist = 1 - cosSim;
                
                % 查找当前标签在 uniqueLabels 中的位置
                label = lbls(i);
                pos = -1;
                for j = 1:length(uniqueLabels)
                    if label == uniqueLabels(j)
                        pos = j;
                        break;
                    end
                end
                if pos == -1
                    uniqueLabels(end+1) = label; %#ok<AGROW>
                    groupMinDist(end+1) = dist; %#ok<AGROW>
                    groupBestIdx(end+1) = i; %#ok<AGROW>
                else
                    if dist < groupMinDist(pos)
                        groupMinDist(pos) = dist;
                        groupBestIdx(pos) = i;
                    end
                end
            end
            
            % 选取组距离最小的类别
            minGroupDistance = inf;
            minPos = 1;
            for j = 1:length(groupMinDist)
                if groupMinDist(j) < minGroupDistance
                    minGroupDistance = groupMinDist(j);
                    minPos = j;
                end
            end
            bestLabel = uniqueLabels(minPos);
            bestIndex = groupBestIdx(minPos);
        end
        
        function [overallAcc, classAccList] = calcAccuracy(testLabels, predictedLabels)
            % calcAccuracy: 基于测试样本集完成批量识别测试，计算整体识别正确率、单类识别正确率
            % 输入:
            %   testLabels: 1 x M 测试集的真实人员标签
            %   predictedLabels: 1 x M 测试集的预测人员标签
            % 输出:
            %   overallAcc: 整体识别正确率 (0~100)
            %   classAccList: N x 2 的 cell 数组，{类别标签, 该类别的正确率(0~100)}
            
            totalSamples = length(testLabels);
            if totalSamples == 0
                overallAcc = 0;
                classAccList = {};
                return;
            end
            
            % 自主编写判断并计算正确数量
            correctCount = 0;
            for i = 1:totalSamples
                if testLabels(i) == predictedLabels(i)
                    correctCount = correctCount + 1;
                end
            end
            overallAcc = correctCount / totalSamples * 100;
            
            % 计算单类识别正确率
            % 找出所有的类别 (此处自主实现类似 unique 的简单逻辑去重，确保 100% 自主编写)
            uniqueLabels = testLabels(1); % 使用第一个元素初始化，避免 string 和 double 比较错误
            for i = 2:totalSamples
                isExist = false;
                for j = 1:length(uniqueLabels)
                    if testLabels(i) == uniqueLabels(j)
                        isExist = true;
                        break;
                    end
                end
                if ~isExist
                    uniqueLabels(end+1) = testLabels(i); %#ok<AGROW>
                end
            end
            
            numClasses = length(uniqueLabels);
            classAccList = cell(numClasses, 2); % 改为 cell 数组以兼容 string 类型标签
            
            for i = 1:numClasses
                label = uniqueLabels(i);
                
                classTotal = 0;
                classCorrect = 0;
                
                for j = 1:totalSamples
                    if testLabels(j) == label
                        classTotal = classTotal + 1;
                        if predictedLabels(j) == label
                            classCorrect = classCorrect + 1;
                        end
                    end
                end
                
                classAccList{i, 1} = label;
                if classTotal > 0
                    classAccList{i, 2} = classCorrect / classTotal * 100;
                else
                    classAccList{i, 2} = 0;
                end
            end
        end
    end
end
