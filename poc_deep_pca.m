% poc_deep_pca.m — PoC: 深度 embedding + PCA/SVD + KNN 准确率验证
% 不依赖 GUI，纯脚本，在 MATLAB 中直接运行

function poc_deep_pca()
    rootDir = fileparts(mfilename('fullpath'));
    if isempty(rootDir), rootDir = pwd; end

    % ---- 加载 embedding ----
    trainMat = load(fullfile(rootDir, 'train_data_embeddings.mat'));
    testMat  = load(fullfile(rootDir, 'test_data_embeddings.mat'));

    Xtrain = trainMat.embeddings';     % d x N  (512 x 256)
    trainLabels = trainMat.labels;
    Xtest  = testMat.embeddings';      % d x M  (512 x 64)
    testLabels  = testMat.labels;

    d = size(Xtrain, 1);
    N = size(Xtrain, 2);
    M = size(Xtest, 2);
    fprintf('Loaded: train %d x %d, test %d x %d\n', d, N, d, M);

    % ---- PCA+SVD 训练（与现有 pipeline 一致） ----
    numComponents = 120;
    [meanFace, eigenFaces, ~, D] = PCA_SVD_Core.computePCA_SVD(Xtrain, numComponents);
    DBFeatures = PCA_SVD_Core.project(Xtrain, meanFace, eigenFaces);

    % 特征标准化（与 eval_pipeline 一致）
    FeatureScale = std(DBFeatures, 0, 2);
    FeatureScale(FeatureScale < 1e-6) = 1;
    DBFeatures = DBFeatures ./ FeatureScale;
    DBFeatures = l2cols(DBFeatures);

    ev = diag(D);
    totalVar = sum(ev);
    explainedVar = sum(ev(1:min(numComponents, length(ev)))) / totalVar * 100;
    fprintf('PCA 主成分数: %d\n', size(eigenFaces, 2));
    fprintf('方差解释率: %.1f%%\n', explainedVar);

    % ---- 测试集 ----
    TestFeatures = PCA_SVD_Core.project(Xtest, meanFace, eigenFaces);
    TestFeatures = TestFeatures ./ FeatureScale;
    TestFeatures = l2cols(TestFeatures);

    % ---- KNN 分类 ----
    knnK = 5;
    pred = cell(1, M);
    for i = 1:M
        [bestLabel, ~, ~] = ClassifierCore.classifyKNNByLabel( ...
            TestFeatures(:, i), DBFeatures, trainLabels, knnK);
        pred{i} = char(bestLabel);
    end

    [overallAcc, classAcc] = ClassifierCore.calcAccuracy(string(testLabels), string(pred));

    fprintf('\n===== PoC RESULT =====\n');
    fprintf('Overall Accuracy: %.2f%%\n', overallAcc);
    fprintf('Correct: %d / %d\n', round(M * overallAcc / 100), M);

    % 最差类别
    accs = cell2mat(classAcc(:, 2));
    [~, ord] = sort(accs, 'ascend');
    fprintf('\n最差类别 (准确率 < 100%%):\n');
    badCount = 0;
    for r = 1:min(length(ord), 16)
        if classAcc{ord(r), 2} < 100
            fprintf('  %-18s %.0f%%\n', string(classAcc{ord(r), 1}), classAcc{ord(r), 2});
            badCount = badCount + 1;
        end
    end
    if badCount == 0
        fprintf('  (全部 100%%！)\n');
    end
end

function F = l2cols(F)
    for i = 1:size(F, 2)
        n = norm(F(:, i));
        if n > 0, F(:, i) = F(:, i) / n; end
    end
end
