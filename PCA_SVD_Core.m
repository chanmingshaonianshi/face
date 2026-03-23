classdef PCA_SVD_Core
    % PCA_SVD_Core 自主实现的 PCA 与 SVD 核心算法类
    % 包含协方差矩阵计算、奇异值分解(SVD)、特征脸生成、样本投影等核心功能
    
    methods(Static)
        function [meanFace, eigenFaces, V, D] = computePCA_SVD(X, numComponents)
            % computePCA_SVD: 基于训练样本集计算人脸平均脸、去中心化处理并生成特征脸
            % 输入:
            %   X: d x N 数据矩阵 (d 为像素总数，N 为样本数量)
            %   numComponents: 保留的主成分数量 (特征脸个数)
            % 输出:
            %   meanFace: d x 1 的平均脸向量
            %   eigenFaces: d x numComponents 的特征脸矩阵
            %   V, D: 协方差矩阵的特征向量与特征值对角阵
            
            [d, N] = size(X);
            
            % 1. 自主计算平均脸
            meanFace = zeros(d, 1);
            for i = 1:N
                meanFace = meanFace + X(:, i);
            end
            meanFace = meanFace / N;
            
            % 2. 去中心化处理
            Phi = zeros(d, N);
            for i = 1:N
                Phi(:, i) = X(:, i) - meanFace;
            end
            
            % 3. 自主计算协方差矩阵相关结构
            % 直接计算 d x d 的 C = Phi * Phi' 会导致内存溢出
            % 采用 SVD/PCA 技巧，计算 L = Phi' * Phi，大小为 N x N
            L = Phi' * Phi;
            
            % 4. 自主编写 SVD 奇异值分解相关函数，求解特征值与特征向量
            [V, D] = PCA_SVD_Core.myEigJacobi(L);
            
            % 提取对角线上的特征值并排序 (降序)
            eigenValues = diag(D);
            [sortedEigenValues, sortIdx] = sort(eigenValues, 'descend');
            
            % 优化：通过能量占比动态决定保留的主成分个数，而不是写死
            totalEnergy = sum(sortedEigenValues);
            cumulativeEnergy = 0;
            energyThreshold = 0.95; % 设定保留 95% 的能量
            dynamicComponents = 0;
            
            for i = 1:length(sortedEigenValues)
                cumulativeEnergy = cumulativeEnergy + sortedEigenValues(i);
                dynamicComponents = dynamicComponents + 1;
                if (cumulativeEnergy / totalEnergy) >= energyThreshold
                    break;
                end
            end
            
            % 选取传入的主成分数和动态计算的主成分数中较大的一个，确保特征充足
            if nargin < 2
                numComponents = dynamicComponents; 
            else
                numComponents = max(numComponents, dynamicComponents);
            end
            numComponents = min(numComponents, length(sortedEigenValues));
            
            % 取出排好序的特征向量
            V_sorted = V(:, sortIdx);
            V_k = V_sorted(:, 1:numComponents);
            
            % 5. 自主编写特征脸生成函数，基于筛选后的主成分构建特征脸空间
            % U = Phi * V_k，得到 d x numComponents 的矩阵
            U = Phi * V_k;
            
            % 归一化每个特征脸，使其成为单位向量
            for i = 1:numComponents
                norm_ui = sqrt(sum(U(:, i).^2));
                if norm_ui > 1e-10
                    U(:, i) = U(:, i) / norm_ui;
                end
            end
            
            eigenFaces = U;
        end
        
        function features = project(X, meanFace, eigenFaces)
            % project: 自主编写样本投影函数
            % 将训练 / 待测人脸样本投影到特征脸空间，生成对应的特征向量
            % 输入:
            %   X: d x M 数据矩阵 (M 个待投影样本)
            %   meanFace: d x 1 的平均脸向量
            %   eigenFaces: d x K 的特征脸矩阵
            % 输出:
            %   features: K x M 的投影特征向量矩阵
            
            [d, M] = size(X);
            Phi = zeros(d, M);
            for i = 1:M
                Phi(:, i) = X(:, i) - meanFace;
            end
            
            % 投影到特征空间: features = U' * Phi
            features = eigenFaces' * Phi;
        end
        
        function [V, D] = myEigJacobi(A, tol, maxIter)
            % myEigJacobi: 自主实现的雅可比方法求解对称矩阵的特征值与特征向量
            % 替代 MATLAB 内置的 eig() 或 svd()
            if nargin < 2
                tol = 1e-8;
            end
            if nargin < 3
                maxIter = 1000;
            end
            
            n = size(A, 1);
            V = eye(n);
            D = A;
            
            for iter = 1:maxIter
                % 寻找绝对值最大的非对角线元素
                max_val = 0;
                p = 1; q = 2;
                for i = 1:n-1
                    for j = i+1:n
                        if abs(D(i,j)) > max_val
                            max_val = abs(D(i,j));
                            p = i;
                            q = j;
                        end
                    end
                end
                
                % 如果最大非对角元足够小，则认为已收敛，退出循环
                if max_val < tol
                    break;
                end
                
                % 计算旋转角度 theta
                if D(p,p) == D(q,q)
                    theta = pi/4;
                else
                    theta = 0.5 * atan(2 * D(p,q) / (D(p,p) - D(q,q)));
                end
                
                c = cos(theta);
                s = sin(theta);
                
                % 仅更新受影响的行列，提高计算效率
                D_pp = D(p,p);
                D_qq = D(q,q);
                D_pq = D(p,q);
                
                D(p,p) = c^2 * D_pp + s^2 * D_qq + 2*c*s * D_pq;
                D(q,q) = s^2 * D_pp + c^2 * D_qq - 2*c*s * D_pq;
                D(p,q) = 0;
                D(q,p) = 0;
                
                for k = 1:n
                    if k ~= p && k ~= q
                        D_pk = D(p,k);
                        D_qk = D(q,k);
                        
                        D(p,k) = c * D_pk + s * D_qk;
                        D(k,p) = D(p,k);
                        
                        D(q,k) = -s * D_pk + c * D_qk;
                        D(k,q) = D(q,k);
                    end
                end
                
                % 更新特征向量矩阵
                for k = 1:n
                    V_kp = V(k,p);
                    V_kq = V(k,q);
                    
                    V(k,p) = c * V_kp + s * V_kq;
                    V(k,q) = -s * V_kp + c * V_kq;
                end
            end
        end
    end
end