%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('数据集.xlsx');
total_samples = size(res, 1); % 样本总数 (103)

%%  设置 5 折交叉验证
K = 5;
% 设置随机种子以保证每次运行结果可复现（回复审稿人必备）
rng(42); 
cv = cvpartition(total_samples, 'KFold', K);

%%  初始化存储结果的变量
R2_train_list = zeros(1, K);
R2_test_list  = zeros(1, K);
RMSE_train_list = zeros(1, K);
RMSE_test_list  = zeros(1, K);
MAE_train_list = zeros(1, K);
MAE_test_list  = zeros(1, K);

% 用于存储“折外预测”(Out-of-Fold) 的完整结果，用于绘制全局散点图
OOF_Pred = zeros(total_samples, 1);
OOF_True = res(:, 8);

%%  开始交叉验证循环
for fold = 1 : K
    fprintf('\n================ 开始第 %d / %d 折交叉验证 ================\n', fold, K);
    
    % 1. 划分训练集和测试集索引
    trainIdx = training(cv, fold);
    testIdx  = test(cv, fold);
    
    % 2. 提取数据并转置 (特征为行，样本为列)
    P_train = res(trainIdx, 1:7)';
    T_train = res(trainIdx, 8)';
    M = size(P_train, 2);
    
    P_test = res(testIdx, 1:7)';
    T_test = res(testIdx, 8)';
    N = size(P_test, 2);
    
    % 3. 数据归一化 (严格在循环内进行，防止数据泄露)
    [p_train, ps_input]  = mapminmax(P_train, 0, 1);
    p_test               = mapminmax('apply', P_test, ps_input);
    [t_train, ps_output] = mapminmax(T_train, 0, 1);
    t_test               = mapminmax('apply', T_test, ps_output);
    
    % 4. 节点个数设定
    inputnum  = size(p_train, 1);  
    hiddennum = 5;                 
    outputnum = size(t_train, 1);   
    
    % 5. 建立网络
    net = newff(p_train, t_train, hiddennum);
    net.trainParam.epochs     = 1000;      
    net.trainParam.goal       = 1e-6;      
    net.trainParam.lr         = 0.01;      
    net.trainParam.showWindow = 0; % 关闭网络自带的训练窗口以免弹窗太多
    
    % 6. PSO 参数初始化
    c1 = 4.494; c2 = 4.494;
    maxgen = 50; sizepop = 5;
    Vmax = 1.0; Vmin = -1.0; popmax = 1.0; popmin = -1.0;
    numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;
    
    pop = zeros(sizepop, numsum);
    V = zeros(sizepop, numsum);
    fitness = zeros(1, sizepop);
    
    for i = 1 : sizepop
        pop(i, :) = rands(1, numsum);  
        V(i, :) = rands(1, numsum);    
        fitness(i) = fun(pop(i, :), hiddennum, net, p_train, t_train);
    end
    
    [fitnesszbest, bestindex] = min(fitness);
    zbest = pop(bestindex, :);     
    gbest = pop;                   
    fitnessgbest = fitness;        
    BestFit = fitnesszbest;        
    
    % 7. PSO 迭代寻优
    for i = 1 : maxgen
        for j = 1 : sizepop
            V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
            V(j, (V(j, :) > Vmax)) = Vmax;
            V(j, (V(j, :) < Vmin)) = Vmin;
            
            pop(j, :) = pop(j, :) + 0.2 * V(j, :);
            pop(j, (pop(j, :) > popmax)) = popmax;
            pop(j, (pop(j, :) < popmin)) = popmin;
            
            pos = unidrnd(numsum);
            if rand > 0.85
                pop(j, pos) = rands(1, 1);
            end
            
            fitness(j) = fun(pop(j, :), hiddennum, net, p_train, t_train);
        end
        
        for j = 1 : sizepop
            if fitness(j) < fitnessgbest(j)
                gbest(j, :) = pop(j, :);
                fitnessgbest(j) = fitness(j);
            end
            if fitness(j) < fitnesszbest
                zbest = pop(j, :);
                fitnesszbest = fitness(j);
            end
        end
    end
    
    % 8. 提取最优权值并赋值给网络
    w1 = zbest(1 : inputnum * hiddennum);
    B1 = zbest(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
    w2 = zbest(inputnum * hiddennum + hiddennum + 1 : inputnum * hiddennum + hiddennum + hiddennum * outputnum);
    B2 = zbest(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);
    
    net.Iw{1, 1} = reshape(w1, hiddennum, inputnum);
    net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);
    net.b{1}     = reshape(B1, hiddennum, 1);
    net.b{2}     = B2';
    
    % 9. 网络训练 (使用最优初始权重)
    net = train(net, p_train, t_train);
    
    % 10. 仿真预测与反归一化
    t_sim1 = sim(net, p_train);
    t_sim2 = sim(net, p_test );
    
    T_sim1 = mapminmax('reverse', t_sim1, ps_output);
    T_sim2 = mapminmax('reverse', t_sim2, ps_output);
    
    % 将该折的测试集预测结果存入全局 OOF 数组中
    OOF_Pred(testIdx) = T_sim2';
    
    % 11. 计算该折的指标并记录
    RMSE_train_list(fold) = sqrt(sum((T_sim1 - T_train).^2, 2)' ./ M);
    RMSE_test_list(fold)  = sqrt(sum((T_sim2 - T_test) .^2, 2)' ./ N);
    
    R2_train_list(fold) = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
    R2_test_list(fold)  = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
    
    MAE_train_list(fold) = sum(abs(T_sim1 - T_train), 2)' ./ M;
    MAE_test_list(fold)  = sum(abs(T_sim2 - T_test ), 2)' ./ N;
    
    fprintf('第 %d 折 Test R2 = %.4f, Test RMSE = %.4f\n', fold, R2_test_list(fold), RMSE_test_list(fold));
end

%% ================= 交叉验证结果统计与输出 =================
clc;
fprintf('================ 5折交叉验证最终统计结果 ================\n');
fprintf('Train R2   : %.4f ± %.4f\n', mean(R2_train_list), std(R2_train_list));
fprintf('Test R2    : %.4f ± %.4f\n', mean(R2_test_list), std(R2_test_list));
fprintf('---------------------------------------------------------\n');
fprintf('Train RMSE : %.4f ± %.4f\n', mean(RMSE_train_list), std(RMSE_train_list));
fprintf('Test RMSE  : %.4f ± %.4f\n', mean(RMSE_test_list), std(RMSE_test_list));
fprintf('---------------------------------------------------------\n');
fprintf('Train MAE  : %.4f ± %.4f\n', mean(MAE_train_list), std(MAE_train_list));
fprintf('Test MAE   : %.4f ± %.4f\n', mean(MAE_test_list), std(MAE_test_list));

% 计算整体(OOF)性能
Overall_R2 = 1 - norm(OOF_True - OOF_Pred)^2 / norm(OOF_True - mean(OOF_True))^2;
Overall_RMSE = sqrt(mean((OOF_True - OOF_Pred).^2));
fprintf('---------------------------------------------------------\n');
fprintf('整体折外预测 (OOF) R2   : %.4f\n', Overall_R2);
fprintf('整体折外预测 (OOF) RMSE : %.4f\n', Overall_RMSE);
fprintf('=========================================================\n');

%% ================= 绘图部分 =================

% 1. 绘制各折 R2 性能箱线图 (展示稳定性)
figure('Position', [100, 100, 600, 400]);
boxplot([R2_train_list', R2_test_list'], 'Labels', {'Train R^2', 'Test R^2'});
title('5-Fold Cross Validation R^2 Distribution');
ylabel('R^2 Value');
grid on;

% 2. 绘制全局交叉验证散点图 (OOF Parity Plot) - 非常重要！向审稿人证明泛化性
figure('Position', [150, 150, 600, 500]);
scatter(OOF_True, OOF_Pred, 40, 'b', 'filled', 'MarkerEdgeColor', 'k');
hold on;
% 绘制 y=x 对角线
min_val = min([OOF_True; OOF_Pred]);
max_val = max([OOF_True; OOF_Pred]);
plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);

xlabel('True Values (Entire Dataset)');
ylabel('Cross-Validated Predicted Values (OOF)');
title(sprintf('Cross-Validation Parity Plot\nOverall R^2 = %.4f, RMSE = %.4f', Overall_R2, Overall_RMSE));
legend('CV Predictions', 'Ideal (y=x)', 'Location', 'NorthWest');
grid on;
axis square;