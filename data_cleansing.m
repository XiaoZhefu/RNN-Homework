clc; clear; close all;

%% 加载数据
% 加载 CSV 数据
script_dir = fileparts(mfilename('fullpath'));
data = readtable(fullfile(script_dir, 'Dateset_For_homework.csv'), 'VariableNamingRule', 'preserve');
varName = data.Properties.VariableNames;
varNameFeature = varName(1:7);    % 前 7 列为输入特征
varNameTarget = varName(8:11);    % 第 8-11 列为四个车轮垂向力标签
varNameExtra = varName(12:17);    % 其余列为额外状态量
% 每列单独导出为工作区 table 变量
for i = 1:width(data)
    colName = varName{i};
    tblName = strrep(colName, '..', 'Ddot');
    tblName = strrep(tblName, '.', 'Dot');
    tblName = strrep(tblName, '_', '');
    tblName = strcat(tblName, 'Raw');
    assignin('base', tblName, data(:, colName));
end
% 确认加载结果
disp('    原始数据加载完成。');
disp(['    列数: ', num2str(width(data))]);
disp(['    行数: ', num2str(height(data))]);
disp(['    特征列: ', strjoin(varNameFeature, ', ')]);
disp(['    目标列: ', strjoin(varNameTarget, ', ')]);
disp(['    其他列: ', strjoin(varNameExtra, ', ')]);

%% 数据预处理
sFactor = 0.25;  % 平滑因子
% 特征 X：平滑 + Z-score 归一化
zcDdot      = normalize(smoothdata(zcDdotRaw,      'sgolay', 'SmoothingFactor', sFactor), 'zscore');
alphacDdot  = normalize(smoothdata(alphacDdotRaw,  'sgolay', 'SmoothingFactor', sFactor), 'zscore');
beitacDdot  = normalize(smoothdata(beitacDdotRaw,  'sgolay', 'SmoothingFactor', sFactor), 'zscore');
zb1Ddot     = normalize(smoothdata(zb1DdotRaw,     'sgolay', 'SmoothingFactor', sFactor), 'zscore');
alphab1Ddot = normalize(smoothdata(alphab1DdotRaw, 'sgolay', 'SmoothingFactor', sFactor), 'zscore');
zb2Ddot     = normalize(smoothdata(zb2DdotRaw,     'sgolay', 'SmoothingFactor', sFactor), 'zscore');
alphab2Ddot = normalize(smoothdata(alphab2DdotRaw, 'sgolay', 'SmoothingFactor', sFactor), 'zscore');
% 标签 y：Z-score 归一化
F1 = normalize(F1Raw, 'zscore');
F2 = normalize(F2Raw, 'zscore');
F3 = normalize(F3Raw, 'zscore');
F4 = normalize(F4Raw, 'zscore');
% 输出信息
disp('    预处理完成：特征平滑+归一化，标签仅归一化。');

%% 导出清洗后数据
% 保持原 CSV 前 11 列的列名和顺序，仅替换为清洗后的数据
cleanedData = data(:, 1:11);
cleanedData{:, 1}  = table2array(zcDdot);
cleanedData{:, 2}  = table2array(alphacDdot);
cleanedData{:, 3}  = table2array(beitacDdot);
cleanedData{:, 4}  = table2array(zb1Ddot);
cleanedData{:, 5}  = table2array(alphab1Ddot);
cleanedData{:, 6}  = table2array(zb2Ddot);
cleanedData{:, 7}  = table2array(alphab2Ddot);
cleanedData{:, 8}  = table2array(F1);
cleanedData{:, 9}  = table2array(F2);
cleanedData{:, 10} = table2array(F3);
cleanedData{:, 11} = table2array(F4);
cleaned_data_path = fullfile(script_dir, 'cleaned_dataset.csv');
writetable(cleanedData, cleaned_data_path);
disp(['    清洗后数据已保存至: ', cleaned_data_path]);

%% 绘制 zc.. 和 F1 原始数据与处理后数据对比图
N = height(data);
t = (1:N)' / 1e4;
f01 = figure(1);
f01.Position = [0, 0, 900, 500];
% 子图 1：zc.. 原始数据与处理后数据
subplot(2, 1, 1);
hold on;
grid on;
h1 = plot(t, data.('zc..'), 'Color', [0.0000 0.4470 0.7410 0.25], 'LineWidth', 1);       % 原始数据，25% 不透明
h2 = plot(t, table2array(zcDdot), 'Color', [0.0000 0.4470 0.7410 1.0], 'LineWidth', 1.2); % 处理后数据，100% 不透明
title('Car Body Vertical Acceleration zc.. (Raw vs Processed)', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Data Index (\times 10^4)', 'FontSize', 12);
ylabel('Acceleration', 'FontSize', 12);
legend([h1, h2], {'Raw', 'Smoothed + Normalized'}, 'Location', 'southeast', 'FontSize', 10);
xlim([0 230] / 1e4);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
% 子图 2：F1 原始数据与归一化后数据
subplot(2, 1, 2);
hold on;
grid on;
yyaxis left;
h3 = plot(t, data.('F1'), 'Color', [0.8500 0.3250 0.0980 0.25], 'LineWidth', 1);      % 原始数据，25% 不透明
ylabel('Vertical Force', 'FontSize', 12);
set(gca, 'YColor', 'k');
yyaxis right;
h4 = plot(t, table2array(F1), 'Color', [0.8500 0.3250 0.0980 1.0], 'LineWidth', 1.2); % 归一化后数据，100% 不透明
ylabel('Normalized Value', 'FontSize', 12);
set(gca, 'YColor', 'k');
title('Wheel-1 Vertical Force F1 (Raw vs Processed)', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Data Index (\times 10^4)', 'FontSize', 12);
legend([h3, h4], {'Raw', 'Normalized'}, 'Location', 'southeast', 'FontSize', 10);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
xlim([0 230] / 1e4);
% 保存图像
output_dir = fullfile(script_dir, 'outputpics');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
exportgraphics(gcf, fullfile(output_dir, 'zc_and_F1_comparison.png'), 'Resolution', 600);
disp(['    图像已保存至: ', fullfile(output_dir, 'zc_and_F1_comparison.png')]);
