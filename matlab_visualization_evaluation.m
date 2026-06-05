clc; clear; close all;

%% 加载评估结果数据
% 设置脚本路径、输出数据目录和图片保存目录
script_dir = fileparts(mfilename('fullpath'));
output_dir = fullfile(script_dir, 'outputdata');
pic_dir = fullfile(script_dir, 'outputpics');
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end
% 两套预处理方案：原有方案保持原文件名，对比方案使用 no_smoothing 后缀
preprocess_configs = struct( ...
    'name', {'Smoothed + Normalized', 'Normalized Only'}, ...
    'suffix', {'', '_no_smoothing'}, ...
    'loss_file', {'loss_history.csv', 'loss_history_no_smoothing.csv'}, ...
    'metrics_file', {'evaluation_metrics.csv', 'evaluation_metrics_no_smoothing.csv'}, ...
    'prediction_file', {'prediction_results.csv', 'prediction_results_no_smoothing.csv'} ...
    );
default_colors = colororder;
metric_names = {'mse', 'rmse', 'mae'};
metric_titles = {'MSE', 'RMSE', 'MAE'};
metric_targets = {'F1', 'F2', 'F3', 'F4', 'Overall'};
targets = {'F1', 'F2', 'F3', 'F4'};
subplot_positions = [
    0.070 0.565 0.420 0.365;
    0.560 0.565 0.420 0.365;
    0.070 0.085 0.420 0.365;
    0.560 0.085 0.420 0.365
    ];
metric_subplot_positions = [
    0.055 0.150 0.280 0.760;
    0.370 0.150 0.280 0.760;
    0.685 0.150 0.280 0.760
    ];

all_overall_metrics = table();

%% 分别绘制每套预处理方案的结果图
for config_idx = 1:numel(preprocess_configs)
    config = preprocess_configs(config_idx);
    % 加载训练损失、预测结果和评价指标
    loss_data = readtable(fullfile(output_dir, config.loss_file), 'VariableNamingRule', 'preserve');
    pred_data = readtable(fullfile(output_dir, config.prediction_file), 'VariableNamingRule', 'preserve');
    metrics_data = readtable(fullfile(output_dir, config.metrics_file), 'VariableNamingRule', 'preserve');
    models = string(unique(loss_data.model, 'stable'));
    % 确认加载结果
    disp(['    评估结果数据加载完成：', config.name]);
    disp(['    模型数量: ', num2str(numel(models))]);
    disp(['    损失记录数: ', num2str(height(loss_data))]);
    disp(['    预测样本数: ', num2str(height(pred_data))]);
    %% 绘制训练集与验证集损失曲线
    f01 = figure(10 * config_idx + 1);
    f01.Position = [0, 0, 900, 500];
    hold on;
    grid on;
    % 每个模型使用同一颜色，训练集为较粗实线，验证集为较细实线
    for i = 1:numel(models)
        model_name = models(i);
        idx = strcmp(string(loss_data.model), model_name);
        model_loss = loss_data(idx, :);
        c = default_colors(i, :);
        plot(model_loss.epoch, model_loss.train_loss, '-', 'Color', c, 'LineWidth', 1.4);
        plot(model_loss.epoch, model_loss.val_loss, '-', 'Color', c, 'LineWidth', 0.9);
    end
    title(['Training and Validation Loss (', config.name, ')'], 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('Epoch', 'FontSize', 12);
    ylabel('MSE Loss', 'FontSize', 12);
    % 构造图例
    legend_entries = {};
    for i = 1:numel(models)
        legend_entries{end + 1} = char(models(i) + " Train"); %#ok<SAGROW>
        legend_entries{end + 1} = char(models(i) + " Validation"); %#ok<SAGROW>
    end
    legend(legend_entries, 'Location', 'northeast', 'FontSize', 10);
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
    yl = ylim; ylim([0, yl(2)]);
    % 保存损失曲线
    loss_pic = fullfile(pic_dir, ['loss_curve', config.suffix, '.png']);
    exportgraphics(f01, loss_pic, 'Resolution', 600);

    %% 绘制评价指标对比图
    % 将 MSE、RMSE、MAE 以分组柱状图形式展示，便于比较不同模型性能
    metric_values = zeros(numel(metric_targets), numel(models), numel(metric_names));
    % 整理评价指标矩阵：行对应目标变量，列对应模型，页对应指标类型
    for metric_idx = 1:numel(metric_names)
        metric_name = metric_names{metric_idx};
        for target_idx = 1:numel(metric_targets)
            target_name = metric_targets{target_idx};
            for model_idx = 1:numel(models)
                model_name = models(model_idx);
                row_idx = strcmp(string(metrics_data.model), model_name) & strcmp(string(metrics_data.target), target_name);
                metric_values(target_idx, model_idx, metric_idx) = metrics_data.(metric_name)(row_idx);
            end
        end
    end
    f02 = figure(10 * config_idx + 2);
    f02.Position = [0, 0, 1800, 430];
    for metric_idx = 1:numel(metric_names)
        axes('Position', metric_subplot_positions(metric_idx, :));
        b = bar(metric_values(:, :, metric_idx), 'BarWidth', 0.86);
        for bar_idx = 1:numel(b)
            b(bar_idx).EdgeColor = 'none';
            x_text = b(bar_idx).XEndPoints;
            y_text = b(bar_idx).YEndPoints;
            labels = compose('%.4f', y_text);
            text(x_text, y_text, labels, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontName', 'Times New Roman', ...
                'FontSize', 7);
        end
        grid on;
        title(metric_titles{metric_idx}, 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Target', 'FontSize', 11);
        ylabel(metric_titles{metric_idx}, 'FontSize', 11);
        xticks(1:numel(metric_targets));
        xticklabels(metric_targets);
        legend(cellstr(models), 'Location', 'northwest', 'FontSize', 9);
        ylim([0, max(metric_values(:, :, metric_idx), [], 'all') * 1.28]);
        box off;
        set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);
    end
    % 保存评价指标对比图
    metrics_pic = fullfile(pic_dir, ['metrics_comparison', config.suffix, '.png']);
    exportgraphics(f02, metrics_pic, 'Resolution', 600);

    %% 绘制预测值与真实值对比图
    % 仅展示测试集前 1000 个样本，避免曲线过密
    plot_count = min(1000, height(pred_data));
    x = pred_data.sample_index(1:plot_count);
    % 分别为每个模型绘制四个车轮垂向力的真实值与预测值
    for model_idx = 1:numel(models)
        model_name = models(model_idx);
        f03 = figure(10 * config_idx + 2 + model_idx);
        f03.Position = [0, 0, 1000, 720];
        for i = 1:numel(targets)
            target = targets{i};
            axes('Position', subplot_positions(i, :));
            hold on;
            grid on;
            true_col = [target, '_true'];
            pred_col = char(model_name + "_" + target + "_pred");
            plot(x, pred_data.(true_col)(1:plot_count), '-', 'Color', [0.0000 0.4470 0.7410], 'LineWidth', 1.1);
            plot(x, pred_data.(pred_col)(1:plot_count), '-', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 0.85);
            title([target, ' Prediction (', char(model_name), ')'], 'FontSize', 13, 'FontWeight', 'bold');
            xlabel('Test Sample Index', 'FontSize', 10);
            ylabel('Normalized Force', 'FontSize', 10);
            legend({'True', 'Predicted'}, 'Location', 'northwest', 'FontSize', 9);
            ylim([-4, 4]);
            set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);
        end
        % 保存当前模型的预测对比图
        prediction_file = fullfile(pic_dir, ['prediction_comparison_', char(model_name), config.suffix, '.png']);
        exportgraphics(f03, prediction_file, 'Resolution', 600);
        disp(['    预测对比图已保存至: ', prediction_file]);
    end
    % 汇总 Overall 指标，用于比较不同预处理方法的影响
    overall_idx = strcmp(string(metrics_data.target), 'Overall');
    current_overall = metrics_data(overall_idx, :);
    current_overall.preprocessing = repmat(string(config.name), height(current_overall), 1);
    all_overall_metrics = [all_overall_metrics; current_overall]; %#ok<AGROW>
    disp(['    损失曲线已保存至: ', loss_pic]);
    disp(['    评价指标对比图已保存至: ', metrics_pic]);
end

%% 绘制不同预处理方法对整体指标的影响
% 使用 Overall 指标比较平滑+归一化与仅归一化两种预处理方案
models_for_preprocess = string(unique(all_overall_metrics.model, 'stable'));
preprocess_names = string(unique(all_overall_metrics.preprocessing, 'stable'));
preprocess_metric_values = zeros(numel(models_for_preprocess), numel(preprocess_names), numel(metric_names));
for metric_idx = 1:numel(metric_names)
    metric_name = metric_names{metric_idx};
    for model_idx = 1:numel(models_for_preprocess)
        model_name = models_for_preprocess(model_idx);
        for prep_idx = 1:numel(preprocess_names)
            prep_name = preprocess_names(prep_idx);
            row_idx = strcmp(string(all_overall_metrics.model), model_name) & strcmp(string(all_overall_metrics.preprocessing), prep_name);
            preprocess_metric_values(model_idx, prep_idx, metric_idx) = all_overall_metrics.(metric_name)(row_idx);
        end
    end
end
f04 = figure(100);
f04.Position = [0, 0, 1500, 430];
for metric_idx = 1:numel(metric_names)
    axes('Position', metric_subplot_positions(metric_idx, :));
    b = bar(preprocess_metric_values(:, :, metric_idx), 'BarWidth', 0.80);
    for bar_idx = 1:numel(b)
        b(bar_idx).EdgeColor = 'none';
        x_text = b(bar_idx).XEndPoints;
        y_text = b(bar_idx).YEndPoints;
        labels = compose('%.4f', y_text);
        text(x_text, y_text, labels, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontName', 'Times New Roman', ...
            'FontSize', 8);
    end
    grid on;
    title(['Overall ', metric_titles{metric_idx}], 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Model', 'FontSize', 11);
    ylabel(metric_titles{metric_idx}, 'FontSize', 11);
    xticks(1:numel(models_for_preprocess));
    xticklabels(cellstr(models_for_preprocess));
    legend(cellstr(preprocess_names), 'Location', 'northwest', 'FontSize', 9);
    ylim([0, max(preprocess_metric_values(:, :, metric_idx), [], 'all') * 1.28]);
    box off;
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);
end
preprocess_pic = fullfile(pic_dir, 'preprocessing_metrics_comparison.png');
exportgraphics(f04, preprocess_pic, 'Resolution', 600);
disp(['    预处理方法对比图已保存至: ', preprocess_pic]);
