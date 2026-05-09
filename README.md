# Tab-Transformer NIDS

本仓库实现基于 Tab-Transformer 的网络流量异常检测系统，覆盖 CICIDS2017 数据处理、特征工程、模型训练、实验评估、模型打包、后端推理服务和前端可视化演示。

## 仓库功能

- 数据预处理：读取 CICIDS2017 `MachineLearningCSV`，执行字段规范化、标签整理、无效值处理、端口分桶、语义去重和二分类目标生成。
- 行为增强特征：从原始流量统计字段派生包数量、字节量、方向比例、包长分布、IAT 变异系数、TCP 标志组合、窗口状态和载荷画像等特征。
- 数据划分：提供随机划分和 `seen_family_no_leak` 划分策略，按来源文件与原始攻击家族组织样本，降低同源样本跨集合泄漏风险。
- 模型训练：实现 Tab-Transformer 主模型，并提供 MLP、Embedding-MLP、LSTM、Isolation Forest 对比模型。
- 实验评估：输出 F1、Precision、Recall、ROC-AUC、PR-AUC、混淆矩阵、ROC 曲线、阈值选择结果和攻击家族级指标。
- 消融实验：支持去除行为增强特征的 Tab-Transformer 训练，以及基于已打包主模型的特征遮蔽消融分析。
- 模型打包：将模型权重、预处理器、特征 schema、阈值和指标文件整理为演示系统可加载的 artifact 目录。
- 后端服务：使用 Django + Channels 提供健康检查、模型信息、单条推理、批量推理、样本回放、历史事件查询和 WebSocket 告警推送。
- 前端演示：使用 Vue 3 + ECharts 展示 API 状态、回放控制、实时趋势、攻击比例、端口分布、告警表格和流量详情。

## 代码说明

- `src/tab_transformer_nids/preprocessing.py`：核心数据处理模块，包含 `normalize_column_name()`、`augment_behavioral_features()`、`semantic_deduplicate_frame()`、`split_frame()` 和 `TabularPreprocessor`。该模块负责原始 CSV 到可训练 DataFrame 的转换，以及类别特征编码和连续特征标准化。
- `src/tab_transformer_nids/evaluation.py`：评估与阈值选择模块，包含 `choose_threshold_with_plateau()`、`classification_metrics()`、`build_attack_family_metrics()`、`save_roc_curve()` 和 `save_confusion_matrix()`。训练脚本通过它生成指标文件和图表。
- `src/tab_transformer_nids/inference.py`：推理封装模块，`InferenceBundle` 将预处理器、特征 schema、阈值、模型和指标组合成统一推理入口。
- `src/tab_transformer_nids/contracts.py`：前后端共享的数据契约，使用 Pydantic 定义推理响应和告警事件结构。
- `src/tab_transformer_nids/artifacts.py`：artifact 打包辅助模块，负责 JSON 写入和模型目录复制。
- `ml/models/tab_transformer_model.py`：Tab-Transformer 构建与 checkpoint 加载逻辑，`build_tab_transformer()` 根据配置生成模型，`load_tab_transformer_checkpoint()` 负责推理侧恢复。
- `ml/models/mlp_baseline.py`、`ml/models/lstm_baseline.py`、`ml/models/isolation_forest_baseline.py`：对比模型实现，用于论文实验中的基线比较。
- `ml/scripts/build_dataset.py`：数据集构建入口，读取配置文件，扫描原始 CSV，生成 `train.pkl.gz`、`val.pkl.gz`、`test.pkl.gz`、`preprocessor.joblib` 和 `feature_schema.json`。
- `ml/scripts/train.py`：训练入口，包含数组准备、batch size 探测、温度缩放、阈值选择、主模型训练、对比模型训练和消融实验逻辑。
- `ml/scripts/package_model.py`：模型打包入口，将训练输出整理到演示系统读取目录。
- `apps/backend/api/views.py`：REST API 视图，提供 `/api/v1/health`、`/api/v1/model/info`、`/api/v1/infer`、`/api/v1/infer/batch`、`/api/v1/replay/start`、`/api/v1/replay/stop`、`/api/v1/replay/status` 和 `/api/v1/replay/events`。
- `apps/backend/streaming/runtime.py`：后端运行时，包含 `ModelRuntime`、`ReplayController` 和 `EventBuffer`，负责加载模型、执行推理、管理回放任务和缓存事件。
- `apps/backend/streaming/consumers.py`：WebSocket 消费者，`AlertsConsumer` 负责 `/ws/alerts` 的状态消息、心跳响应和实时告警推送。
- `apps/frontend/src/App.vue`：前端主界面，负责状态卡片、回放控制、告警列表、流详情和图表数据聚合。
- `apps/frontend/src/api.js`：REST API 调用封装。
- `apps/frontend/src/ws.js`：WebSocket 连接封装。
- `apps/frontend/src/composables/useCharts.js`：ECharts 初始化与更新辅助函数。
- `configs/train.paper.yaml`：论文主实验配置。
- `configs/train.smoke.yaml`：快速链路验证配置。
- `scripts/prepare_smoke_demo.ps1`：准备 smoke 演示数据、训练 artifact 并完成打包。
- `scripts/start_backend.ps1`：启动 Django + Channels 后端服务。
- `scripts/start_frontend.ps1`：启动 Vue + Vite 前端服务。
- `scripts/run_demo.ps1`：按顺序启动演示链路。

## 数据划分说明

`seen_family_no_leak` 是面向论文实验的 seen-family 划分策略：实现上按 `source_file + label_original` 组织样本组，并在组内按 `row_id` 的连续片段顺序切分训练集、验证集和测试集。同一攻击家族可以同时出现在训练集、验证集和测试集中，但相邻连续片段不会跨集合重叠；该策略不是严格的 family-holdout。

## 使用流程

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]

cd apps\frontend
npm install
cd ..\..
```

```powershell
python -m ml.scripts.build_dataset --config configs/train.paper.yaml

python -m ml.scripts.train --config configs/train.paper.yaml --model tab_transformer
python -m ml.scripts.train --config configs/train.paper.yaml --model mlp
python -m ml.scripts.train --config configs/train.paper.yaml --model embedding_mlp
python -m ml.scripts.train --config configs/train.paper.yaml --model lstm
python -m ml.scripts.train --config configs/train.paper.yaml --model isolation_forest

python -m ml.scripts.train --config configs/train.paper.yaml --model tab_transformer_no_engineered
python -m ml.scripts.package_model --config configs/train.paper.yaml --model tab_transformer
python -m ml.scripts.train --config configs/train.paper.yaml --model feature_mask_ablation
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_smoke_demo.ps1
powershell -ExecutionPolicy Bypass -File scripts\start_backend.ps1
powershell -ExecutionPolicy Bypass -File scripts\start_frontend.ps1
```

后端默认地址为 `http://127.0.0.1:8000`，前端默认地址为 `http://127.0.0.1:5173`。

## 测试

```powershell
python -m pytest
```
