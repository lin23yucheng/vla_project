# VLA_Project

VLA_Project 是一套基于 Python、pytest 和 Allure 的自动化测试项目，用于验证机器人数据平台的业务接口、工作台统计和 LeRobot 数据转换结果。

## 测试范围

| 套件 | 验证内容 |
| --- | --- |
| `TestWorkbenchData` | 将工作台 API 返回的任务状态、采集与标注时长、Episode 数量与时长、人效统计等指标与 PostgreSQL 查询结果核对。 |
| `TestBusinessApiWorkflow` | 创建并查询任务、上传采集文件、轮询上传状态、提交采集完成，随后提交自动标注并检查处理结果。 |
| 夹爪 V2 | 覆盖 L1/L2 标注、质检和 `lerobot_v2` 转换，校验各标注片段的相机图像、机器人数据向量和标注信息。 |
| 摇操 V3 | 覆盖 L1 标注、质检和 `lerobot_v3` 转换，校验固定帧率时间轴、相机画面、机器人数据向量和标注信息。 |
| 天机 V2 | 覆盖标注、质检和 `lerobot_v2` 转换，校验 Episode 映射、固定帧率时间轴、相机图像、机器人数据向量和标注信息。 |

格式校验从 S3 兼容对象存储读取原始 MCAP 和转换产物，按时间戳定位对应帧，并根据 `Robot_Configuration/` 中的相机 topic、主时间轴和字段配置核对数据。已完成转换的任务可以复用转换产物。

## 本地安装

在项目根目录使用 Windows PowerShell 执行以下命令。项目已在 Python 3.10 环境验证。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q unit_testcase
```

单元测试无需连接业务 API、PostgreSQL 或对象存储，覆盖 MCAP 和 Parquet 解析、时间轴验证、图像比对、标注解析和配置加载等逻辑。生成 HTML 报告需要另外安装 Allure CLI，并确认 `allure --version` 可用。

## 集成测试配置

首次配置时，从仓库模板创建本地文件。已有本地文件时直接编辑对应环境段。

```powershell
Copy-Item config/env_config.ini config/env_config.local.ini
```

模板默认使用 `fat`。可通过 `[environment]` 下的 `execution_env` 或环境变量 `VLA_EXECUTION_ENV` 选择 `dev`、`fat` 或 `prod`，并在相应的 `[dev-vla]`、`[fat-vla]` 或 `[prod-vla]` 中填写连接信息。配置加载顺序为模板、本地文件、受支持的环境变量，后者覆盖前者。本地配置文件已被 Git 忽略。 [配置说明](config/README.md)

| 测试目标 | 所需配置 |
| --- | --- |
| 所有集成测试 | `api_url`、`username`、`password`。`token_url` 可留空，由 `api_url` 推导。 |
| 工作台统计 | `db_host`、`db_port`、`db_user`、`db_password`、`db_name`。 |
| 三套格式校验 | `s3_endpoint`、`s3_access_key`、`s3_secret_key`、`s3_bucket`，以及对应的 `gripper_task_no`、`shake_task_no` 或 `tianji_task_no`。 |
| 业务 API 流程 | 创建新任务时，用 `VLA_ROBOT_CONFIG_NAME` 和 `VLA_COLLECTION_PATH` 指定机器人构型与本地采集文件或目录，也可按运行提示输入。设置 `VLA_TASK_NO` 可直接使用已有任务并跳过创建与采集上传步骤。 |

## 运行集成测试

`run.py` 默认运行工作台统计校验。也可以通过 pytest 文件路径或类的节点 ID 指定套件。以下命令使用项目虚拟环境中的 Python。

```powershell
# 工作台统计
.\.venv\Scripts\python.exe run.py

# 任务创建、采集上传与自动标注
.\.venv\Scripts\python.exe run.py testcase/test_business_api_workflow.py::TestBusinessApiWorkflow

# 按构型验证转换数据
.\.venv\Scripts\python.exe run.py testcase/test_gripper_v2_verify.py
.\.venv\Scripts\python.exe run.py testcase/test_shake_v3_verify.py
.\.venv\Scripts\python.exe run.py testcase/test_tianji_v2_verify.py
```

业务 API 流程和格式校验可能改变目标环境中的任务状态，例如上传采集文件、提交标注、完成质检或发起转换。运行时请选择对应的测试环境和测试任务。这些用例共享前序步骤产生的任务和数据，应按套件顺序执行。

## 日志与报告

| 路径 | 内容 |
| --- | --- |
| `Log/` | 运行日志和错误日志。 |
| `report/allure-results/` | pytest 生成的 Allure 原始结果与附件。 |
| `report/allure-report/` | Allure CLI 生成的 HTML 报告。 |
| `mcap_image/`、`parquet_image/` | 格式校验提取的相机帧。 |
| `image_compare/` | 图像校验生成的差异图。 |

`run.py` 每次启动测试时会清理历史日志和 Allure 原始结果，并在测试结束后尝试生成 HTML 报告。使用 `allure open report/allure-report` 在浏览器中查看报告。

## 数据检查工具

```powershell
# 预览本地 Parquet 的字段和数据行
.\.venv\Scripts\python.exe -m common.parse_parquet_file --file 'path/to/data.parquet' --rows 5

# 按提示输入 MCAP 文件、图像 topic 和纳秒时间戳，提取相机帧
.\.venv\Scripts\python.exe -m common.extract_mcap_image
```

Parquet 解析结果写入 `parquet_file/parquet_analysis.txt`。省略 `--file` 时，工具默认读取 `parquet_file/` 中最新的 Parquet 文件。

## 原始 MCAP 稳定性审计

将多个 episode 文件夹放在同一目录下，每个 episode 保留 `meta.json`、`rosbag/metadata.yaml` 和单个 MCAP 文件。可以通过独立命令指定本地目录：

```powershell
.\.venv\Scripts\python.exe -m common.mcap_stability --input-dir 'C:\path\to\episodes'
```

审计阈值位于 `config/mcap_audit_config.json`，与 `Robot_Configuration/robot_config.yaml` 分开维护。默认以 30 FPS、正负 5 FPS、60 秒统计窗口和 40 ms 最大帧间隔检查四路相机。帧间隔推算的缺帧数是估算值，MCAP 的 `CompressedImage` 消息没有序列号时不能精确确认丢帧。跨相机时间差和 `header.stamp` 到 `log_time` 的差值默认只作诊断，只有确认时钟条件后才可配置阈值判定。

结果默认写入项目 `report/mcap_audit/summary.csv`、`window_summary.csv` 和 `anomalies.csv`，不会写入 MCAP 输入目录。也可用 `--output-dir` 指定报告目录，或用 `--audit-config` 指定另一份审计配置。pytest 入口仍可通过 `VLA_MCAP_ROOT` 和 `VLA_MCAP_REPORT_DIR` 指定输入及报告目录。

## 仓库结构

```text
api/                  登录和业务 API 封装
common/               配置、HTTP 客户端、对象存储访问和数据校验工具
config/               环境配置模板和本地配置说明
Robot_Configuration/  夹爪手持、遥操机械臂和天机的构型定义
testcase/             业务流程、工作台统计和格式校验集成测试
unit_testcase/        本地单元测试
run.py                测试执行与 Allure 报告入口
requirements.txt      Python 依赖
```

## 许可证

项目版权及使用条款见 [LICENSE](LICENSE)。第三方依赖的声明及许可证见 [NOTICE](NOTICE) 和 [LICENSES/](LICENSES/)。
