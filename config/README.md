# 本地连接配置

`env_config.ini` 是可以提交的空配置模板。真实地址、账号、密码、Key、存储桶和数据库信息放在 `env_config.local.ini` 中；该文件已被 Git 忽略。

首次配置可复制模板：

```powershell
Copy-Item config/env_config.ini config/env_config.local.ini
```

已有本地配置时不要覆盖。修改本地文件中对应环境的配置即可，读取顺序为：模板 → 本地配置 → 环境变量。

支持 `VLA_EXECUTION_ENV`（dev、fat、prod），以及配置键名前加 `VLA_` 并转大写的环境变量，例如 `VLA_API_URL`、`VLA_USERNAME`、`VLA_PASSWORD`、`VLA_S3_ENDPOINT`、`VLA_S3_ACCESS_KEY`、`VLA_S3_SECRET_KEY`、`VLA_S3_BUCKET`、`VLA_DB_HOST`、`VLA_DB_USER`、`VLA_DB_PASSWORD`、`VLA_DB_NAME` 和 `VLA_DB_PORT`。`token_url` 留空时从 `api_url` 推导登录地址。

上传时仅包含 Git 管理的脱敏文件。不要直接打包整个项目目录：本地配置、旧报告和 `.git` 历史可能包含真实连接信息。此次配置调整不会删除已有 Git 历史中的凭据。
