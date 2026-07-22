"""
VLA登录环境封装
"""
import time
import requests
import configparser
from pathlib import Path

# 读取配置
config = configparser.ConfigParser()
config_path = Path(__file__).resolve().parent.parent / "config" / "env_config.ini"
config.read(config_path, encoding="utf-8")

env = config.get("environment", "execution_env", fallback="").strip().lower()
if env not in {"dev", "fat", "prod"}:
    raise ValueError(f"execution_env 配置错误: {env}，仅支持 dev、fat 或 prod")

section = f"{env}-vla"
if not config.has_section(section):
    raise ValueError(f"配置文件缺少节: [{section}]")

env_login_map = {
    "prod": {
        "token_url": "",
        "url": "",
    },
    "fat": {
        "token_url": "https://fat-vla-gw.svfactory.com:6143/api/v1/auth/login",
        "url": "https://fat-vla-gw.svfactory.com:6143",
    },
    "dev": {
        "token_url": "https://dev-vla-gw.svfactory.com:6143/api/v1/auth/login",
        "url": "https://dev-vla-gw.svfactory.com:6143",
    },
}

token_url = env_login_map[env]["token_url"]
username = "admin"
password = "123456"
url = env_login_map[env]["url"]


class ApiLogin:
    def __init__(self):
        pass

    def login(self, max_retries=5, retry_delay=3, timeout=60):
        login_data = {"username": username, "password": password}
        login_header = {"content-type": "application/json"}

        for attempt in range(max_retries):
            try:
                login_rep = requests.post(url=token_url, json=login_data, headers=login_header, timeout=timeout)

                # 检查响应状态
                if login_rep.status_code == 200:
                    response_json = login_rep.json()

                    # token 信息位于 data 字段中
                    response_data = response_json.get("data", {})
                    if "token_type" in response_data and "access_token" in response_data:
                        token_type = response_data["token_type"]
                        access_token = response_data["access_token"]
                        token = token_type + " " + access_token
                        print(f"登录成功，账号为: {username}")
                        return token
                    else:
                        print(f"第{attempt + 1}次尝试：响应缺少必要字段")
                else:
                    print(f"第{attempt + 1}次尝试：HTTP状态码 {login_rep.status_code}")

            except Exception as e:
                print(f"第{attempt + 1}次尝试失败：{str(e)}")

            # 如果不是最后一次尝试，则等待后重试
            if attempt < max_retries - 1:
                print(f"等待{retry_delay}秒后进行第{attempt + 2}次重试...")
                time.sleep(retry_delay)

        # 所有重试都失败
        raise Exception(f"登录失败，已重试{max_retries}次")


if __name__ == '__main__':
    m = ApiLogin()
    m.login()
