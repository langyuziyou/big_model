# -*- coding: utf-8 -*-
"""启动入口：python run.py"""
import os
import sys

# 确保以项目根目录为工作目录，保证 config / core / app 可被导入
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import uvicorn  # noqa: E402
import config  # noqa: E402

if __name__ == "__main__":
    print(f"启动服务: http://{config.HOST}:{config.PORT}")
    print(f"客户窗口: http://{config.HOST}:{config.PORT}/")
    print(f"客服工作台: http://{config.HOST}:{config.PORT}/agent")
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, log_level="info")
