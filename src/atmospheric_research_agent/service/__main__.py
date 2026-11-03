"""固定本地单 worker 启动；执行图仍在进程内，同一 SQLite 只允许一个服务实例。"""
import uvicorn


if __name__ == '__main__':
    uvicorn.run('atmospheric_research_agent.service.api:create_app',factory=True,
                host='127.0.0.1',port=8000,workers=1)
