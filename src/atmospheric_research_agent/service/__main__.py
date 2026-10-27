"""固定本地单 worker 启动；不要用 --reload 或多个 worker 承载内存任务。"""
import uvicorn


if __name__ == '__main__':
    uvicorn.run('atmospheric_research_agent.service.api:create_app',factory=True,
                host='127.0.0.1',port=8000,workers=1)
