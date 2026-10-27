"""无套接字的ASGI协议测试驱动；仅适用于不挂起的内存接口。

真实Uvicorn/事件循环另做启动验收，不把本驱动冒充TCP测试。
"""
import httpx
import socket
import pytest


def require_loopback():
    """预检本机回环，避免受限Windows的socketpair无限阻塞；不修改网络配置。"""
    with socket.socket() as listener, socket.socket() as connection:
        listener.settimeout(1)
        connection.settimeout(1)
        try:
            listener.bind(('127.0.0.1',0))
            listener.listen()
            connection.connect(listener.getsockname())
            peer,_=listener.accept()
            peer.close()
        except OSError:
            pytest.skip('本机回环TCP不可用，无法启动标准事件循环；需在可用环境重跑。')


def complete(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    finally:
        coroutine.close()
    raise RuntimeError('内存接口发生异步挂起；请用标准ASGI测试客户端验证，不能伪造调度。')


class MemoryASGITransport(httpx.BaseTransport):
    def __init__(self, app):
        self.app=app

    def handle_request(self, request):
        messages=[]
        async def receive():
            return {'type':'http.request','body':request.read(),'more_body':False}
        async def send(message):
            messages.append(message)
        scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},
               'http_version':'1.1','method':request.method,'scheme':'http',
               'path':request.url.path,'raw_path':request.url.raw_path.split(b'?')[0],
               'query_string':request.url.query,'root_path':'',
               'headers':[(k.lower(),v) for k,v in request.headers.raw],
               'server':('localhost',80),'client':('127.0.0.1',1234)}
        complete(self.app(scope,receive,send))
        start=next(m for m in messages if m['type']=='http.response.start')
        body=b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body')
        return httpx.Response(start['status'],headers=start['headers'],content=body)


class MemoryClient(httpx.Client):
    def __init__(self, app):
        self._lifespan=app.router.lifespan_context(app)
        super().__init__(base_url='http://localhost',transport=MemoryASGITransport(app),trust_env=False)

    def __enter__(self):
        complete(self._lifespan.__aenter__())
        return super().__enter__()

    def __exit__(self,*args):
        try:
            return super().__exit__(*args)
        finally:
            complete(self._lifespan.__aexit__(*args))
