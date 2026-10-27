"""兼容旧的执行类导入；网页执行已迁移到 HTTP 客户端。"""

from ..service.engine import ResearchSession

__all__ = ["ResearchSession"]
