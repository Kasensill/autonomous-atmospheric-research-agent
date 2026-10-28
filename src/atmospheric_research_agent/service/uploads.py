"""本地原材料保存，不做解析缓存或数据库；只接受文件名和已限制大小的字节。"""
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path

from ..agent.settings import (
    MATERIAL_MAX_FILES, MATERIAL_UPLOAD_MAX_BYTES, MATERIAL_UPLOAD_MAX_NAME_CHARACTERS,
    MATERIAL_UPLOAD_DIRECTORY,
)


@dataclass(frozen=True)
class MaterialUpload:
    name: str
    content: bytes = field(repr=False)


def validate_uploads(files: list[MaterialUpload], role: str) -> None:
    if role not in {"auto", "clue", "analysis", "both"}:
        raise ValueError("材料用途无效。")
    if not 1 <= len(files) <= MATERIAL_MAX_FILES:
        raise ValueError(f"每个任务需要 1 至 {MATERIAL_MAX_FILES} 份材料。")
    for item in files:
        name = item.name
        if (not isinstance(name, str) or not name or name != name.strip()
                or len(name) > MATERIAL_UPLOAD_MAX_NAME_CHARACTERS
                or any(ch in '<>:"/\\|?*' or ord(ch) < 32 or ord(ch) == 127 for ch in name)
                or name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                    *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}):
            raise ValueError("材料名称无效；请使用普通文件名，不要包含路径、控制字符或设备名。")
        if Path(name).suffix.lower() not in {'.txt', '.md', '.pdf'}:
            raise ValueError("仅支持 TXT、MD 和有文字层的 PDF 材料。")
        if not isinstance(item.content, bytes) or not item.content:
            raise ValueError("上传的材料文件不能为空。")
        if len(item.content) > MATERIAL_UPLOAD_MAX_BYTES:
            raise ValueError(f"单份材料不能超过 {MATERIAL_UPLOAD_MAX_BYTES // (1024 * 1024)}MB。")


def upload_fingerprint(files: list[MaterialUpload], role: str) -> str:
    """文件顺序会影响送入范围，因此名称、顺序、内容与角色共同参与幂等判断。"""
    return sha256(json.dumps({'role': role, 'files': [
        (item.name, sha256(item.content).hexdigest()) for item in files
    ]}, ensure_ascii=False).encode('utf-8')).hexdigest()


class UploadStore:
    def __init__(self, root: Path | None = None):
        self.root = root if root is not None else Path(__file__).resolve().parents[3] / MATERIAL_UPLOAD_DIRECTORY

    def save(self, run_id: str, files: list[MaterialUpload]) -> list[Path]:
        # run_id 只能由管理器生成；文件名已验证，既不解释材料正文，也不接受客户端目录。
        from uuid import UUID
        directory = self.root / str(UUID(run_id))
        paths = []
        for index, item in enumerate(files):
            target = directory / str(index) / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                output.write(item.content)
            paths.append(target)
        return paths
