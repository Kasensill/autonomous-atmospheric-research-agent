"""仅供模型表达人工澄清意图的声明；没有可执行的研究工具实现。"""

ASK_USER_NAME = "ask_user"

ASK_USER_TOOL = {
    "type": "function",
    "function": {
        "name": ASK_USER_NAME,
        "description": "缺少只能由用户提供的关键条件时追问用户；不执行检索，不提供研究证据。一次提出必要问题。",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "minLength": 1, "description": "向用户提出的具体问题"}},
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}
