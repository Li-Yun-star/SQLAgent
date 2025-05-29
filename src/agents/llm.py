from typing import Literal, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os
load_dotenv()

LLMType = Literal["openai", "deepseek"]

def create_llm(
    llm_type: LLMType,
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    streaming: bool = False,
    **kwargs
) -> Union[ChatOpenAI, ChatDeepSeek]:
    """
    通用 LLM 构造函数，根据类型返回对应模型实例
    :param llm_type: "openai" 或 "deepseek"
    :param model: 模型名称
    :param base_url: 接口地址（OpenAI: base_url，DeepSeek: api_base）
    :param api_key: 密钥
    :param temperature: 随机性
    :param streaming: 是否开启流式响应
    :return: 模型实例
    """
    llm_kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
        **kwargs,
    }

    if llm_type == "openai":
        if base_url:
            llm_kwargs["base_url"] = base_url
        llm_kwargs["streaming"] = streaming
        return ChatOpenAI(**llm_kwargs)

    elif llm_type == "deepseek":
        if base_url:
            llm_kwargs["api_base"] = base_url
        # DeepSeek 默认支持 stream 方法，不需要 streaming 参数
        return ChatDeepSeek(**llm_kwargs)

    else:
        raise ValueError(f"❌ 不支持的 LLM 类型: {llm_type}")

if __name__ == "__main__":
    # llm = create_llm(
    #     llm_type="deepseek",
    #     model=os.getenv("REASONING_MODEL"),
    #     base_url=os.getenv("REASONING_BASE_URL"),
    #     api_key=os.getenv("REASONING_API_KEY"),
    # )
    llm = create_llm(
        model=os.getenv("REASONING_MODEL", "deepseek-chat"),
        llm_type=os.getenv("REASONING_TYPE", "deepseek"),
        base_url=os.getenv("REASONING_BASE_URL"),
        api_key=os.getenv("REASONING_API_KEY"),
    )

    print(llm.invoke("Hello, world!"))  