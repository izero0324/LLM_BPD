from abc import abstractmethod
from typing import AsyncIterator, ClassVar, List

from text_generation import AsyncClient

from vision_chat_api.rest_api.data_models.code_models import LLMConversationMessage
from vision_chat_api.service.ai.base_llm import BaseLLM, rchop
from vision_chat_api.service.conf.conf_manager import conf_mgr
from vision_chat_api.service.models.llm import LLMType


class HFInferenceLLM(BaseLLM):

    url: ClassVar[str]
    eos_token: ClassVar[str]
    user_token: ClassVar[str]
    can_stream: bool = True

    @classmethod
    @abstractmethod
    def attr(cls):
        super().attr()
        if not hasattr(cls, "url"):
            raise AttributeError
        if not hasattr(cls, "eos_token"):
            raise AttributeError
        if not hasattr(cls, "user_token"):
            raise AttributeError

    @classmethod
    async def _ask(cls, prompt: str, **kwargs) -> str:
        client = AsyncClient(f"http://{cls.url}", timeout=30)
        response = (await client.generate(prompt=prompt, max_new_tokens=1024, do_sample=True, **kwargs)).generated_text
        return response

    @classmethod
    async def _ask_stream(cls, prompt: str, **kwargs) -> AsyncIterator[str]:
        client = AsyncClient(f"http://{cls.url}", timeout=30)
        async for token in client.generate_stream(prompt=prompt, max_new_tokens=1024, do_sample=True, **kwargs):
            if not token.token.special:
                yield token.token.text


class ArtemisLLM(HFInferenceLLM):
    # BaseLLM
    name = "Artemis 34B"
    description = "A CodeLlama-based large language model trained on a custom code optimisation dataset by TurinTech."
    type = LLMType.artemis_llm
    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "### System Prompt\n"
        "{% elif message['role'] == 'user' %}"
        "### User Message\n"
        "{% else %}"
        "### Assistant Response\n"
        "{% endif %}"
        "{{ message['content'] }}\n"
        "{% endfor %}"
        "### Assistant Response"
    )

    # HFInferenceLLM
    url = conf_mgr.artemis_conf.artemis_llm_url
    eos_token = ""
    user_token = ""

    @classmethod
    async def ask(cls, messages: List[LLMConversationMessage], *args, **kwargs) -> str:
        prompt = cls.prepare_prompt(messages)

        return await cls._ask(prompt)

    @classmethod
    async def ask_stream(cls, messages: List[LLMConversationMessage]) -> AsyncIterator[str]:
        prompt = cls.prepare_prompt(messages)
        async for token in cls._ask_stream(prompt):
            yield token

    @classmethod
    def is_available(cls) -> bool:
        return conf_mgr.artemis_conf.artemis_llm_url is not None


class Llama38BLLM(HFInferenceLLM):

    @classmethod
    def attr(cls):
        super().attr()

    # BaseLLM
    name = "LLama 3 8B"
    description = "LLama 3 is a new 8 billion parameter language model that represents a major advance in large language model (LLM) capabilities."
    type = LLMType.llama_3_8b
    chat_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}"
        "{% set content = '<|begin_of_text|>' + content %}"
        "{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    )

    # HFInferenceLLM
    url = conf_mgr.artemis_conf.llama_3_8b_url
    eos_token = ""
    user_token = ""

    @classmethod
    async def ask(cls, messages: List[LLMConversationMessage], *args, **kwargs) -> str:
        prompt = cls.prepare_prompt(messages)
        stop_sequences = ["<|end_of_text|>", "<|eot_id|>"]
        res = await cls._ask(prompt, stop_sequences=stop_sequences)
        for seq in stop_sequences:
            res = rchop(res, seq)
        return res

    @classmethod
    async def ask_stream(cls, messages: List[LLMConversationMessage]) -> AsyncIterator[str]:
        prompt = cls.prepare_prompt(messages)
        stop_sequences = ["<|end_of_text|>", "<|eot_id|>"]
        async for token in cls._ask_stream(prompt, stop_sequences=stop_sequences):
            if token not in stop_sequences:
                yield token

    @classmethod
    def is_available(cls):
        return conf_mgr.artemis_conf.llama_3_8b_url is not None


class NeuralChatLLM(HFInferenceLLM):

    @classmethod
    def attr(cls):
        super().attr()

    # BaseLLM
    name = "Intel Neural Chat"
    description = (
        "Intel Neural Chat model is a fine-tuned model for chat based on mosaicml/mpt-7b with a max sequence length of 2048 "
        "on the dataset Intel/neural-chat-dataset-v1-1, which is a compilation of open-source datasets."
    )
    type = LLMType.neural_chat
    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "### System:\n"
        "{% elif message['role'] == 'user' %}"
        "### User:\n"
        "{% else %}"
        "### Assistant:\n"
        "{% endif %}"
        "{{ message['content'] }}\n"
        "{% endfor %}"
        "### Assistant:"
    )

    # HFInferenceLLM
    url = conf_mgr.artemis_conf.neural_chat_url
    eos_token = ""
    user_token = ""

    @classmethod
    async def ask(cls, messages: List[LLMConversationMessage], *args, **kwargs) -> str:
        prompt = cls.prepare_prompt(messages)

        return await cls._ask(prompt)

    @classmethod
    async def ask_stream(cls, messages: List[LLMConversationMessage]) -> AsyncIterator[str]:
        prompt = cls.prepare_prompt(messages)
        async for token in cls._ask_stream(prompt):
            yield token

    @classmethod
    def is_available(cls):
        return conf_mgr.artemis_conf.neural_chat_url is not None
