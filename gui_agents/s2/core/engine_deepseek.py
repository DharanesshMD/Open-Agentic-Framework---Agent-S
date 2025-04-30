import os
import backoff
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from gui_agents.s2.core.engine import LMMEngine

class LMMEngineDeepSeek(LMMEngine):
    def __init__(self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs):
        assert model is not None, "model must be provided"
        self.model = model

        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named DEEPSEEK_API_KEY"
            )

        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

        self.llm_client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        """Generate the next message based on previous messages"""
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .choices[0]
            .message.content
        )