We support the following APIs for MLLM inference: NVIDIA NIM (Llama & Vila), OpenAI, Anthropic, Gemini, Azure OpenAI, vLLM for local models, and Open Router. To use these APIs, you need to set the corresponding environment variables:

1. NVIDIA NIM (Default Provider)

For the Llama-3.3 model (used for reasoning/planning):
```
export LLAMA_API_KEY=<YOUR_LLAMA_API_KEY>
```

For the Vila model (used for multimodal grounding):
```
export VILA_API_KEY=<YOUR_VILA_API_KEY>
```

2. OpenAI

```
export OPENAI_API_KEY=<YOUR_API_KEY>
```

3. Anthropic

```
export ANTHROPIC_API_KEY=<YOUR_API_KEY>
```

4. Gemini

```
export GEMINI_API_KEY=<YOUR_API_KEY>
export GEMINI_ENDPOINT_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
```

5. OpenAI on Azure

```
export AZURE_OPENAI_API_BASE=<DEPLOYMENT_NAME>
export AZURE_OPENAI_API_KEY=<YOUR_API_KEY>
```

6. vLLM for Local Models

```
export vLLM_ENDPOINT_URL=<YOUR_DEPLOYMENT_URL>
```

7. Open Router

```
export OPENROUTER_API_KEY=<YOUR_API_KEY>
export OPEN_ROUTER_ENDPOINT_URL="https://openrouter.ai/api/v1"
```

Alternatively you can directly pass the API keys into the engine_params argument while instantiating the agent.

```python
from gui_agents.s2.agents.agent_s import AgentS2

# Example using NVIDIA models (default)
engine_params = {
    "engine_type": 'nvidia',  # Using NVIDIA as provider
    "model": 'nvidia/llama-3.3-nemotron-super-49b-v1',  # Main reasoning model
}

# Configure grounding with NVIDIA Vila
engine_params_for_grounding = {
    "engine_type": 'nvidia',
    "model": 'nvidia/vila',  # Multimodal grounding model
}

agent = AgentS2(
    engine_params,
    grounding_agent,
    platform=current_platform,
    action_space="pyautogui",
    observation_type="mixed",
    search_engine="LLM"
)
```

Allowed Values for engine_type:
- 'nvidia' (default) - Uses NVIDIA NIM models (Llama for reasoning, Vila for vision)
- 'openai' - Uses OpenAI models
- 'anthropic' - Uses Anthropic models
- 'gemini' - Uses Google's Gemini models
- 'azure' - Uses Azure OpenAI models
- 'vllm' - Uses locally hosted models via vLLM
- 'open_router' - Uses Open Router API
- 'huggingface' - Uses Hugging Face models

To use the underlying Multimodal Agent (LMMAgent) which wraps LLMs with message handling functionality, you can use the following code snippet:

```python
from gui_agents.core.mllm import LMMAgent

# Example with NVIDIA models
engine_params = {
    "engine_type": 'nvidia',
    "model": 'nvidia/llama-3.3-nemotron-super-49b-v1',
}
agent = LMMAgent(
    engine_params=engine_params,
)

# Example with other providers
engine_params = {
    "engine_type": 'anthropic',
    "model": 'claude-3-5-sonnet-20240620',
}
agent = LMMAgent(
    engine_params=engine_params,
)
```

The `AgentS2` also utilizes this `LMMAgent` internally.

Note: The NVIDIA models have specific configuration recommendations:
- Llama-3.3: Uses temperature=0.6 and top_p=0.95 by default for optimal reasoning
- Vila: Automatically formats image inputs to match the required HTML tag format
