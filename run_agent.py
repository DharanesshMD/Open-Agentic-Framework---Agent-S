import os
from dotenv import load_dotenv
import pyautogui
import io
import requests
import base64
from openai import OpenAI
# import pytesseract
import tempfile

# Configure Tesseract path if needed
# Uncomment and modify the path below if Tesseract is installed in a non-standard location
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from gui_agents.s2.agents.agent_s import AgentS2
from gui_agents.s2.agents.grounding import OSWorldACI

# Load environment variables from .env file
load_dotenv()

# Set your platform
current_platform = "windows"  # or "linux" or "darwin" for Mac

# Initialize OpenAI client for DeepSeek
deepseek_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",  # NVIDIA API endpoint
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# Initialize the grounding agent with Gemma
grounding_agent = OSWorldACI(
    platform=current_platform,
    engine_params_for_generation={
        "engine_type": "gemini",
        "model": "google/gemma-3-27b-it",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 4096,
        "api_endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",
        "api_key": os.getenv("GEMMA_API_KEY"),
        "base_url": "https://integrate.api.nvidia.com/v1/chat/completions"
    },
    engine_params_for_grounding={
        "engine_type": "gemini",
        "model": "google/gemma-3-27b-it",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 4096,
        "api_endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",
        "api_key": os.getenv("GEMMA_API_KEY"),
        "base_url": "https://integrate.api.nvidia.com/v1/chat/completions"
    },
    width=1920,
    height=1080
)

# Create a temporary directory for the knowledge base
temp_dir = tempfile.mkdtemp()

# Initialize the main agent with DeepSeek
main_agent = AgentS2(
    engine_params={
        "engine_type": "deepseek",
        "model": "deepseek-ai/deepseek-r1-distill-qwen-32b",
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 4096,
        "api_endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",  # Updated endpoint
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://integrate.api.nvidia.com/v1",  # Base URL for client
        "client": deepseek_client
    },
    grounding_agent=grounding_agent,
    platform=current_platform,
    action_space="pyautogui",
    observation_type="mixed",
    search_engine="LLM",
    memory_root_path=temp_dir,  # Use temporary directory
    memory_folder_name="kb_s2",  # Default folder name
    kb_release_tag=None  # Skip knowledge base download
)

# Now you can use the agents to perform tasks
if __name__ == "__main__":
    print("Agent S2 initialized with DeepSeek (main) and Gemma (grounding)")
    print("Enter your task when prompted...")
    
    while True:
        task = input("\nEnter your task (or 'quit' to exit): ")
        if task.lower() == 'quit':
            break
            
        try:
            # Get screenshot for observation
            screenshot = pyautogui.screenshot()
            buffered = io.BytesIO() 
            screenshot.save(buffered, format="PNG")
            screenshot_bytes = buffered.getvalue()
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

            obs = {
                "screenshot": screenshot_b64,
            }

            # Use the main agent to process the task
            info, actions = main_agent.predict(instruction=task, observation=obs)
            print(f"\nResponse: {info}")
            print(f"Actions: {actions}")
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")