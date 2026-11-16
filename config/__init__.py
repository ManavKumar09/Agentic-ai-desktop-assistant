"""
config.py - Centralized Configuration System for AI Assistant.
This module acts as the Single Source of Truth for environment variables,
API keys, file system paths, model names, and system hardware settings.
Complies with 12-Factor App principles and supports .env and JSON fallback.
"""

import os
import sys
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Attempt to load python-dotenv if installed
try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


# 1. Base Directory Detection (PyInstaller & Normal Execution Compatible) 
def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent

BASE_DIR = _get_base_dir()

# Load environment variables from .env file immediately

if _DOTENV_AVAILABLE:
    env_file = BASE_DIR / ".env"
    if env_file.exists():
       load_dotenv(dotenv_path=env_file, override=False) 

# 2. Settings Datacass
@dataclass
class Settings:

    # --- Project Paths ---
    base_dir: Path = BASE_DIR
    config_dir: Path = BASE_DIR / "config"
    memory_dir: Path = BASE_DIR / "memory"
    core_dir: Path = BASE_DIR / "core"

    # --- API Keys ---
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "").strip())

    # --- OS & Platform ---
    os_system: str = field(default_factory=lambda: os.getenv("OS_SYSTEM", platform.system()).lower())

    # --- Model Configurations ---
    live_model: str = "models/gemini-2.5-flash-native-audio-preview-12-2025"
    fast_model: str = "gemini-3.8-flash"
    lite_model: str = "gemini-2.5-flash-lite"
    openrouter_default_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    # --- Audio Streaming Constants ---
    send_sample_rate: int = 16000      # 16kHz for Gemini input
    receive_sample_rate: int = 24000   # 24kHz for Gemini output
    channels: int = 2                  # Stereo capture
    chunk_size: int = 1024

        # --- System Hardware Settings ---
    camera_index: int = field(default_factory=lambda: int(os.getenv("CAMERA_INDEX", "-1")))

    # --- Platform Helpers ---
    @property
    def is_windows(self) -> bool:
        return "windows" in self.os_system or self.os_system == "win32"

    @property
    def is_mac(self) -> bool:
        return "darwin" in self.os_system or "mac" in self.os_system

    @property
    def is_linux(self) -> bool:
        return "linux" in self.os_system

    def validate_keys(self) -> tuple[bool, list[str]]:
        """Validates that necessary API keys are present."""
        missing = []
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        return len(missing) == 0, missing
    
# 3. Global Singleton Instance & Accessors
_setting_instance: Optional[Settings] = None

def get_settings() -> Settings:
    """Access the global cached Settings singleton."""
    global _setting_instance 
    if _setting_instance is None:
        _setting_instance = Settings()
    return _setting_instance

# Export global instance for direct import: `from config import settings`
settings = get_settings()

# 4. Backward-Compatibility Functions (Preserves existing module imports)
def get_config() -> dict:
    """Legacy helper: returns a dict of config keys."""
    return {
        "gemini_api_key": settings.gemini_api_key,
        "openrouter_api_key": settings.openrouter_api_key,
        "os_system": settings.os_system,
    }

def get_os() -> str:
    """Legacy helper: returns 'windows' | 'mac' | 'linux'."""
    return settings.os_system

def is_windows() -> bool:
    return settings.is_windows

def is_mac() -> bool:
    return settings.is_mac

def is_linux() -> bool:
    return settings.is_linux

def get_gemini_api_key() -> str:
    """Direct accessor for Gemini API key."""
    return settings.gemini_api_key

def get_openrouter_api_key() -> str:
    """Direct accessor for OpenRouter API key."""
    return settings.openrouter_api_key