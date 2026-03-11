"""
LLM client for Legal Drafting AI.
Supports Ollama (native /api/generate) and vLLM/OpenAI-compatible (/v1/chat/completions).
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


def _is_ollama_base_url(base_url: str) -> bool:
    """True if base_url points at Ollama (e.g. localhost:11434)."""
    u = base_url.rstrip("/").lower()
    return ":11434" in u or "ollama" in u


class LegalLLMClient:
    """
    Client for local LLM inference.
    - Ollama: uses native POST /api/generate (prompt + model, response in data["response"]).
    - vLLM/OpenAI: uses POST /v1/chat/completions with messages.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str = "meta-llama/Llama-3-8B-Instruct",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        timeout: float = 300.0,
    ):
        base_url = base_url.rstrip("/")
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._use_ollama = _is_ollama_base_url(base_url)
        if self._use_ollama:
            # Ollama root URL (no /v1)
            if self.base_url.endswith("/v1"):
                self.base_url = self.base_url[:-3]
            self._ollama_chat_url = f"{self.base_url}/api/chat"
            self._ollama_generate_url = f"{self.base_url}/api/generate"

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Send prompt to LLM and return generated text.
        For Ollama: uses /api/generate with a single combined prompt.
        For vLLM/OpenAI: uses /chat/completions with messages.
        """
        if self._use_ollama:
            return self._complete_ollama(prompt, system_prompt, max_tokens, temperature, stop)
        return self._complete_openai(prompt, system_prompt, max_tokens, temperature, stop)

    def _complete_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        stop: Optional[list[str]],
    ) -> str:
        """Ollama: try POST /api/chat first; on 404 try POST /api/generate (single prompt)."""
        opts = {
            "num_predict": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if stop:
            opts["stop"] = stop

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 1) Try /api/chat (messages format)
        chat_payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": opts,
        }
        last_error = None
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self._ollama_chat_url, json=chat_payload)
            if resp.status_code == 200:
                data = resp.json()
                msg = data.get("message") or {}
                text = msg.get("content", "") if isinstance(msg, dict) else ""
                return (text if isinstance(text, str) else str(text)).strip()
            if resp.status_code != 404:
                resp.raise_for_status()
            last_error = f"404 for {self._ollama_chat_url}"

            # 2) Fallback: /api/generate (single prompt)
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            gen_payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": opts,
            }
            resp2 = client.post(self._ollama_generate_url, json=gen_payload)
            if resp2.status_code == 200:
                data = resp2.json()
                text = data.get("response", "")
                return (text if isinstance(text, str) else str(text)).strip()
            if resp2.status_code != 404:
                resp2.raise_for_status()
            last_error = f"404 for both {self._ollama_chat_url} and {self._ollama_generate_url}"

        raise RuntimeError(
            f"LLM inference failed: {last_error}. "
            "From this app, 'localhost' is the machine running the app (e.g. Docker/WSL use their own localhost). "
            "Ensure Ollama is reachable at that URL, run 'ollama list', and set LEGAL_AI_LLM_MODEL_NAME to a pulled model (e.g. llama3.2)."
        )

    def _complete_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        stop: Optional[list[str]],
    ) -> str:
        """OpenAI-compatible API: POST /v1/chat/completions (vLLM, etc.)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/chat/completions"
        if not self.base_url.endswith("/v1") and "/v1" not in self.base_url:
            url = f"{self.base_url.rstrip('/')}/v1/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            logger.exception("LLM request failed: %s", e)
            raise RuntimeError(f"LLM inference failed: {e}") from e

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("LLM returned no choices")
        return choices[0].get("message", {}).get("content", "").strip()

    def complete_legacy(self, prompt: str, **kwargs) -> str:
        """Legacy single-prompt completion; uses Ollama /api/generate or OpenAI /completions."""
        if self._use_ollama:
            return self._complete_ollama(prompt, None, kwargs.get("max_tokens"), kwargs.get("temperature"), None)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False,
        }
        url = f"{self.base_url}/completions" if "/v1" in self.base_url else f"{self.base_url.rstrip('/')}/v1/completions"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("LLM returned no choices")
        return choices[0].get("text", "").strip()
