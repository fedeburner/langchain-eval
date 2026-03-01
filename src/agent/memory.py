"""Memory strategy implementations used by CLI and harness."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages


class MemoryStrategy(Protocol):
    """Protocol for memory strategies."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def add_user_message(self, text: str) -> None: ...

    def get_messages(self) -> list[Any]: ...

    def add_agent_response(self, result_messages: list[Any]) -> None: ...

    def start_new_session(self) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...


def _trim_to_window(messages: list[Any], exchanges: int) -> list[Any]:
    max_messages = max(exchanges * 2, 1)
    if len(messages) <= max_messages:
        return list(messages)
    return trim_messages(
        messages,
        max_tokens=max_messages,
        token_counter=lambda msgs, **_kw: len(msgs),
        strategy="last",
        start_on="human",
    )


def _new_profile() -> dict[str, Any]:
    return {
        "name": None,
        "profession": None,
        "location": None,
        "dog_name": None,
        "dog_breed": None,
        "favorite_language": None,
        "learning_languages": [],
        "preferences": [],
        "updates": [],
    }


class FullBufferMemory:
    """Keep and send the full message buffer each turn."""

    name = "Full Buffer"
    description = "Keeps all messages. Best recall, highest token growth."

    def __init__(self):
        self._messages: list[Any] = []

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def get_messages(self) -> list[Any]:
        return list(self._messages)

    def add_agent_response(self, result_messages: list[Any]) -> None:
        self._messages = list(result_messages)

    def start_new_session(self) -> None:
        self._messages = []

    def snapshot(self) -> dict[str, Any]:
        return {"strategy": "full", "stored_message_count": len(self._messages)}


class WindowMemory:
    """Keep only the last K user/assistant exchanges."""

    name = "Window"
    description: str

    def __init__(self, k: int = 6):
        self.k = k
        self.description = f"Keeps last {k} exchanges. Cheap but forgetful."
        self._messages: list[Any] = []

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def get_messages(self) -> list[Any]:
        return _trim_to_window(_to_lc_messages(self._messages), self.k)

    def add_agent_response(self, result_messages: list[Any]) -> None:
        self._messages = list(result_messages)

    def start_new_session(self) -> None:
        self._messages = []

    def snapshot(self) -> dict[str, Any]:
        return {
            "strategy": "window",
            "window_exchanges": self.k,
            "stored_message_count": len(self._messages),
        }


class SummaryMemory:
    """Compress old context into a running summary beyond a token threshold."""

    name = "Summary"
    description: str

    def __init__(
        self,
        model_str: str = "anthropic:claude-haiku-4-5-20251001",
        max_tokens: int = 4000,
        keep_recent: int = 4,
    ):
        self.model_str = model_str
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self._summary_model = None
        self._running_summary: str | None = None
        self._last_summarized_count = 0
        self._messages: list[Any] = []
        self.description = (
            f"Summarizes when >{max_tokens} tokens. Keeps last {keep_recent} exchanges."
        )

    @property
    def summary_model(self):
        if self._summary_model is None:
            self._summary_model = init_chat_model(self.model_str)
        return self._summary_model

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def get_messages(self) -> list[Any]:
        lc_messages = _to_lc_messages(self._messages)
        total_tokens = count_tokens_approximately(lc_messages)
        if total_tokens <= self.max_tokens:
            return lc_messages

        keep_count = max(self.keep_recent * 2, 1)
        old_cutoff = len(lc_messages) - keep_count
        if old_cutoff <= 0:
            return lc_messages

        old_messages = lc_messages[:old_cutoff]
        new_segment = old_messages[self._last_summarized_count :]
        if new_segment:
            self._running_summary = self._summarize(new_segment, self._running_summary)
            self._last_summarized_count = old_cutoff

        if not self._running_summary:
            return lc_messages

        recent_messages = lc_messages[old_cutoff:]
        summary_msg = SystemMessage(
            content=(
                "Conversation summary memory:\n"
                f"{self._running_summary}\n"
                "Prefer this summary for older facts when needed."
            )
        )
        return [summary_msg, *recent_messages]

    def add_agent_response(self, result_messages: list[Any]) -> None:
        self._messages = list(result_messages)

    def start_new_session(self) -> None:
        self._messages = []
        self._running_summary = None
        self._last_summarized_count = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "strategy": "summary",
            "max_tokens": self.max_tokens,
            "keep_recent_exchanges": self.keep_recent,
            "stored_message_count": len(self._messages),
            "has_summary": bool(self._running_summary),
            "summary_chars": len(self._running_summary or ""),
            "summarized_message_count": self._last_summarized_count,
        }

    def _summarize(self, messages: list[Any], existing_summary: str | None) -> str:
        buffer = "\n".join(f"{_msg_role(m)}: {msg_content(m)}" for m in messages)
        if existing_summary:
            prompt = (
                "Update the running summary with the new conversation segment.\n\n"
                f"Current summary:\n{existing_summary}\n\n"
                f"New segment:\n{buffer}\n\n"
                "Keep all concrete facts (names, locations, preferences, updates). "
                "Return only the updated summary."
            )
        else:
            prompt = (
                "Summarize the conversation segment with emphasis on user facts, "
                "preferences, and corrections. Return only the summary.\n\n"
                f"{buffer}"
            )
        result = self.summary_model.invoke([HumanMessage(content=prompt)])
        return msg_content(result)


class ProfileMemory:
    """Maintain structured user profile fields and prepend them as system context."""

    name = "Profile"
    description: str

    def __init__(self, k: int = 6):
        self.k = k
        self.description = (
            f"Structured profile + last {k} exchanges. Strong on factual user memory."
        )
        self._messages: list[Any] = []
        self._profile: dict[str, Any] = _new_profile()

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})
        self._update_profile(text)

    def get_messages(self) -> list[Any]:
        recency = _trim_to_window(_to_lc_messages(self._messages), self.k)
        profile_system = self._profile_system_message()
        if profile_system is None:
            return recency
        return [profile_system, *recency]

    def add_agent_response(self, result_messages: list[Any]) -> None:
        self._messages = list(result_messages)

    def start_new_session(self) -> None:
        self._messages = []
        self._profile = _new_profile()

    def snapshot(self) -> dict[str, Any]:
        return {
            "strategy": "profile",
            "window_exchanges": self.k,
            "stored_message_count": len(self._messages),
            "profile": self._profile,
        }

    def _profile_system_message(self) -> SystemMessage | None:
        lines: list[str] = []
        if self._profile["name"]:
            lines.append(f"- Name: {self._profile['name']}")
        if self._profile["profession"]:
            lines.append(f"- Profession: {self._profile['profession']}")
        if self._profile["location"]:
            lines.append(f"- Location: {self._profile['location']}")
        if self._profile["dog_name"]:
            dog = self._profile["dog_name"]
            if self._profile["dog_breed"]:
                dog = f"{dog} ({self._profile['dog_breed']})"
            lines.append(f"- Dog: {dog}")
        if self._profile["favorite_language"]:
            lines.append(f"- Favorite language: {self._profile['favorite_language']}")
        if self._profile["learning_languages"]:
            langs = ", ".join(self._profile["learning_languages"])
            lines.append(f"- Learning: {langs}")
        if self._profile["preferences"]:
            prefs = "; ".join(self._profile["preferences"])
            lines.append(f"- Stated preferences: {prefs}")
        if self._profile["updates"]:
            lines.append(f"- Latest update: {self._profile['updates'][-1]}")

        if not lines:
            return None

        content = (
            "Structured user profile memory (prefer latest values if conflicts):\n"
            + "\n".join(lines)
        )
        return SystemMessage(content=content)

    def _update_profile(self, text: str) -> None:
        raw = text.strip()
        if not raw:
            return

        moved_match = re.search(
            r"\bmoved from ([^,.!?]+?) to ([^.!?]+)", raw, flags=re.IGNORECASE
        )
        if moved_match:
            old_loc = _clean_fact(moved_match.group(1))
            new_loc = _clean_fact(moved_match.group(2))
            self._profile["location"] = new_loc
            self._profile["updates"].append(f"Location updated from {old_loc} to {new_loc}")

        name_match = re.search(r"\bmy name is ([a-z][a-z' -]{1,40})", raw, flags=re.IGNORECASE)
        if name_match:
            self._profile["name"] = _title_case(name_match.group(1))

        prof_match = re.search(
            r"\bi(?:'m| am)\s+(?:an?\s+)?([a-z][a-z -]{2,50})",
            raw,
            flags=re.IGNORECASE,
        )
        if prof_match:
            profession = _clean_fact(prof_match.group(1))
            if any(
                token in profession.lower()
                for token in ("engineer", "developer", "manager", "designer", "analyst", "scientist", "student")
            ):
                self._profile["profession"] = profession

        live_match = re.search(
            r"\b(?:i live in|i am based in|i'm based in)\s+([^.!?]+)",
            raw,
            flags=re.IGNORECASE,
        )
        if live_match:
            self._profile["location"] = _clean_fact(live_match.group(1))

        dog_match = re.search(
            r"\bmy dog's name is ([a-z][a-z'-]{1,30})(?:,\s*(?:he|she|they)\s*(?:is|'s)\s*(?:a|an)\s*([^.!?]+))?",
            raw,
            flags=re.IGNORECASE,
        )
        if dog_match:
            self._profile["dog_name"] = _title_case(dog_match.group(1))
            if dog_match.group(2):
                self._profile["dog_breed"] = _clean_fact(dog_match.group(2))

        fav_lang = re.search(
            r"\bmy favorite programming language is ([a-z0-9+#.\-]+)",
            raw,
            flags=re.IGNORECASE,
        )
        if fav_lang:
            self._profile["favorite_language"] = fav_lang.group(1).strip()

        learning_match = re.search(
            r"\b(?:started learning|start learning|learning)\s+([a-z0-9+#.\-]+)",
            raw,
            flags=re.IGNORECASE,
        )
        if learning_match:
            lang = learning_match.group(1).strip()
            if lang not in self._profile["learning_languages"]:
                self._profile["learning_languages"].append(lang)

        pref_match = re.search(
            r"\b(?:please )?(?:prefer|i prefer|i like)\s+(.+)",
            raw,
            flags=re.IGNORECASE,
        )
        if pref_match:
            pref_text = _clean_fact(pref_match.group(1))
            if pref_text and pref_text not in self._profile["preferences"]:
                self._profile["preferences"].append(pref_text)


class HybridMemory(ProfileMemory):
    """Combine profile memory, running summary, and recency window."""

    name = "Hybrid"
    description: str

    def __init__(
        self,
        k: int = 6,
        model_str: str = "anthropic:claude-haiku-4-5-20251001",
        max_tokens: int = 3500,
        keep_recent: int = 4,
    ):
        super().__init__(k=k)
        self.model_str = model_str
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self._summary_model = None
        self._running_summary: str | None = None
        self._last_summarized_count = 0
        self.description = (
            "Profile + summary + recency window. Balanced recall/cost strategy."
        )

    @property
    def summary_model(self):
        if self._summary_model is None:
            self._summary_model = init_chat_model(self.model_str)
        return self._summary_model

    def get_messages(self) -> list[Any]:
        lc_messages = _to_lc_messages(self._messages)
        total_tokens = count_tokens_approximately(lc_messages)
        recent_messages = lc_messages

        if total_tokens > self.max_tokens:
            keep_count = max(self.keep_recent * 2, 1)
            old_cutoff = len(lc_messages) - keep_count
            if old_cutoff > 0:
                old_messages = lc_messages[:old_cutoff]
                new_segment = old_messages[self._last_summarized_count :]
                if new_segment:
                    self._running_summary = self._summarize(
                        new_segment,
                        self._running_summary,
                    )
                    self._last_summarized_count = old_cutoff
                recent_messages = lc_messages[old_cutoff:]

        recent_window = _trim_to_window(recent_messages, self.k)
        prefix_messages: list[Any] = []
        profile_system = self._profile_system_message()
        if profile_system is not None:
            prefix_messages.append(profile_system)
        if self._running_summary:
            prefix_messages.append(
                SystemMessage(
                    content=(
                        "Conversation summary memory:\n"
                        f"{self._running_summary}\n"
                        "Use this as compressed history context."
                    )
                )
            )
        return [*prefix_messages, *recent_window]

    def start_new_session(self) -> None:
        self._messages = []
        self._profile = _new_profile()
        self._running_summary = None
        self._last_summarized_count = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "strategy": "hybrid",
            "window_exchanges": self.k,
            "max_tokens": self.max_tokens,
            "keep_recent_exchanges": self.keep_recent,
            "stored_message_count": len(self._messages),
            "has_summary": bool(self._running_summary),
            "summary_chars": len(self._running_summary or ""),
            "summarized_message_count": self._last_summarized_count,
            "profile": self._profile,
        }

    def _summarize(self, messages: list[Any], existing_summary: str | None) -> str:
        buffer = "\n".join(f"{_msg_role(m)}: {msg_content(m)}" for m in messages)
        if existing_summary:
            prompt = (
                "Update the running summary with the new segment.\n\n"
                f"Current summary:\n{existing_summary}\n\n"
                f"New segment:\n{buffer}\n\n"
                "Keep concrete user facts and their latest updates. "
                "Return only the updated summary."
            )
        else:
            prompt = (
                "Summarize the conversation segment with emphasis on user facts, "
                "preferences, and corrections. Return only the summary.\n\n"
                f"{buffer}"
            )
        result = self.summary_model.invoke([HumanMessage(content=prompt)])
        return msg_content(result)


class LongTermProfileMemory(ProfileMemory):
    """Persist profile fields across sessions via a local JSON store."""

    name = "Long-Term Profile"

    def __init__(
        self,
        k: int = 6,
        memory_store_path: str = "evals/longterm_memory_store.json",
        memory_key: str = "default_user",
    ):
        super().__init__(k=k)
        self.memory_store_path = Path(memory_store_path)
        self.memory_key = memory_key
        self.description = (
            f"Persists profile across sessions ({memory_key}) + last {k} exchanges."
        )
        self._load_persisted_profile()

    def add_user_message(self, text: str) -> None:
        super().add_user_message(text)
        self._persist_profile()

    def start_new_session(self) -> None:
        self._messages = []
        self._load_persisted_profile()

    def snapshot(self) -> dict[str, Any]:
        base = super().snapshot()
        base["strategy"] = "longterm_profile"
        base["memory_store_path"] = str(self.memory_store_path)
        base["memory_key"] = self.memory_key
        return base

    def _load_store(self) -> dict[str, Any]:
        if not self.memory_store_path.exists():
            return {}
        try:
            with self.memory_store_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
        if isinstance(payload, dict):
            return payload
        return {}

    def _persist_profile(self) -> None:
        store = self._load_store()
        store[self.memory_key] = self._profile
        self.memory_store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.memory_store_path.open("w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2, ensure_ascii=True)
            fh.write("\n")

    def _load_persisted_profile(self) -> None:
        store = self._load_store()
        persisted = store.get(self.memory_key)
        if not isinstance(persisted, dict):
            self._profile = _new_profile()
            return
        merged = _new_profile()
        for key in merged:
            if key in persisted:
                merged[key] = persisted[key]
        self._profile = merged


def _to_lc_messages(messages: list[Any]) -> list[Any]:
    """Convert dict messages to LangChain message objects."""
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            elif role == "system":
                result.append(SystemMessage(content=content))
            else:
                result.append(HumanMessage(content=content))
        else:
            result.append(msg)
    return result


def _msg_role(msg: Any) -> str:
    if isinstance(msg, HumanMessage):
        return "User"
    if isinstance(msg, AIMessage):
        return "Assistant"
    if isinstance(msg, SystemMessage):
        return "System"
    if isinstance(msg, dict):
        return str(msg.get("role", "unknown")).capitalize()
    return type(msg).__name__


def msg_content(msg: Any) -> str:
    """Extract textual content from a message object or dict."""
    if isinstance(msg, dict):
        return str(msg.get("content", ""))
    raw = getattr(msg, "content", None)
    if raw is None:
        return str(msg)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts) if parts else str(raw)
    return str(raw)


def _clean_fact(value: str) -> str:
    cleaned = value.strip().strip(".!?")
    return re.sub(r"\s+", " ", cleaned)


def _title_case(value: str) -> str:
    return " ".join(part.capitalize() for part in _clean_fact(value).split())


STRATEGIES: dict[str, type] = {
    "full": FullBufferMemory,
    "window": WindowMemory,
    "summary": SummaryMemory,
    "profile": ProfileMemory,
    "hybrid": HybridMemory,
    "longterm_profile": LongTermProfileMemory,
}

STRATEGY_NAMES = tuple(STRATEGIES.keys())


def get_strategy(name: str, **kwargs) -> MemoryStrategy:
    """Get a memory strategy by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown memory strategy: {name!r}. Choose from {list(STRATEGIES)}")
    return STRATEGIES[name](**kwargs)
