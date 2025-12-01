# memory/short_term.py
from typing import List, Dict
from collections import defaultdict

class ShortTermMemory:
    """
    Simple in-process short-term memory keyed by session_id.
    Replace with Redis later if needed.
    """

    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    def append(self, session_id: str, role: str, content: str) -> None:
        self.sessions[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.sessions[session_id]
