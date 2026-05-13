import json
import os
import uuid
import time
from typing import Dict, List, Optional, Any

class SessionManager:
    """
    Manages conversation sessions stored in a JSON file.
    Provides methods to create, retrieve, and update sessions.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._lock = None
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load sessions from JSON file; create if missing."""
        if not os.path.exists(self.filepath):
            self._save({"sessions": {}})
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # Fallback to empty dict if corrupted
            return {"sessions": {}}

    def _save(self, data: Dict[str, Any]) -> None:
        """Atomically write data to JSON file."""
        temp_path = self.filepath + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, self.filepath)

    def _write(self) -> None:
        """Write current data to disk."""
        self._save(self._data)

    def create_session(self, title: Optional[str] = None) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        timestamp = time.time()
        session = {
            "session_id": session_id,
            "timestamp": timestamp,
            "title": title or "New Conversation",
            "messages": []
        }
        self._data["sessions"][session_id] = session
        self._write()
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return session dict or None if not found."""
        return self._data["sessions"].get(session_id)

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """
        Append a message to the session and update timestamp.
        Returns True if successful, False if session not found.
        """
        session = self.get_session(session_id)
        if not session:
            return False
        session["messages"].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        session["timestamp"] = time.time()
        # Update title if it's the first user message
        if len(session["messages"]) == 1 and role == "user":
            session["title"] = content[:50] + ("..." if len(content) > 50 else "")
        self._write()
        return True

    def get_recent_messages(self, session_id: str, limit: int = 5) -> List[Dict[str, str]]:
        """Return the last N messages (role, content) without timestamps."""
        session = self.get_session(session_id)
        if not session:
            return []
        messages = session["messages"][-limit:]
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return metadata for all sessions, sorted by most recent."""
        sessions = list(self._data["sessions"].values())
        sessions.sort(key=lambda s: s["timestamp"], reverse=True)
        # Return only necessary fields for the UI
        return [{
            "session_id": s["session_id"],
            "title": s["title"],
            "timestamp": s["timestamp"],
            "message_count": len(s["messages"])
        } for s in sessions]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID. Returns True if existed."""
        if session_id in self._data["sessions"]:
            del self._data["sessions"][session_id]
            self._write()
            return True
        return False