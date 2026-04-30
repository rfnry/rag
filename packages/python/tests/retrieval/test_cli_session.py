from unittest.mock import patch

from rfnry_rag.cli.commands.session import clear_session, list_sessions, load_session, save_turn


class TestSessionStorage:
    def test_empty_session(self, tmp_path):
        with patch("rfnry_rag.cli.commands.session.SESSIONS_DIR", tmp_path / "sessions"):
            result = load_session("nonexistent")
        assert result == []

    def test_save_and_load(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        with patch("rfnry_rag.cli.commands.session.SESSIONS_DIR", sessions_dir):
            save_turn("test", "What is X?", "X is a thing.")
            save_turn("test", "Tell me more", "More details here.")
            history = load_session("test")

        assert len(history) == 2
        assert history[0] == ("What is X?", "X is a thing.")
        assert history[1] == ("Tell me more", "More details here.")

    def test_list_sessions(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        with patch("rfnry_rag.cli.commands.session.SESSIONS_DIR", sessions_dir):
            assert list_sessions() == []
            save_turn("alpha", "q", "a")
            save_turn("beta", "q", "a")
            result = list_sessions()

        assert result == ["alpha", "beta"]

    def test_clear_session(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        with patch("rfnry_rag.cli.commands.session.SESSIONS_DIR", sessions_dir):
            save_turn("temp", "q", "a")
            assert clear_session("temp") is True
            assert load_session("temp") == []

    def test_clear_nonexistent(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        with patch("rfnry_rag.cli.commands.session.SESSIONS_DIR", sessions_dir):
            assert clear_session("nope") is False
