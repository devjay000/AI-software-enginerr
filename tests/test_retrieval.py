import unittest

from ai_engineer.config import Settings
from ai_engineer.models import CodeChunk
from ai_engineer.retrieval import ContextBuilder, HybridRetriever


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            CodeChunk(
                file="auth/session.py",
                symbol="restore_session",
                kind="function",
                language="python",
                start_line=10,
                end_line=18,
                content="def restore_session():\n    return token_storage.load()",
            ),
            CodeChunk(
                file="ui/dashboard.py",
                symbol="Dashboard",
                kind="class",
                language="python",
                start_line=1,
                end_line=5,
                content="class Dashboard:\n    def render(self):\n        return None",
            ),
        ]
        self.retriever = HybridRetriever(
            self.chunks, {"auth/session.py": ["token_storage"]}, Settings()
        )

    def test_hybrid_search_finds_structural_auth_chunk(self):
        plan = self.retriever.rewrite_query(
            "Users logout after refresh token session restore"
        )
        hits = self.retriever.search(plan, top_k=2)
        self.assertEqual(hits[0].chunk.file, "auth/session.py")
        self.assertTrue(hits[0].channels)

    def test_context_builder_respects_untrusted_boundary(self):
        hits = self.retriever.search(
            self.retriever.rewrite_query("session token"), top_k=1
        )
        bundle = ContextBuilder(2000).build("session token", "Python app", hits)
        self.assertIn("untrusted_repository_content", bundle.prompt_context)
        self.assertLessEqual(bundle.estimated_tokens, 2000)


if __name__ == "__main__":
    unittest.main()
