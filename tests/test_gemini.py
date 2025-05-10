import unittest
from app.gemini import create_prompt


class TestGemini(unittest.TestCase):
    def test_create_prompt(self):
        """Test that create_prompt correctly formats the prompt with question and context."""
        # Arrange
        question = "What are my skills?"
        context = "John Doe has skills in Python, FastAPI, and Docker."

        # Act
        prompt = create_prompt(question, context)

        # Assert
        self.assertIn(question, prompt)
        self.assertIn(context, prompt)
        self.assertIn("Given the context information and not prior knowledge", prompt)
        self.assertIn("If the answer cannot be found in the context", prompt)


if __name__ == "__main__":
    unittest.main()
