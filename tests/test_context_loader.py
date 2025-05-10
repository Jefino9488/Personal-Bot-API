import unittest
from app.context_loader import chunk_text


class TestContextLoader(unittest.TestCase):
    def test_chunk_text_small_text(self):
        """Test that chunk_text returns the entire text as one chunk if it's small enough."""
        # Arrange
        text = "This is a small text that should be returned as a single chunk."
        chunk_size = 100
        overlap = 20

        # Act
        chunks = chunk_text(text, chunk_size, overlap)

        # Assert
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_chunk_text_large_text(self):
        """Test that chunk_text correctly splits large text into overlapping chunks."""
        # Arrange
        words = ["word" + str(i) for i in range(100)]  # Generate 100 unique words
        text = " ".join(words)
        chunk_size = 30
        overlap = 10

        # Act
        chunks = chunk_text(text, chunk_size, overlap)

        # Assert
        # Should have 5 chunks: 0-29, 20-49, 40-69, 60-89, 80-99
        self.assertEqual(len(chunks), 5)

        # Check first and last chunks
        self.assertEqual(chunks[0], " ".join(words[0:30]))
        self.assertEqual(chunks[-1], " ".join(words[80:100]))

        # Check overlap between chunks
        for i in range(1, len(chunks)):
            # The last 10 words of the previous chunk should be the first 10 words of the current chunk
            prev_chunk_words = chunks[i - 1].split()[-overlap:]
            curr_chunk_words = chunks[i].split()[:overlap]
            self.assertEqual(prev_chunk_words, curr_chunk_words)


if __name__ == "__main__":
    unittest.main()
