import os
import unittest

from elqm.logging.conversation_logger import ConversationLogger


class TestConversationLogger(unittest.TestCase):

    def test_conversation_logger(self) -> None:
        """Test if the ConversationLogger works correctly."""
        logger = ConversationLogger("pytest")
        self.log_file = logger.log_file
        logger.log(role="pytest", content="pytest")
        with open(logger.log_file, 'r') as file:
            self.assertEqual(file.read(), '{"role": "pytest", "content": "pytest"}\n')

    def tearDown(self) -> None:
        # Remove the log file
        os.remove(self.log_file)
