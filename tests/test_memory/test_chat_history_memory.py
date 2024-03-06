import unittest

from langchain_core.messages import AIMessage, HumanMessage

from elqm.memory.chat_history_memory import ChatHistoryMemory


class TestChatHistoryMemory(unittest.TestCase):

    def setUp(self) -> None:
        self.chat_history_memory = ChatHistoryMemory(history_size=2, max_length=100)

    def test_add_ai_message(self) -> None:
        """Test if the ChatHistoryMemory adds an AI message correctly."""
        self.chat_history_memory.add_ai_message("AI message")
        self.assertEqual(self.chat_history_memory.get_memory(), [AIMessage(content="AI message")])

    def test_add_user_message(self) -> None:
        """Test if the ChatHistoryMemory adds a user message correctly."""
        self.chat_history_memory.add_user_message("User message")
        self.assertEqual(self.chat_history_memory.get_memory(), [HumanMessage(content="User message")])

    def test_add_ai_and_user_message(self) -> None:
        """Test if the ChatHistoryMemory adds an AI and a user message correctly."""
        self.chat_history_memory.add_ai_message("AI message")
        self.chat_history_memory.add_user_message("User message")
        self.assertEqual(self.chat_history_memory.get_memory(), [
            AIMessage(content="AI message"),
            HumanMessage(content="User message")])

    def test_add_user_and_ai_message(self) -> None:
        """Test if the ChatHistoryMemory adds a user and an AI message correctly."""
        self.chat_history_memory.add_user_message("User message")
        self.chat_history_memory.add_ai_message("AI message")
        self.assertEqual(self.chat_history_memory.get_memory(), [
            HumanMessage(content="User message"),
            AIMessage(content="AI message")])

    def test_clear_memory(self) -> None:
        """Test if the ChatHistoryMemory clears the memory correctly."""
        self.chat_history_memory.add_ai_message("AI message")
        self.chat_history_memory.add_user_message("User message")
        self.chat_history_memory.clear_memory()
        self.assertEqual(self.chat_history_memory.get_memory(), [])
