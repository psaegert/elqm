from langchain.memory import ConversationBufferWindowMemory


class ChatHistoryMemory:
    def __init__(self, history_size: int, max_length: int):
        self.history_size = history_size
        self.max_length = max_length  # TODO: Find out how to truncate output of chat history memory
        self.memory_key = "chat_history"
        self.memory = ConversationBufferWindowMemory(memory_key=self.memory_key, k=self.history_size, return_messages=True)

    def add_ai_message(self, message: str) -> None:
        """
        Add a message from the AI to the chat history memory.

        Parameters
        ----------
        message : str
            The message to add.
        """
        self.memory.chat_memory.add_ai_message(message)

    def add_user_message(self, message: str) -> None:
        """
        Add a message from the user to the chat history memory.

        Parameters
        ----------
        message : str
            The message to add.
        """
        self.memory.chat_memory.add_user_message(message)

    def clear_memory(self) -> None:
        """
        Clear the chat history memory.
        """
        self.memory.clear()

    def get_memory(self) -> list[str]:
        """
        Get the chat history memory.

        Returns
        -------
        list[str]
            The chat history memory.
        """
        return self.memory.load_memory_variables({})[self.memory_key]
