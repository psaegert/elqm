import json
import os
from datetime import datetime

from elqm.utils import get_dir


class ConversationLogger:
    """
    Creates a current log file and a in sequence numbered log file in the specified folder
    Keeps track of the IO of the query sessions

    Attributes
    ----------
    log_file : str
        The path to the log file.
    """
    def __init__(self, filename: str) -> None:
        self.log_file = os.path.join(get_dir("logs", create=True), f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}_{filename}.jsonl')

    def log(self, role: str, content: str) -> None:
        """
        Log a message to the log file

        Parameters
        ----------
        message : str
            The message to log.
        """
        with open(self.log_file, 'a') as file:
            file.write(json.dumps({
                "role": role,
                "content": content
            }, ensure_ascii=False) + "\n")
