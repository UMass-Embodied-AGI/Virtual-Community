import logging

class AgentLogger(logging.Logger):
    def __init__(self, name: str, level: str, output_path: str = None):
        super().__init__(f"Agent <{name}>")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging._nameToLevel[level.upper()])
        self.addHandler(console_handler)
        if output_path:
            file_handler = logging.FileHandler(output_path)
            file_handler.setLevel(logging.DEBUG)
            self.addHandler(file_handler)
        self.formatter = logging.Formatter(
            f"[%(asctime)s] [%(levelname)s] [Agent <{name}>] %(message)s"
        )
        for handler in self.handlers:
            handler.setFormatter(self.formatter)
