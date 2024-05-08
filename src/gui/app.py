import sys

from PySide6.QtWidgets import QApplication

from src.gui.utils import ModelConfig
from src.gui.window import MainWindow


class App:
    def __init__(self, models: list[ModelConfig] | ModelConfig):
        self.models = self.__to_list(models)

    @staticmethod
    def __to_list(models: list[ModelConfig] | ModelConfig) -> list[ModelConfig]:
        if isinstance(models, list):
            return models
        return [models]

    def run(self):
        app = QApplication(sys.argv)
        window = MainWindow(self.models)
        window.show()
        sys.exit(app.exec())
