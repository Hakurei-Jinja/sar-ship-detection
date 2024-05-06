import sys

from PySide6.QtWidgets import QApplication

from src.gui.utils import ModelConfig
from src.gui.window import MainWindow


class App:
    def __init__(self, models: list[ModelConfig]):
        self.models = models

    def run(self):
        app = QApplication(sys.argv)
        window = MainWindow(self.models)
        window.show()
        sys.exit(app.exec())
