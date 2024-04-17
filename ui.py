import sys

from PySide6.QtWidgets import QApplication

from src.gui.utils import NPImagePredictor
from src.gui.window import Window
from src.nn.model import MyYOLO


if __name__ == "__main__":
    model = MyYOLO("./models/trained/v8n/weights/best.pt", verbose=True)
    app = QApplication(sys.argv)
    form = Window(NPImagePredictor(model))
    form.show()
    sys.exit(app.exec())
