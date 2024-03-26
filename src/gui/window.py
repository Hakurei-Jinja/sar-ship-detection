from PIL import Image
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .utils.np_image import NPImage, NPImageProcessor, NPImageProcessError


class ImageOpenError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class Window(QWidget):

    __image_processor: NPImageProcessor

    def __init__(self, image_processor: NPImageProcessor):
        self.__image_processor = image_processor

        super(Window, self).__init__()
        self.button = QPushButton("Pick Picture")
        self.img_raw = QLabel(self)
        self.img_processed = QLabel(self)

        vertical_layout = QVBoxLayout()
        horizon_layout = QHBoxLayout()
        vertical_layout.addWidget(self.button)
        horizon_layout.addWidget(self.img_raw)
        horizon_layout.addWidget(self.img_processed)
        vertical_layout.addLayout(horizon_layout)
        self.setLayout(vertical_layout)

        self.button.clicked.connect(self.__button_callback)

    def __button_callback(self):
        try:
            image_raw_np = self.__select_image()
            image_processed_np = self.__process_image(image_raw_np)
        except ImageOpenError or NPImageProcessError:
            return

        self.img_raw.setPixmap(image_raw_np.get_Qt_pixmap())
        self.img_processed.setPixmap(image_processed_np.get_Qt_pixmap())

    def __select_image(self) -> NPImage:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        return self.__open_image(file_path)

    def __open_image(self, file_path: str) -> NPImage:
        try:
            return NPImage(Image.open(file_path))
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            raise ImageOpenError(str(e))

    def __process_image(self, image: NPImage) -> NPImage:
        try:
            return self.__image_processor.process(image)
        except NPImageProcessError as e:
            QMessageBox.critical(self, "Error", str(e))
            raise e
