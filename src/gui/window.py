from dataclasses import asdict, dataclass

from PIL import Image
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from ..nn.model import MyYOLO
from .predictor import Predictor, PredictorConfig
from .ui import Ui_MainWindow
from .utils import ModelConfig, NPImage


@dataclass
class ModelItem:
    predictor: Predictor
    structure_img: NPImage
    train_img: NPImage
    eval_img: NPImage

    __getitem__ = lambda self, key: getattr(self, key)
    __setitem__ = lambda self, key, value: setattr(self, key, value)
    dict = asdict


class ModelItemList:
    __models: list[ModelItem]

    def __init__(self, models: list[ModelConfig]):
        self.__models = [
            ModelItem(
                predictor=Predictor(MyYOLO(model.path)),
                structure_img=NPImage(Image.open(model.structure_img_path)),
                train_img=NPImage(Image.open(model.train_img_path)),
                eval_img=NPImage(Image.open(model.eval_img_path)),
            )
            for model in models
        ]

    def get_predictor(self, index: int) -> Predictor:
        return self.__models[index].predictor

    def get_structure_img(self, index: int) -> NPImage:
        return self.__models[index].structure_img

    def get_train_img(self, index: int) -> NPImage:
        return self.__models[index].train_img

    def get_eval_img(self, index: int) -> NPImage:
        return self.__models[index].eval_img


class MainWindow(QMainWindow):
    __raw_img: NPImage | None = None
    __raw_img_qt: QPixmap | None = None
    __result_img_qt: QPixmap | None = None
    __model_item_list: ModelItemList

    __scale_strategy = {
        "aspectMode": Qt.AspectRatioMode.KeepAspectRatio,
        "mode": Qt.TransformationMode.SmoothTransformation,
    }

    def resizeEvent(self, event):
        self.__update_imgs()

    def __update_imgs(self):
        if self.ui.tabWidget.currentIndex() == 0:
            self.__update_result_tab_img()
        elif self.ui.tabWidget.currentIndex() == 1:
            self.__update_structure_tab_img()
        elif self.ui.tabWidget.currentIndex() == 2:
            self.__update_train_tab_img()
        elif self.ui.tabWidget.currentIndex() == 3:
            self.__update_eval_tab_img()

    def __update_result_tab_img(self):
        layout_size = self.ui.resultTab.size()
        result_tab_size = QSize(layout_size.width(), int(layout_size.height() / 2) - 2)
        if self.__raw_img_qt:
            self.ui.rawLabel.setPixmap(
                self.__raw_img_qt.scaled(result_tab_size, **self.__scale_strategy)
            )
        if self.__result_img_qt:
            self.ui.resultLabel.setPixmap(
                self.__result_img_qt.scaled(result_tab_size, **self.__scale_strategy)
            )

    def __update_structure_tab_img(self):
        layout_size = self.ui.structureTab.size()
        structure_tab_size = QSize(layout_size.width(), layout_size.height() - 2)
        self.ui.structureLabel.setPixmap(
            self.__model_item_list.get_structure_img(
                self.ui.modelComboBox.currentIndex()
            )
            .get_Qt_pixmap()
            .scaled(structure_tab_size, **self.__scale_strategy)
        )

    def __update_train_tab_img(self):
        layout_size = self.ui.trainTab.size()
        train_tab_size = QSize(layout_size.width(), layout_size.height() - 2)
        self.ui.trainLabel.setPixmap(
            self.__model_item_list.get_train_img(self.ui.modelComboBox.currentIndex())
            .get_Qt_pixmap()
            .scaled(train_tab_size, **self.__scale_strategy)
        )

    def __update_eval_tab_img(self):
        layout_size = self.ui.evalTab.size()
        eval_tab_size = QSize(layout_size.width(), layout_size.height() - 2)
        self.ui.evalLabel.setPixmap(
            self.__model_item_list.get_eval_img(self.ui.modelComboBox.currentIndex())
            .get_Qt_pixmap()
            .scaled(eval_tab_size, **self.__scale_strategy)
        )

    def __init__(self, models: list[ModelConfig]):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.__model_item_list = ModelItemList(models)
        self.__init_ui(models)
        self.__connect_all()

    def __init_ui(self, models: list[ModelConfig]):
        self.ui.setupUi(self)
        self.ui.modelComboBox.addItems([model.name for model in models])
        self.__update_imgs()

    def __connect_all(self):
        self.ui.tabWidget.currentChanged.connect(self.__update_imgs)
        self.ui.modelComboBox.currentIndexChanged.connect(self.__update_imgs)

        self.ui.selectButton.clicked.connect(self.__select_button_callback)
        self.ui.processButton.clicked.connect(self.__process_button_callback)

        self.ui.confSlider.valueChanged.connect(self.__conf_slider_callback)
        self.ui.confSpinBox.valueChanged.connect(self.__confSpinBox_callback)
        self.ui.iouSlider.valueChanged.connect(self.__iou_slider_callback)
        self.ui.iouSpinBox.valueChanged.connect(self.__iouSpinBox_callback)
        self.ui.labelCheckBox.checkStateChanged.connect(self.__label_checkbox_callback)
        self.ui.confCheckBox.checkStateChanged.connect(self.__conf_checkbox_callback)

    def __conf_slider_callback(self, value):
        self.ui.confSpinBox.setValue(value / self.ui.confSlider.maximum())

    def __confSpinBox_callback(self, value):
        self.ui.confSlider.setValue(round(value * self.ui.confSlider.maximum()))

    def __iou_slider_callback(self, value):
        self.ui.iouSpinBox.setValue(value / self.ui.iouSlider.maximum())

    def __iouSpinBox_callback(self, value):
        self.ui.iouSlider.setValue(round(value * self.ui.iouSlider.maximum()))

    def __label_checkbox_callback(self, state):
        if state == Qt.CheckState.Unchecked:
            self.ui.confCheckBox.setChecked(False)

    def __conf_checkbox_callback(self, state):
        if state == Qt.CheckState.Checked:
            self.ui.labelCheckBox.setChecked(True)

    def __select_button_callback(self):
        try:
            self.__raw_img = self.__select_image()
            self.__raw_img_qt = self.__raw_img.get_Qt_pixmap()
            self.__update_imgs()
        except:
            QMessageBox.critical(self, "Error", "加载图片图像")

    def __select_image(self) -> NPImage:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        return NPImage(Image.open(file_path))

    def __process_button_callback(self):
        if not self.__raw_img:
            QMessageBox.critical(self, "Error", "请先选择图像")
            return
        try:
            predictor = self.__model_item_list.get_predictor(
                self.ui.modelComboBox.currentIndex()
            )
            result_img = predictor.predict(self.__raw_img, self.__get_predict_config())
            self.__result_img_qt = result_img.get_Qt_pixmap()
            self.__update_imgs()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def __get_predict_config(self):
        return PredictorConfig(
            conf=self.ui.confSpinBox.value(),
            iou=self.ui.iouSpinBox.value(),
            augment=self.ui.augmentCheckBox.isChecked(),
            show_labels=self.ui.labelCheckBox.isChecked(),
            show_conf=self.ui.confCheckBox.isChecked(),
            save=self.ui.saveCheckBox.isChecked(),
            save_dir=self.ui.modelComboBox.currentText().replace("/", "_"),
        )
