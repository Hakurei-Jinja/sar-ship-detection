# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'window.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QSlider, QSpacerItem,
    QTabWidget, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(768, 576)
        MainWindow.setMinimumSize(QSize(640, 480))
        MainWindow.setIconSize(QSize(24, 24))
        MainWindow.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.mainHorizontalLayout = QHBoxLayout(self.centralwidget)
        self.mainHorizontalLayout.setObjectName(u"mainHorizontalLayout")
        self.leftPannel = QWidget(self.centralwidget)
        self.leftPannel.setObjectName(u"leftPannel")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.leftPannel.sizePolicy().hasHeightForWidth())
        self.leftPannel.setSizePolicy(sizePolicy)
        self.leftPannel.setMinimumSize(QSize(175, 0))
        self.leftPannel.setMaximumSize(QSize(200, 16777215))
        self.pannelVerticalLayout = QVBoxLayout(self.leftPannel)
        self.pannelVerticalLayout.setSpacing(4)
        self.pannelVerticalLayout.setObjectName(u"pannelVerticalLayout")
        self.pannelVerticalLayout.setContentsMargins(1, 2, 1, 2)
        self.modelVerticalLayout = QVBoxLayout()
        self.modelVerticalLayout.setSpacing(0)
        self.modelVerticalLayout.setObjectName(u"modelVerticalLayout")
        self.modelLabel = QLabel(self.leftPannel)
        self.modelLabel.setObjectName(u"modelLabel")

        self.modelVerticalLayout.addWidget(self.modelLabel)

        self.modelComboBox = QComboBox(self.leftPannel)
        self.modelComboBox.setObjectName(u"modelComboBox")
        self.modelComboBox.setEditable(False)

        self.modelVerticalLayout.addWidget(self.modelComboBox)


        self.pannelVerticalLayout.addLayout(self.modelVerticalLayout)

        self.confVerticalLayout = QVBoxLayout()
        self.confVerticalLayout.setSpacing(0)
        self.confVerticalLayout.setObjectName(u"confVerticalLayout")
        self.confLabel = QLabel(self.leftPannel)
        self.confLabel.setObjectName(u"confLabel")
        self.confLabel.setTextFormat(Qt.PlainText)
        self.confLabel.setScaledContents(False)
        self.confLabel.setWordWrap(False)

        self.confVerticalLayout.addWidget(self.confLabel)

        self.confWidget = QWidget(self.leftPannel)
        self.confWidget.setObjectName(u"confWidget")
        self.confHorizontalLayout = QHBoxLayout(self.confWidget)
        self.confHorizontalLayout.setSpacing(2)
        self.confHorizontalLayout.setObjectName(u"confHorizontalLayout")
        self.confHorizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.confSlider = QSlider(self.confWidget)
        self.confSlider.setObjectName(u"confSlider")
        self.confSlider.setMaximum(100)
        self.confSlider.setSingleStep(5)
        self.confSlider.setValue(25)
        self.confSlider.setTracking(True)
        self.confSlider.setOrientation(Qt.Horizontal)
        self.confSlider.setInvertedAppearance(False)
        self.confSlider.setInvertedControls(False)
        self.confSlider.setTickPosition(QSlider.TicksBelow)
        self.confSlider.setTickInterval(10)

        self.confHorizontalLayout.addWidget(self.confSlider)

        self.confSpinBox = QDoubleSpinBox(self.confWidget)
        self.confSpinBox.setObjectName(u"confSpinBox")
        self.confSpinBox.setDecimals(2)
        self.confSpinBox.setMaximum(1.000000000000000)
        self.confSpinBox.setSingleStep(0.050000000000000)
        self.confSpinBox.setValue(0.250000000000000)

        self.confHorizontalLayout.addWidget(self.confSpinBox)

        self.confHorizontalLayout.setStretch(0, 7)
        self.confHorizontalLayout.setStretch(1, 3)

        self.confVerticalLayout.addWidget(self.confWidget)


        self.pannelVerticalLayout.addLayout(self.confVerticalLayout)

        self.iouVerticalLayout = QVBoxLayout()
        self.iouVerticalLayout.setSpacing(0)
        self.iouVerticalLayout.setObjectName(u"iouVerticalLayout")
        self.iouLabel = QLabel(self.leftPannel)
        self.iouLabel.setObjectName(u"iouLabel")

        self.iouVerticalLayout.addWidget(self.iouLabel)

        self.iouWidget = QWidget(self.leftPannel)
        self.iouWidget.setObjectName(u"iouWidget")
        self.iouHorizontalLayout = QHBoxLayout(self.iouWidget)
        self.iouHorizontalLayout.setSpacing(2)
        self.iouHorizontalLayout.setObjectName(u"iouHorizontalLayout")
        self.iouHorizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.iouSlider = QSlider(self.iouWidget)
        self.iouSlider.setObjectName(u"iouSlider")
        self.iouSlider.setMaximum(10)
        self.iouSlider.setSingleStep(1)
        self.iouSlider.setPageStep(1)
        self.iouSlider.setValue(7)
        self.iouSlider.setOrientation(Qt.Horizontal)
        self.iouSlider.setTickPosition(QSlider.TicksBelow)
        self.iouSlider.setTickInterval(0)

        self.iouHorizontalLayout.addWidget(self.iouSlider)

        self.iouSpinBox = QDoubleSpinBox(self.iouWidget)
        self.iouSpinBox.setObjectName(u"iouSpinBox")
        self.iouSpinBox.setDecimals(1)
        self.iouSpinBox.setMaximum(1.000000000000000)
        self.iouSpinBox.setSingleStep(0.100000000000000)
        self.iouSpinBox.setValue(0.600000000000000)

        self.iouHorizontalLayout.addWidget(self.iouSpinBox)

        self.iouHorizontalLayout.setStretch(0, 7)
        self.iouHorizontalLayout.setStretch(1, 3)

        self.iouVerticalLayout.addWidget(self.iouWidget)


        self.pannelVerticalLayout.addLayout(self.iouVerticalLayout)

        self.augmentCheckBox = QCheckBox(self.leftPannel)
        self.augmentCheckBox.setObjectName(u"augmentCheckBox")

        self.pannelVerticalLayout.addWidget(self.augmentCheckBox)

        self.labelCheckBox = QCheckBox(self.leftPannel)
        self.labelCheckBox.setObjectName(u"labelCheckBox")
        self.labelCheckBox.setChecked(True)

        self.pannelVerticalLayout.addWidget(self.labelCheckBox)

        self.confCheckBox = QCheckBox(self.leftPannel)
        self.confCheckBox.setObjectName(u"confCheckBox")
        self.confCheckBox.setChecked(True)
        self.confCheckBox.setTristate(False)

        self.pannelVerticalLayout.addWidget(self.confCheckBox)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.pannelVerticalLayout.addItem(self.verticalSpacer)

        self.saveCheckBox = QCheckBox(self.leftPannel)
        self.saveCheckBox.setObjectName(u"saveCheckBox")

        self.pannelVerticalLayout.addWidget(self.saveCheckBox)

        self.selectButton = QPushButton(self.leftPannel)
        self.selectButton.setObjectName(u"selectButton")

        self.pannelVerticalLayout.addWidget(self.selectButton)

        self.processButton = QPushButton(self.leftPannel)
        self.processButton.setObjectName(u"processButton")

        self.pannelVerticalLayout.addWidget(self.processButton)


        self.mainHorizontalLayout.addWidget(self.leftPannel)

        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setTabShape(QTabWidget.Rounded)
        self.resultTab = QWidget()
        self.resultTab.setObjectName(u"resultTab")
        self.resultVerticalLayout = QVBoxLayout(self.resultTab)
        self.resultVerticalLayout.setSpacing(0)
        self.resultVerticalLayout.setObjectName(u"resultVerticalLayout")
        self.resultVerticalLayout.setContentsMargins(0, 0, 0, 0)
        self.rawLabel = QLabel(self.resultTab)
        self.rawLabel.setObjectName(u"rawLabel")
        self.rawLabel.setAlignment(Qt.AlignCenter)

        self.resultVerticalLayout.addWidget(self.rawLabel)

        self.line = QFrame(self.resultTab)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.resultVerticalLayout.addWidget(self.line)

        self.resultLabel = QLabel(self.resultTab)
        self.resultLabel.setObjectName(u"resultLabel")
        self.resultLabel.setAlignment(Qt.AlignCenter)

        self.resultVerticalLayout.addWidget(self.resultLabel)

        self.tabWidget.addTab(self.resultTab, "")
        self.structureTab = QWidget()
        self.structureTab.setObjectName(u"structureTab")
        self.structureVerticalLayout = QVBoxLayout(self.structureTab)
        self.structureVerticalLayout.setSpacing(0)
        self.structureVerticalLayout.setObjectName(u"structureVerticalLayout")
        self.structureVerticalLayout.setContentsMargins(0, 0, 0, 0)
        self.structureLabel = QLabel(self.structureTab)
        self.structureLabel.setObjectName(u"structureLabel")
        self.structureLabel.setAlignment(Qt.AlignCenter)

        self.structureVerticalLayout.addWidget(self.structureLabel)

        self.tabWidget.addTab(self.structureTab, "")
        self.trainTab = QWidget()
        self.trainTab.setObjectName(u"trainTab")
        self.verticalLayout = QVBoxLayout(self.trainTab)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.trainLabel = QLabel(self.trainTab)
        self.trainLabel.setObjectName(u"trainLabel")
        self.trainLabel.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.trainLabel)

        self.tabWidget.addTab(self.trainTab, "")
        self.evalTab = QWidget()
        self.evalTab.setObjectName(u"evalTab")
        self.evalVerticalLayout = QVBoxLayout(self.evalTab)
        self.evalVerticalLayout.setSpacing(0)
        self.evalVerticalLayout.setObjectName(u"evalVerticalLayout")
        self.evalVerticalLayout.setContentsMargins(0, 0, 0, 0)
        self.evalLabel = QLabel(self.evalTab)
        self.evalLabel.setObjectName(u"evalLabel")
        self.evalLabel.setAlignment(Qt.AlignCenter)

        self.evalVerticalLayout.addWidget(self.evalLabel)

        self.tabWidget.addTab(self.evalTab, "")

        self.mainHorizontalLayout.addWidget(self.tabWidget)

        self.mainHorizontalLayout.setStretch(0, 1)
        self.mainHorizontalLayout.setStretch(1, 3)
        MainWindow.setCentralWidget(self.centralwidget)
        QWidget.setTabOrder(self.modelComboBox, self.confSlider)
        QWidget.setTabOrder(self.confSlider, self.confSpinBox)
        QWidget.setTabOrder(self.confSpinBox, self.iouSlider)
        QWidget.setTabOrder(self.iouSlider, self.iouSpinBox)
        QWidget.setTabOrder(self.iouSpinBox, self.augmentCheckBox)
        QWidget.setTabOrder(self.augmentCheckBox, self.labelCheckBox)
        QWidget.setTabOrder(self.labelCheckBox, self.confCheckBox)
        QWidget.setTabOrder(self.confCheckBox, self.saveCheckBox)
        QWidget.setTabOrder(self.saveCheckBox, self.selectButton)
        QWidget.setTabOrder(self.selectButton, self.processButton)
        QWidget.setTabOrder(self.processButton, self.tabWidget)

        self.retranslateUi(MainWindow)

        self.modelComboBox.setCurrentIndex(-1)
        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"SAR\u56fe\u50cf\u8230\u8239\u76ee\u6807\u8bc6\u522b", None))
        self.modelLabel.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u6a21\u578b", None))
        self.modelComboBox.setCurrentText("")
        self.modelComboBox.setPlaceholderText("")
        self.confLabel.setText(QCoreApplication.translate("MainWindow", u"\u7f6e\u4fe1\u5ea6\u9608\u503c", None))
        self.iouLabel.setText(QCoreApplication.translate("MainWindow", u"IoU\u9608\u503c(\u7528\u4e8eNMS\u7b97\u6cd5)", None))
        self.augmentCheckBox.setText(QCoreApplication.translate("MainWindow", u"\u6d4b\u8bd5\u65f6\u6570\u636e\u589e\u5f3a(TTA)", None))
        self.labelCheckBox.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u6807\u7b7e", None))
        self.confCheckBox.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u7f6e\u4fe1\u5ea6", None))
        self.saveCheckBox.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u9884\u6d4b\u7ed3\u679c", None))
        self.selectButton.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u56fe\u50cf", None))
        self.processButton.setText(QCoreApplication.translate("MainWindow", u"\u5904\u7406\u56fe\u50cf", None))
        self.rawLabel.setText("")
        self.resultLabel.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.resultTab), QCoreApplication.translate("MainWindow", u"\u5b9e\u65f6\u5904\u7406\u7ed3\u679c", None))
        self.structureLabel.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.structureTab), QCoreApplication.translate("MainWindow", u"\u6a21\u578b\u7ed3\u6784", None))
        self.trainLabel.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.trainTab), QCoreApplication.translate("MainWindow", u"\u8bad\u7ec3\u8fc7\u7a0b", None))
        self.evalLabel.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.evalTab), QCoreApplication.translate("MainWindow", u"PR\u66f2\u7ebf", None))
    # retranslateUi

