import pandas as pd
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QLabel, QMainWindow, QWidget, QFileDialog, QInputDialog, QRadioButton, QButtonGroup, QComboBox, QGroupBox
from PyQt5.QtCore import pyqtSignal
from ..nav_row import NavRow

class SecondPage(QWidget):
    done = pyqtSignal()
    back = pyqtSignal()

    def __init__(self, analyses):
        super().__init__()

        time = QRadioButton('Time Series')
        time.clicked.connect(self.btn_selection_changed)
        batch = QRadioButton('Batch Level')
        batch.clicked.connect(self.btn_selection_changed)
        tb_group = QButtonGroup()
        tb_group.addButton(time)
        tb_group.addButton(batch)
        data_widget = QGroupBox('Data Type')
        data_row = QHBoxLayout()
        data_widget.setLayout(data_row)
        data_row.addWidget(time)
        data_row.addWidget(batch)

        qualitative = QRadioButton('Qualitative | Classification')
        qualitative.clicked.connect(self.btn_selection_changed)
        quantitative = QRadioButton('Quantitative | Regression')
        quantitative.clicked.connect(self.btn_selection_changed)
        qq_group = QButtonGroup()
        qq_group.addButton(qualitative)
        qq_group.addButton(quantitative)
        type_widget = QGroupBox('Analysis type')
        type_row = QHBoxLayout()
        type_widget.setLayout(type_row)
        type_row.addWidget(qualitative)
        type_row.addWidget(quantitative)

        sel_type = QComboBox()
        sel_type.addItem('Please select an option for each of the radio buttons above')
        sel_type.currentIndexChanged.connect(self.combobox_changed)

        primary_layout = QVBoxLayout()

        nav_row = NavRow(1, 5)
        nav_row.enable_next(False)
        nav_row.next_.connect(self.done.emit)
        nav_row.back.connect(self.back.emit)


        primary_layout.addWidget(data_widget)
        primary_layout.addWidget(type_widget)
        primary_layout.addWidget(sel_type)
        primary_layout.addStretch()
        primary_layout.addWidget(nav_row)

        self.setLayout(primary_layout)

        # internal state

        self.analyses = analyses

        self.nav_row = nav_row
        self.combobox = sel_type
        self.btns = {'time': time, 'batch': batch, 'qualitative': qualitative, 'quantitative': quantitative}

    def btn_selection_changed(self):
        # get the currently selected method to try to restore it later
        current_method = self.combobox.currentText()

        self.combobox.clear()


        time_or_batch = self.get_time_or_batch()
        classification_or_regression = self.get_classification_or_regression()

        if time_or_batch is None or classification_or_regression is None:
            self.combobox.addItem('Please select an option for each of the radio buttons above')
            return
        else:
            self.combobox.addItem('Choose analysis method')


        for analysis in self.analyses:
            if analysis.is_available(time_or_batch, classification_or_regression):
                self.combobox.addItem(analysis.get_name())

        idx = self.combobox.findText(current_method)
        if idx != -1:
            self.combobox.setCurrentIndex(idx)
        else:
            self.combobox_changed()

    def combobox_changed(self):
        current_index = self.combobox.currentIndex()
        current_method = self.combobox.currentText()
        self.nav_row.enable_next(current_index > 0)

    def get_analysis(self):
        # get the analysis onject corresponding to the currently selected text. We first only keep the elements, which match the name (should be exactly 1) annd then return the next (that one) element (or None)
        return next(filter(lambda x: self.combobox.currentText() == x.get_name(), self.analyses), None)

    def get_time_or_batch(self):
        return 'time' if self.btns['time'].isChecked() else 'batch' if self.btns['batch'].isChecked() else None

    def get_classification_or_regression(self):
        return 'classification' if self.btns['qualitative'].isChecked() else 'regression' if self.btns['quantitative'].isChecked() else None

if __name__ == '__main__':
    app = QApplication([])
    main_window = SecondPage()
    main_window.show()
    app.exec_()
