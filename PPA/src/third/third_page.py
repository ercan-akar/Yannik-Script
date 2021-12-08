import pandas as pd
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QLabel, QMainWindow, QWidget, QFileDialog, QInputDialog, QRadioButton, QButtonGroup, QComboBox, QGroupBox, QGridLayout, QListWidget, QAbstractItemView
from PyQt5.QtCore import pyqtSignal
from ..nav_row import NavRow
from .list_selection import ListSelection

class ThirdPage(QWidget):
    done = pyqtSignal()
    back = pyqtSignal()

    def __init__(self):
        super().__init__()

        # General stuff

        g_group = QGroupBox('General')
        g_layout = QGridLayout()
        g_group.setLayout(g_layout)

        g_labels = { 'Timestamp': QLabel('Timestamp Column'), 'Batch ID': QLabel('Batch ID Column') }
        g_boxes = { 'Timestamp': QComboBox(), 'Batch ID': QComboBox() }

        g_layout.addWidget(g_labels['Timestamp'], 0, 0)
        g_layout.addWidget(g_boxes['Timestamp'],  0, 1)
        g_layout.addWidget(g_labels['Batch ID'],  1, 0)
        g_layout.addWidget(g_boxes['Batch ID'],   1, 1)

        self.g_boxes = g_boxes

        # Y Variable
        # since this depends crucially on the selected method, this won't be initialized here

        y_group = QGroupBox('Y Variables')
        y_layout = QHBoxLayout()
        y_group.setLayout(y_layout)

        self.y_layout = y_layout

        # X Variables
        # since this depends crucially on the selected method, this won't be initialized here

        x_group = QGroupBox('X Variables')
        x_layout = QHBoxLayout()
        x_group.setLayout(x_layout)
        self.x_layout = x_layout

        # this will be a handle to the x selection widgets (like y_box and g_boxes)
        self.x = None

        primary_layout = QVBoxLayout()

        nav_row = NavRow(2, 5)
        nav_row.enable_next(True)
        nav_row.next_.connect(self.done.emit)
        nav_row.back.connect(self.back.emit)
        self.nav_row = nav_row

        primary_layout.addWidget(g_group)
        primary_layout.addWidget(y_group)
        primary_layout.addWidget(x_group)
        primary_layout.addWidget(nav_row)

        self.setLayout(primary_layout)

    def set_cols_analysis_and_parameters(self, cols, analysis, time_or_batch, classification_or_regression):
        current_time = self.g_boxes['Timestamp'].currentText()
        current_batch = self.g_boxes['Batch ID'].currentText()

        # clear all boxes
        self.g_boxes['Timestamp'].clear()
        self.g_boxes['Batch ID'].clear()
        # self.y_box.clear()

        # refill boxes
        self.g_boxes['Timestamp'].addItems(cols)
        self.g_boxes['Batch ID'].addItems(cols)

        # restore boxes
        t_idx = self.g_boxes['Timestamp'].findText(current_time)
        b_idx = self.g_boxes['Batch ID'].findText(current_batch)

        if t_idx >= 0:
            self.g_boxes['Timestamp'].setCurrentIndex(t_idx)
        if b_idx >= 0:
            self.g_boxes['Batch ID'].setCurrentIndex(b_idx)

        # reset/change Y selector
        while self.y_layout.count() > 0:
            self.y_layout.takeAt(0).widget().setParent(None)

        if not analysis.single_y(time_or_batch, classification_or_regression):
            # multiple Y
            ls = ListSelection(cols)

            self.y_layout.addWidget(ls)
            self.y = ls
        else:
            # single Y
            y_label = QLabel('Choose X Variable')
            y_box = QComboBox()
            y_box.addItems(cols)

            self.y_layout.addWidget(x_label)
            self.y_layout.addWidget(x_box)

            self.y = y_box

        # reset/change X selector
        while self.x_layout.count() > 0:
            self.x_layout.takeAt(0).widget().setParent(None)

        if not analysis.single_x(time_or_batch, classification_or_regression):
            # multiple X
            ls = ListSelection(cols)

            self.x_layout.addWidget(ls)
            self.x = ls
        else:
            # single X
            x_label = QLabel('Choose X Variable')
            x_box = QComboBox()
            x_box.addItems(cols)

            self.x_layout.addWidget(x_label)
            self.x_layout.addWidget(x_box)

            self.x = x_box

    def get_variables(self):
        data = {'time': self.g_boxes['Timestamp'].currentText(), 'batch': self.g_boxes['Batch ID'].currentText()}
        if type(self.x) == ListSelection:
            data.update({'x': self.x.get_selected()})
        else:
            data.update({'x': self.x.currentText()})
        if type(self.y) == ListSelection:
            data.update({'y': self.y.get_selected()})
        else:
            data.update({'y': self.y.currentText()})
        return data

if __name__ == '__main__':
    app = QApplication([])
    main_window = ThirdPage()
    main_window.set_df_and_method(pd.DataFrame(columns=['Time', 'Batch', 'Quality', 'Data 1', 'Data 2']), 'PLS')
    main_window.show()
    app.exec_()
