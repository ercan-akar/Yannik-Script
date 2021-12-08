import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QHBoxLayout, QCheckBox
from PyQt5.QtCore import pyqtSignal


class RowElement(QWidget):
    remove = pyqtSignal(int)

    def __init__(self, sid, filepath):
        super().__init__()

        # assert type(filepath) == str, 'the filepath must be a str'

        layout = QHBoxLayout()
        check = QCheckBox()
        check.setCheckState(2)
        text = QLabel('{} | Not Loaded'.format(filepath))
        del_button = QPushButton('Remove')
        del_button.clicked.connect(self.fire_remove_signal)
        width = del_button.fontMetrics().boundingRect('Remove').width() + 20
        del_button.setMaximumWidth(width)

        layout.addWidget(check)
        layout.addWidget(text)
        layout.addStretch()
        layout.addWidget(del_button)

        self.check = check
        self.label = text
        self.filepath = filepath
        self.df = None
        self.setLayout(layout)
        self.sid = sid

    def fire_remove_signal(self):
        self.remove.emit(self.sid)

    def match_sid(self, sid):
        return self.sid == sid

    def is_selected(self):
        return self.check.isChecked()

    def get_filepath(self):
        return filepath

    def update_label(self):
        if self.df is None:
            self.label.setText('{} | Not Loaded'.format(self.filepath))
        else:
            self.label.setText('{} | Rows: {}, Columns: {}'.format(self.filepath, self.df.shape[0], self.df.shape[1]))

    def get_df(self):
        if self.df is None:
            self.load()
            self.update_label()
        return self.df

    def load(self):
        self.df = None
        if self.filepath.endswith('.xlsx'):
            self.df = pd.read_excel(self.filepath)
        elif self.filepath.endwith('.csv'):
            self.df = pd.read_csv(self.filepath)
        else:
            raise Exception('The file must end either with xlsx or csv. {} ends with neither of them'.format(self.filepath))

if __name__ == '__main__':

    app = QApplication([])
    elem = RowElement('Test')
    elem.show()
    app.exec_()
