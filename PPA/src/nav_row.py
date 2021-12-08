from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QHBoxLayout, QCheckBox
from PyQt5.QtCore import pyqtSignal

class NavRow(QWidget):
    next_ = pyqtSignal()
    back = pyqtSignal()

    def __init__(self, idx, max_idx):
        super().__init__()

        layout = QHBoxLayout()
        back_btn = QPushButton('Previous')
        back_btn.setEnabled(idx > 0)
        back_btn.clicked.connect(self.back.emit)
        text = QLabel('Page {} of {}'.format(idx+1, max_idx))
        next_btn = QPushButton('Next')
        next_btn.setEnabled(idx < max_idx -1 )
        next_btn.clicked.connect(self.next_.emit)

        layout.addWidget(back_btn)
        layout.addStretch()
        layout.addWidget(text)
        layout.addStretch()
        layout.addWidget(next_btn)

        self.setLayout(layout)

        self.idx = idx
        self.max_idx = max_idx
        self.next_btn = next_btn

    def enable_next(self, en):
        self.next_btn.setEnabled(en and self.idx < self.max_idx - 1)

if __name__ == '__main__':

    app = QApplication([])
    elem = NavRow(0, 5)
    elem.show()
    app.exec_()
