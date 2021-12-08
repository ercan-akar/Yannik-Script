from PyQt5.QtWidgets import QApplication
from src.main_window import MainWindow

app = QApplication([])
mw = MainWindow()
mw.show()
app.exec_()
