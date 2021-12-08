from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QScrollArea, QListWidget, QListWidgetItem, QAbstractItemView
from PyQt5.QtCore import pyqtSignal

from .row_element import RowElement

class RowBox(QWidget):
    count_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)

        listview = QListWidget()
        listview.setDragDropMode(QAbstractItemView.InternalMove)

        scroll = QScrollArea()
        # scroll.setWidgetResizable(True)

        scrollContainer = QWidget(scroll)
        scrollLayout = QVBoxLayout()
        scrollContainer.setLayout(scrollLayout)

        layout.addWidget(listview)

        self.listview = listview
        self.scrollLayout = scrollLayout
        self.sid_counter = 0

    def add_item(self, item, emit=True):
        self.sid_counter += 1
        re = RowElement(self.sid_counter, item)
        re.remove.connect(self.remove_item)
        # https://stackoverflow.com/questions/25187444/pyqt-qlistwidget-custom-items
        lwi = QListWidgetItem()
        lwi.setSizeHint(re.sizeHint())
        self.listview.addItem(lwi)
        self.listview.setItemWidget(lwi, re)
        if emit:
            self.count_changed.emit()

    def add_items(self, items):
        for item in items:
            self.add_item(item, emit=False)
        self.count_changed.emit()

    def remove_item(self, sid):
        print('removing {}'.format(sid))
        items = self.get_items()
        for (idx, item) in enumerate(items):
            if item.match_sid(sid):
                # the documentation says that Qt will not take care of cleaning up the element and one has to do it manually
                # however, as all references to the object are dropped quickly, the garbage collector should take care of that for us
                self.listview.takeItem(idx)
                self.count_changed.emit()

    def count(self):
        return len(self.get_items())

    def selected_count(self):
        return len(self.get_selected_items())

    def get_items(self):
        return [self.listview.itemWidget(self.listview.item(idx)) for idx in range(self.listview.count())]

    def get_selected_items(self):
        return [elem for elem in self.get_items() if elem.is_selected()]

    def get_dataframes(self):
        return [elem.get_df() for elem in self.get_selected_items()]

    def clear(self):
        self.listview.clear()

if __name__ == '__main__':

    app = QApplication([])
    elem = RowBox()
    elem.add_item('Test 01')
    elem.add_item('Test 02')
    elem.add_item('Test 03')
    elem.show()
    app.exec_()
