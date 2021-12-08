from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QAbstractItemView, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import pyqtSignal

class ListSelection(QWidget):
    count_changed = pyqtSignal()

    def __init__(self, cols):
        super().__init__()

        main_layout = QHBoxLayout()
        # purely aesthetic. By nesting layouts, the inner layout normally becomes smaller. we don't want that
        main_layout.setContentsMargins(0, 0, 0, 0)

        left_list = QListWidget()
        left_list.addItems(cols)
        left_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # https://www.unicode.org/charts/nameslist/n_2190.html
        move_right = QPushButton('Add \u2192')
        move_right.clicked.connect(self.move_right)
        move_left = QPushButton('\u2190 Remove')
        move_left.clicked.connect(self.move_left)
        reset = QPushButton('Reset')
        reset.clicked.connect(self.reset)
        b_layout = QVBoxLayout()
        b_layout.addWidget(move_right)
        b_layout.addWidget(move_left)
        b_layout.addWidget(reset)
        b_widget = QWidget()
        b_widget.setLayout(b_layout)

        right_list = QListWidget()
        right_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        main_layout.addWidget(left_list)
        main_layout.addWidget(b_widget)
        main_layout.addWidget(right_list)

        self.left_list = left_list
        self.right_list = right_list

        self.cols = cols

        self.setLayout(main_layout)

    def move(self, from_, to):
        selected_indices = []
        for idx in range(from_.count()):
            if from_.item(idx).isSelected():
                selected_indices.append(idx)

        for idx in reversed(selected_indices):
            item = from_.takeItem(idx)
            to.insertItem(0, item.text())

        if len(selected_indices) > 0:
            self.count_changed.emit()

    def move_right(self):
        '''
        selected = self.left_list.selectedItems()
        for item in selected:
            self.left_list.removeItemWidget(item)
            self.right_list.addItem(item.text())
        '''
        # the above makes it hard to remove an item by index, so just implement a custom selection logic
        self.move(self.left_list, self.right_list)

    def move_left(self):
        self.move(self.right_list, self.left_list)
        print(self.get_selected())

    def reset(self):
        self.left_list.clear()
        self.left_list.addItems(self.cols)
        self.right_list.clear()

    def get_selected(self):
        return list(map(lambda idx: self.right_list.item(idx).text(), range(self.right_list.count())))

if __name__ == '__main__':
    app = QApplication([])
    ls = ListSelection(['a', 'b', 'c'])
    ls.show()
    app.exec_()
