from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QListWidget
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np

from .plot_widget import PlotWidget

import pandas as pd

class ScorePlot(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        layout = QHBoxLayout(self)
        plot_layout = QVBoxLayout()
        sub_layout = QHBoxLayout()

        pl = PlotWidget(15, 15, 72)

        x_label = QLabel('X Axis')
        y_label = QLabel('Y Axis')

        x_combobox = QComboBox()
        x_combobox.addItems(map(lambda x: str(x), self.df.columns.tolist()))
        x_combobox.currentIndexChanged.connect(lambda x: self.axis_changed())

        y_combobox = QComboBox()
        y_combobox.addItems(map(lambda x: str(x), self.df.columns.tolist()))
        y_combobox.currentIndexChanged.connect(lambda y: self.axis_changed())

        selection_list = QListWidget()

        sub_layout.addWidget(x_label)
        sub_layout.addWidget(x_combobox)
        sub_layout.addStretch()
        sub_layout.addWidget(y_label)
        sub_layout.addWidget(y_combobox)

        plot_layout.addWidget(pl)
        plot_layout.addLayout(sub_layout)

        layout.addLayout(plot_layout)
        layout.addWidget(selection_list)

        self.pl = pl
        self.x = x_combobox
        self.y = y_combobox
        self.x_idx = 0
        self.y_idx = 0
        self.selection_list = selection_list

        self.lasso = LassoSelector(self.pl.axes, onselect=self.onselect)

        self.mask = [True for _ in range(self.df.shape[0])]
        self.axis_changed()

    def update(self):
        self.selection_list.clear()
        self.pl.axes.clear()

        x = self.df.iloc[:, self.x_idx]
        y = self.df.iloc[:, self.y_idx]

        self.points = self.pl.axes.scatter(x, y)
        self.coordinates = self.points.get_offsets()

        colors = np.tile(self.points.get_facecolors(), (len(self.points.get_offsets()), 1))
        colors[:, -1] = 0.125
        colors[self.mask, -1] = 1.0
        self.points.set_facecolors(colors)

        self.pl.axes.set_xlabel(self.x.currentText())
        self.pl.axes.set_ylabel(self.y.currentText())

        selected_batches = self.df.index[self.mask]
        self.selection_list.addItems(list(map(str, selected_batches)))

        self.pl.draw()

    def axis_changed(self):
        self.x_idx = self.x.currentIndex()
        self.y_idx = self.y.currentIndex()
        self.update()

    def onselect(self, verts):
        path = Path(verts)
        self.selection_changed(path.contains_points(self.coordinates))
        # self.pl.figure.canvas.draw_idle()

    def selection_changed(self, mask):
        self.mask = mask
        self.update()

if __name__ == '__main__':
    import random
    app = QApplication([])

    names = []
    for i in range(100000):
        name = ''
        for j in range(10):
            name += chr(random.randint(60,100))
        names.append(name)

    data = {}
    for name in names:
        data[name] = {'First Component': 10*random.random()-5, 'Second Component': 10*random.random()-5}

    print(pd.DataFrame(data))
    sp = ScorePlot(pd.DataFrame(data))
    sp.show()

    app.exec_()
