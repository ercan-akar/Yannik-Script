from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel

from .plot_widget import PlotWidget

# This could be used for importance plots, where the x axis would be different features
# also, it could be used for contribution plots
# also for plotting R2, Q2 maybe?
# but we need to be able to distinguish between those two, as their behaviour is different
class ImportancePlot(QWidget):
    def __init__(self, df):
        super().__init__()
        # For now, I assume the features to be the columns and the rows are different possible methods or something like that

        self.df = df
        layout = QVBoxLayout(self)
        sub_layout = QHBoxLayout()

        pl = PlotWidget(15, 15, 72)

        y_label = QLabel('Importance method')

        y_combobox = QComboBox()
        y_combobox.addItems(map(lambda x: str(x), self.df.index.tolist()))
        y_combobox.currentIndexChanged.connect(lambda y: self.axis_changed())

        sub_layout.addWidget(y_label)
        sub_layout.addWidget(y_combobox)

        layout.addWidget(pl)
        layout.addLayout(sub_layout)

        self.pl = pl
        self.y = y_combobox
        self.y_idx = 0

        self.mask = [True for _ in range(self.df.shape[1])]
        self.axis_changed()

    def update(self):
        self.pl.axes.clear()

        y = self.df.iloc[self.y_idx, :]
        self.pl.axes.bar(range(self.df.shape[1]), y, tick_label=self.df.columns.tolist())
        self.pl.axes.set_ylabel(self.y.currentText())

        self.pl.draw()

    def axis_changed(self):
        self.y_idx = self.y.currentIndex()
        self.update()

if __name__ == '__main__':
    import random
    import pandas as pd
    app = QApplication([])

    data = {}
    for name in ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']:
        data[name] = {'Method 1': 10*random.random(), 'Method 2': 10*random.random()}

    ip = ImportancePlot(pd.DataFrame(data))
    ip.show()


    app.exec_()
