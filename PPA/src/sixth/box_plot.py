from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel

from .plot_widget import PlotWidget

class BoxPlot(QWidget):
    def __init__(self, df):
        super().__init__()
        
        self.df = df
        layout = QVBoxLayout(self)
        
        pl = PlotWidget(15, 15, 96)
        
        layout.addWidget(pl)

        self.pl = pl
        
        self.update()
        
    def update(self):
        self.pl.axes.clear()

        self.pl.axes.boxplot(self.df, labels = self.df.columns.tolist())

        self.pl.draw()


if __name__ == '__main__':
    import random
    import pandas as pd
    app = QApplication([])

    data = {}
    for name in ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']:
        data[name] = {'Tree {}'.format(idx): 10*random.random() for idx in range(100)}

    print(pd.DataFrame(data))
    ip = BoxPlot(pd.DataFrame(data))
    ip.show()


    app.exec_()
