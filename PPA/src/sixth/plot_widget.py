import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# taken from the matplotlib documentation on how to use matplotlib with qt
class PlotWidget(FigureCanvasQTAgg):
    def __init__(self, width, height, dpi):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg.__init__(self, self.figure)
