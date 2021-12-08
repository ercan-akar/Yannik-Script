from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QPushButton, QGridLayout, QLabel, QGroupBox

from PyQt5.QtCore import pyqtSignal

from ..nav_row import NavRow

from .bar_plot import ImportancePlot
from .score_plot import ScorePlot
from .box_plot import BoxPlot
# nav row might not be needed

class SixthPage(QWidget):
    done = pyqtSignal()
    back = pyqtSignal()
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        hparams = QGridLayout()
        hparams_box = QGroupBox('Best Hyperparameters')
        hparams_box.setLayout(hparams)

        metrics = QGridLayout()
        # metrics.setContentsMargins(0, 0, 0, 0)
        metrics_box = QGroupBox('Model statistics')
        metrics_box.setLayout(metrics)


        plots_buttons = QVBoxLayout()
        plots_buttons.setContentsMargins(0, 0, 0, 0)

        nav_row = NavRow(5,6)
        nav_row.next_.connect(self.done.emit)
        nav_row.back.connect(self.back.emit)

        layout.addWidget(hparams_box)
        layout.addWidget(metrics_box)
        layout.addStretch()
        layout.addStretch()
        layout.addLayout(plots_buttons)
        layout.addWidget(nav_row)

        self.hparams = hparams
        self.metrics = metrics
        self.plots_buttons = plots_buttons
        self.results = {}
        self.analysis = None
        self.plots = []

    def rebuild_ui(self):
        self.clear()
        if self.analysis is None:
            return

        outputs = self.results['output']
        
        for (row, param) in enumerate(self.results['hyperparams']):
            label = QLabel(self.analysis.get_common_name_of_hyperparam(param))
            data = QLabel(str(self.results['hyperparams'][param]))
            print(self.analysis.get_common_name_of_hyperparam(param))
            print(str(self.results['hyperparams'][param]))
            self.hparams.addWidget(label, row, 0)
            self.hparams.addWidget(data, row, 1)
            
        for (row, metric) in enumerate(outputs['metrics']):
            metric = outputs['metrics'][metric]
            # for now, ignore style, it was an artifact
            label = QLabel(metric['name'])
            data = QLabel(str(metric['value']))
            self.metrics.addWidget(label, row, 0)
            self.metrics.addWidget(data, row, 1)
        
        for plot in outputs['plots']:
            plot = outputs['plots'][plot]
            btn = QPushButton(plot['name'])

            if plot['style'] == 'bar_h':
                plt_widget = ImportancePlot(plot['value'])
            elif plot['style'] == 'scatter':
                plt_widget = ScorePlot(plot['value'])
            elif plot['style'] == 'box':
                plt_widget = BoxPlot(plot['value'])
            else:
                raise Exception('plot style {} is unknown'.format(plot['style']))

            self.plots_buttons.addWidget(btn)
            self.plots.append(plt_widget)
            btn.clicked.connect((lambda plt: (lambda: plt.show()))(plt_widget))

    def clear(self):
        while self.metrics.count() > 0:
            self.metrics.takeAt(0).widget().setParent(None)
        while self.plots_buttons.count() > 0:
            self.plots_buttons.takeAt(0).widget().setParent(None)

    def set_results(self, results_dict, analysis):
        self.results = results_dict
        self.analysis = analysis
        self.plots = []
        self.rebuild_ui()

if __name__ == "__main__":
    from sklearn import datasets
    import pandas as pd
    iris = datasets.load_iris()

    # turn iris into a pandas df to be closer to what we ant to do

    data_df = pd.DataFrame(iris.data, columns = iris.feature_names)
    target_df = pd.DataFrame(iris.target, columns = ['Target'])

    total_df = pd.concat([data_df, target_df], axis=1)

    total_df['Sample Name'] = pd.Series(['Sample {}'.format(idx+1) for idx in range(total_df.shape[0])])

    inp = {
        'df': total_df,
        'columns': {
            'time': None,
            'batch': 'Sample Name',
            'x': ['sepal length (cm)', 'sepal width (cm)',  'petal length (cm)',  'petal width (cm)'],
            'y': 'Target'
        },
        'data_type': 'batch',
        'analysis_type': 'regression',
        'analysis_method': 'PLS',
        'hyperparams': {
            'n_estimators': range(1,100,25),
            'max_depth': range(2,7)            
        }
    }

    from rf import run
    
    result = run(inp)
    print(result)

    class dummy:
        def get_common_name_of_hyperparam(self, param):
            if param == 'n_components': return 'Number of components'
            if param == 'n_estimators': return 'Number of estimators'
            if param == 'max_depth': return 'Maximum tree depth'

    app = QApplication([])
    widget = SixthPage()
    widget.set_results(result, dummy())
    widget.show()
    app.exec()
