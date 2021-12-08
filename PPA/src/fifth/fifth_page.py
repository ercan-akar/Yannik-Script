from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout, QVBoxLayout, QHBoxLayout, QListWidget, QGroupBox
from PyQt5.QtCore import pyqtSignal

from ..nav_row import NavRow

class FifthPage(QWidget):
    done = pyqtSignal()
    back = pyqtSignal()
    def __init__(self):
        super().__init__()

        primary_layout = QVBoxLayout()

        settings_layout = QGridLayout()
        settings_layout.setContentsMargins(0, 0, 0, 0)

        nav_row = NavRow(4, 6)
        nav_row.next_.connect(self.done.emit)
        nav_row.back.connect(self.back.emit)

        primary_layout.addLayout(settings_layout)
        primary_layout.addStretch()
        primary_layout.addWidget(nav_row)

        self.setLayout(primary_layout)

        self.settings_layout = settings_layout

    def clear(self):
        while self.settings_layout.count() > 0:
            self.settings_layout.takeAt(0).widget().setParent(None)

    def set_summary(self, df, data_type, analysis_type, analysis_method, columns, hyperparams):
        timestamp_column = columns['time']
        batch_id_column = columns['batch']
        x_columns = [columns['x']] if type(columns['x']) == str else columns['x']
        y_columns = [columns['y']] if type(columns['y']) == str else columns['y']

        self.clear()

        # Dataframe data
        df_box = QGroupBox('Dataframe metrics: ')
        df_layout = QVBoxLayout()
        df_box.setLayout(df_layout)

        # df_label = QLabel('')
        df_info = QLabel('Rows: {}; Columns: {}'.format(df.shape[0], df.shape[1]))
        df_layout.addWidget(df_info)

        # Analysis
        analysis_box = QGroupBox('Analysis')
        analysis_layout = QGridLayout()
        analysis_box.setLayout(analysis_layout)

        data_label = QLabel('Data type: ')
        data_info = QLabel(data_type)

        type_label = QLabel('Analysis type: ')
        type_info = QLabel(analysis_type)

        method_label = QLabel('Analysis method: ')
        method_info = QLabel(analysis_method.get_name())

        analysis_layout.addWidget(data_label, 0, 0)
        analysis_layout.addWidget(data_info, 0, 1)
        analysis_layout.addWidget(type_label, 1, 0)
        analysis_layout.addWidget(type_info, 1, 1)
        analysis_layout.addWidget(method_label, 2, 0)
        analysis_layout.addWidget(method_info, 2, 1)

        # Columns
        column_box = QGroupBox('Columns')
        column_layout = QGridLayout()
        column_box.setLayout(column_layout)

        timestamp_label = QLabel('Timestamp column: ')
        timestamp_info = QLabel(timestamp_column)

        batch_id_label =  QLabel('Batch ID column: ')
        batch_id_info = QLabel(batch_id_column)

        x_label = QLabel('X columns: ')
        x_info = QListWidget()
        x_info.addItems(x_columns)

        y_label = QLabel('Y columns: ')
        y_info = QListWidget()
        y_info.addItems(y_columns)

        column_layout.addWidget(timestamp_label, 0, 0)
        column_layout.addWidget(timestamp_info, 0, 1)
        column_layout.addWidget(batch_id_label, 1, 0)
        column_layout.addWidget(batch_id_info, 1, 1)
        column_layout.addWidget(x_label, 2, 0)
        column_layout.addWidget(x_info, 2, 1)
        column_layout.addWidget(y_label, 3, 0)
        column_layout.addWidget(y_info, 3, 1)

        # Hyperparameters
        hyper_box = QGroupBox('Hyperparameters')
        hyper_layout = QGridLayout()
        hyper_box.setLayout(hyper_layout)

        labels = []
        infos = []
        search_steps = 1
        for hyperparam in hyperparams:
            param_label = QLabel(hyperparam)
            param_info = QLabel(str(hyperparams[hyperparam]))
            labels.append(param_label)
            infos.append(param_info)

            if type(hyperparams[hyperparam]) == list:
                search_steps *= len(hyperparams[hyperparam])

        for (idx, (label, info)) in enumerate(zip(labels, infos)):
            hyper_layout.addWidget(label, idx, 0)
            hyper_layout.addWidget(info, idx, 1)

        run_box = QGroupBox('Run metrics')
        run_layout = QGridLayout()
        run_box.setLayout(run_layout)

        search_label = QLabel('Amount of parameter combinations: ')
        search_info = QLabel(str(search_steps))
        cv_label = QLabel('Performed cross validations: ')
        cv_info = QLabel(str(5))
        total_runs_label = QLabel('<b>Total runs:</b> ')
        total_runs_info = QLabel('<b>{}</b>'.format(5*search_steps))


        run_layout.addWidget(search_label, 0, 0)
        run_layout.addWidget(search_info, 0, 1)
        run_layout.addWidget(cv_label, 1, 0)
        run_layout.addWidget(cv_info, 1, 1)
        run_layout.addWidget(total_runs_label, 2, 0)
        run_layout.addWidget(total_runs_info, 2, 1)

        self.settings_layout.addWidget(df_box)
        self.settings_layout.addWidget(analysis_box)
        self.settings_layout.addWidget(column_box)
        self.settings_layout.addWidget(hyper_box)
        self.settings_layout.addWidget(run_box)



if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import pandas as pd

    app = QApplication([])

    f = FifthPage()
    f.set_summary(pd.DataFrame(), 'time', 'classification', 'PCA', 'Time [UTC]', 'Batch', ['T_in', 'T_out', 'Flow_Rate'], ['Quality'], {'Number of components: ': [2, 3, 5], 'other parameter': ['ReLU', 'sigmoid']})
    f.show()

    app.exec_()
