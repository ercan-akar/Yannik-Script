import pandas as pd
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QLabel, QMainWindow, QWidget, QFileDialog, QInputDialog, QStackedLayout

import importlib

from .first.first_page import FirstPage
from .second.second_page import SecondPage
from .third.third_page import ThirdPage
from .fourth.fourth_page import FourthPage
from .fifth.fifth_page import FifthPage
from .sixth.sixth_page import SixthPage

from .analysis import read_from_file as load_analyses

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.analyses = load_analyses('spec/options.json')

        page_widget = QWidget()
        page_layout = QStackedLayout()
        page_widget.setLayout(page_layout)

        first_page = FirstPage()
        first_page.done.connect(self.go_forward)
        page_layout.addWidget(first_page)

        second_page = SecondPage(self.analyses)
        second_page.done.connect(self.go_forward)
        second_page.back.connect(self.go_back)
        page_layout.addWidget(second_page)

        third_page = ThirdPage()
        third_page.done.connect(self.go_forward)
        third_page.back.connect(self.go_back)
        page_layout.addWidget(third_page)

        fourth_page = FourthPage()
        fourth_page.done.connect(self.go_forward)
        fourth_page.back.connect(self.go_back)
        page_layout.addWidget(fourth_page)

        fifth_page = FifthPage()
        fifth_page.done.connect(self.go_forward)
        fifth_page.back.connect(self.go_back)
        page_layout.addWidget(fifth_page)

        sixth_page = SixthPage()
        #sixth_page.done.connect(self.go_forward)
        #sixth_page.back.connect(self.go_back)
        page_layout.addWidget(sixth_page)


        primary_layout = QVBoxLayout()
        primary_layout_container = QWidget()
        primary_layout_container.setLayout(primary_layout)
        primary_layout.addWidget(page_widget)

        self.setCentralWidget(primary_layout_container)

        self.pages = [first_page, second_page, third_page, fourth_page, fifth_page, sixth_page]

        self.stack = page_layout
        self.current_idx = 0

    def go_forward(self):
        # the idea is that we can do other stuff like checking selections between page changes

        # after data choosing
        if self.current_idx == 0:
            pass

        # after method choosing
        if self.current_idx == 1:
            self.pages[2].set_cols_analysis_and_parameters(self.pages[0].get_merged().columns.tolist(), self.pages[1].get_analysis(), self.pages[1].get_time_or_batch(), self.pages[1].get_classification_or_regression())

        # after column choosing
        if self.current_idx == 2:
            self.pages[3].set_analysis(self.pages[1].get_analysis())

        # after choosing hyperparams
        if self.current_idx == 3:
            self.pages[4].set_summary(**self.create_summary())

        # after viewing the summary
        if self.current_idx == 4:
            analysis = self.pages[1].get_analysis()

            analysis_module =  importlib.import_module('src.analyses.{}'.format(analysis.get_module_name()))

            result = getattr(analysis_module, analysis.get_function_name())(self.create_summary(user_facing = False))
            
            self.pages[5].set_results(result, self.pages[1].get_analysis())

        self.current_idx += 1
        self.stack.setCurrentIndex(self.current_idx)

    def go_back(self):
        self.current_idx -= 1
        self.stack.setCurrentIndex(self.current_idx)

    def create_summary(self, user_facing=True):
        df = self.pages[0].get_merged()

        data_type = self.pages[1].get_time_or_batch()
        analysis_type = self.pages[1].get_classification_or_regression()
        method = self.pages[1].get_analysis()

        columns = self.pages[2].get_variables()

        hyperparams = self.pages[3].get_parameters_for_summary() if user_facing else self.pages[3].get_parameters_for_analysis()

        return {
            'df': df,
            'data_type': data_type,
            'analysis_type': analysis_type,
            'analysis_method': method,
            'columns': columns,
            'hyperparams': hyperparams
        }

if __name__ == '__main__':
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()
