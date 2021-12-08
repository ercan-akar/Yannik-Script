from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QVBoxLayout, QLineEdit, QLabel, QGridLayout

from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import pyqtSignal

from ..nav_row import NavRow
from .hyperparam import create_hyperparam_widget

# either a single number or a list of numbers
'''
regex_int = '[+-]?[0-9]+(e[+]?[0-9]+)?'
regex_int_list = '\[\s*{}(\s*,\s*{})*\s*\]'.format(regex_int, regex_int)
regex_int_span = '\[\s*{}\s*;\s*{}\s*;\s*{}\s*]'.format(regex_int, regex_int, regex_int)
regex_any_int = '({})|({})|({})'.format(regex_int, regex_int_list, regex_int_span)

regex_decimal = '([0-9]+(\.[0-9]*)?|([0-9]*)?\.[0-9]+)'
regex_float = '[+-]?{}(e[+-]?{})?'.format(regex_decimal, regex_decimal)
regex_float_list = '\[\s*{}(\s*,\s*{})*\s*\]'.format(regex_float, regex_float)
regex_float_span = '\[\s*{}\s*;\s*{}\s*;\s*{}\s*]'.format(regex_float, regex_float, regex_float)
regex_any_float = '({})|({})|({})'.format(regex_float, regex_float_list, regex_float_span)

print(regex_any_int)
print(regex_any_float)
'''

class FourthPage(QWidget):
    done = pyqtSignal()
    back = pyqtSignal()
    def __init__(self):
        super().__init__()
        primary_layout = QVBoxLayout()

        param_layout = QGridLayout()
        param_layout.setContentsMargins(0, 0, 0, 0)

        nav_row = NavRow(3, 5)
        nav_row.next_.connect(self.done.emit)
        nav_row.back.connect(self.back.emit)

        primary_layout.addLayout(param_layout)
        primary_layout.addStretch()
        primary_layout.addStretch()
        primary_layout.addStretch()
        primary_layout.addStretch()
        primary_layout.addWidget(nav_row)

        self.setLayout(primary_layout)

        self.nav_row = nav_row
        self.param_layout = param_layout

        self.analysis = None
        self.inputs = {}

    def set_analysis(self, analysis):
        if analysis == self.analysis:
            return

        self.analysis = analysis
        self.inputs = []
        while self.param_layout.count() > 0:
            self.param_layout.takeAt(0).widget().setParent(None)

        for (idx, parameter) in enumerate(analysis.get_hyperparameters()):
            label = QLabel('{} ({}): '.format(parameter['name'], parameter['type']))
            hp = create_hyperparam_widget(parameter)
            self.param_layout.addWidget(label, idx, 0)
            self.param_layout.addWidget(hp, idx, 1, 1, 3)
            self.inputs.append(hp)
            '''
            p_name     = parameter['name']
            p_type     = parameter['type']
            p_internal = parameter['internal_name']
            p_min      = parameter['minimum'] if p_type in ['float', 'int'] and 'minimum' in parameter.keys() else None

            label = QLabel(p_name)
            edit  = QLineEdit()
            if p_type == 'float':
                edit.setValidator(QRegExpValidator(QRegExp(regex_any_float)))
            if p_type == 'int':
                edit.setValidator(QRegExpValidator(QRegExp(regex_any_int)))
            self.inputs.update({p_internal: edit})

            self.param_layout.addWidget(label, idx, 0)
            self.param_layout.addWidget(edit, idx, 1)
            '''

    def get_parameters_for_summary(self):
        params = {}
        for inp in self.inputs:
            params.update({inp.name(): inp.get_values()})
        return params

    def get_parameters_for_analysis(self):
        params = {}
        for inp in self.inputs:
            params.update({inp.internal_name(): inp.get_values()})
        return params


if __name__ == '__main__':
    app = QApplication([])
    page = FourthPage()
    page.set_analysis()
    page.show()
    app.exec_()
