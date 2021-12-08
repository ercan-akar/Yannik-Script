from PyQt5.QtWidgets import QWidget, QHBoxLayout, QComboBox, QStackedLayout, QLabel, QLineEdit, QApplication, QCheckBox

from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import pyqtSignal


# either a single number or a list of numbers
regex_decimal = r'([0-9]+(\.[0-9]*)?|([0-9]*)?\.[0-9]+)'
regex_int = r'[+-]?[0-9]+(e[+]?[0-9]+)?'
regex_float = r'[+-]?{}(e[+-]?{})?'.format(regex_decimal, regex_decimal)

regex_comma_separated_list = r'\s*{}\s*(,\s*{}\s*)*,?\s*'
regex_comma_separated_float_list = regex_comma_separated_list.format(regex_float, regex_float)
regex_comma_separated_int_list = regex_comma_separated_list.format(regex_int, regex_int)

'''
regex_int_list = '\[\s*{}(\s*,\s*{})*\s*\]'.format(regex_int, regex_int)
regex_int_span = '\[\s*{}\s*;\s*{}\s*;\s*{}\s*]'.format(regex_int, regex_int, regex_int)
regex_any_int = '({})|({})|({})'.format(regex_int, regex_int_list, regex_int_span)

regex_decimal = '([0-9]+(\.[0-9]*)?|([0-9]*)?\.[0-9]+)'
regex_float_list = '\[\s*{}(\s*,\s*{})*\s*\]'.format(regex_float, regex_float)
regex_float_span = '\[\s*{}\s*;\s*{}\s*;\s*{}\s*]'.format(regex_float, regex_float, regex_float)
regex_any_float = '({})|({})|({})'.format(regex_float, regex_float_list, regex_float_span)
'''

def get_type(desc):
    if desc['type'] == 'int':
        return int
    if desc['type'] == 'float':
        return float
    if desc['type'] == 'option-of-str':
        return 'option-of-str'
    raise Exception('Type {} is invalid!"'.format(desc['type']))


def create_hyperparam_widget(desc):
    if get_type(desc) == float or get_type(desc) == int:
        return NumericParam(desc)
    if get_type(desc) == 'option-of-str':
        return OptionParam(desc)

class NumericParam(QWidget):
    def __init__(self, desc):
        super().__init__()
        self.desc = desc

        # always shown
        selector = QComboBox()
        selector.addItems(['Single value', 'List of values', 'Range of values'])
        selector.currentIndexChanged.connect(self.selection_changed)

        # if single value
        sv_line = QLineEdit()
        if self.type_() == float:
            sv_line.setValidator(QRegExpValidator(QRegExp(regex_float)))
        elif self.type_() == int:
            sv_line.setValidator(QRegExpValidator(QRegExp(regex_int)))
        else:
            raise Exception('invalid type')
        sv_layout = QHBoxLayout()
        sv_layout.addWidget(sv_line)
        sv_layout.setContentsMargins(0, 0, 0, 0)
        sv_widget = QWidget()
        sv_widget.setLayout(sv_layout)

        # if list
        lv_line = QLineEdit()
        if self.type_() == float:
            lv_line.setValidator(QRegExpValidator(QRegExp(regex_comma_separated_float_list)))
        elif self.type_() == int:
            lv_line.setValidator(QRegExpValidator(QRegExp(regex_comma_separated_int_list)))
        else:
            raise Exception('invalid type')
        lv_layout = QHBoxLayout()
        lv_layout.addWidget(lv_line)
        lv_layout.setContentsMargins(0, 0, 0, 0)
        lv_widget = QWidget()
        lv_widget.setLayout(lv_layout)

        # if range
        min_label = QLabel('Minimum:')
        min_line = QLineEdit()
        max_label = QLabel('Maximum:')
        max_line = QLineEdit()
        step_label = QLabel('Step size:')
        step_line = QLineEdit()
        if self.type_() == float:
            min_line.setValidator(QRegExpValidator(QRegExp(regex_float)))
            max_line.setValidator(QRegExpValidator(QRegExp(regex_float)))
            step_line.setValidator(QRegExpValidator(QRegExp(regex_float)))
        elif self.type_() == int:
            min_line.setValidator(QRegExpValidator(QRegExp(regex_int)))
            max_line.setValidator(QRegExpValidator(QRegExp(regex_int)))
            step_line.setValidator(QRegExpValidator(QRegExp(regex_int)))
        else:
            raise Exception('invalid type')
        range_layout = QHBoxLayout()

        range_layout.addWidget(min_label)
        range_layout.addWidget(min_line)
        range_layout.addWidget(max_label)
        range_layout.addWidget(max_line)
        range_layout.addWidget(step_label)
        range_layout.addWidget(step_line)
        range_layout.setContentsMargins(0, 0, 0, 0)

        range_widget = QWidget()
        range_widget.setLayout(range_layout)

        option_layout = QStackedLayout()
        option_layout.addWidget(sv_widget)
        option_layout.addWidget(lv_widget)
        option_layout.addWidget(range_widget)
        option_layout.setContentsMargins(0, 0, 0, 0)

        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(selector)
        layout.addLayout(option_layout)

        self.setLayout(layout)

        self.elements = {'selector': selector, 'stacked': option_layout, 'single': sv_line, 'list': lv_line, 'range': {'min': min_line, 'max': max_line, 'step': step_line}}

        # make sure the UI is in a consistent state
        self.selection_changed()
        self.set_defaults()

    def set_defaults(self):
        default = self.desc['default']
        if default['type'] == 'single':
            self.elements['selector'].setCurrentIndex(0)
            self.elements['single'].setText(str(default['value']))

        if default['type'] == 'list':
            self.elements['selector'].setCurrentIndex(1)

            self.elements['list'].setText(str(default['values']).replace('[', '').replace(']', ''))
        if default['type'] == 'range':
            self.elements['selector'].setCurrentIndex(2)

            for e in ['min', 'max', 'step']:
                self.elements['range'][e].setText(default[e])

    def selection_changed(self, idx = 0):
        self.elements['stacked'].setCurrentIndex(idx) # self.elements['selector'].currentIndex())


    def read_single_value(self):
        # we cannot guarantee that the regex will filter out all possibly wrong values
        # also, this additionally handles the case that nothing was entered
        try:
            if self.desc['type'] == 'float':
                return [float(self.elements['single'].text())]
            else:
                return [int(self.elements['single'].text())]
        except:
            return None

    def read_list_values(self):
        # we cannot guarantee that the regex will filter out all possibly wrong values
        # also, this additionally handles the case that nothing was entered
        try:
            nums = self.elements['list'].text().split(',')
            if self.desc['type'] == 'float':    
                return [float(num) for num in nums if num != ''] # the if filters out a trailing comma
            else:    
                return [int(num) for num in nums if num != ''] # the if filters out a trailing comma
        except:
            return None

    def read_range_values(self):
        try:
            if self.desc['type'] == 'float':
                minimum = float(self.elements['range']['min'].text())
                maximum = float(self.elements['range']['max'].text())
                step = float(self.elements['range']['step'].text())
            else:
                minimum = int(self.elements['range']['min'].text())
                maximum = int(self.elements['range']['max'].text())
                step = int(self.elements['range']['step'].text())

            if step == 0 or step == 0.0 or minimum > maximum:
                return None

            nums = []
            if step > 0.0:
                nums.append(minimum)
            if step < 0.0:
                nums.append(maximum)
            while nums[-1] + step <= maximum and nums[-1] + step >= minimum:
                nums.append(nums[-1] + step)

            return nums
        except:
            return None

    def get_values(self):
        idx = self.elements['selector'].currentIndex()
        if idx == 0:
            return self.read_single_value()
        elif idx == 1:
            return self.read_list_values()
        elif idx == 2:
            return self.read_range_values()
        else:
            raise Exception('Unhandled case, this is a bug')

    def is_valid(self):
        return self.get_values() is not None

    def name(self):
        return self.desc['name']

    def type_(self):
        return get_type(self.desc)

    def internal_name(self):
        return self.desc['internal_name']


class OptionParam(QWidget):
    def __init__(self, desc):
        super().__init__()
        self.desc = desc

        self.boxes = {}
        for option in self.desc['options']:
            self.boxes[option] = QCheckBox(option)

        layout = QHBoxLayout()
        for box in self.boxes:
            layout.addWidget(self.boxes[box])

        self.setLayout(layout)
        self.set_defaults()

    def set_defaults(self):
        default = self.desc['default']
        if default['type'] == 'single':
            for box in self.boxes:
                self.boxes[box].setChecked(box == default['value'])

        if default['type'] == 'list':
            for box in self.boxes:
                self.boxes[box].setChecked(box in default['values'])

    def get_values(self):
        return [self.boxes[box].text() for box in self.boxes if self.boxes[box].isChecked()]

    def name(self):
        return self.desc['name']

    def type_(self):
        return get_type(self.desc)

    def internal_name(self):
        return self.desc['internal_name']


if __name__ == '__main__':
    app = QApplication([])
    param = NumericParam({'name': 'Test', 'type': 'float', 'internal_name': 'test'})
    param.show()
    app.exec_()
