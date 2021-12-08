import pandas as pd
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QLabel, QWidget, QInputDialog
from PyQt5.QtCore import pyqtSignal
from .row_box import RowBox
from ..nav_row import NavRow

class FirstPage(QWidget):
    done = pyqtSignal()

    def __init__(self):
        super().__init__()

        primary_layout = QVBoxLayout(self)

        secondary_layout = QHBoxLayout()
        secondary_layout.setContentsMargins(0, 0, 0, 0)

        tertiary_layout = QVBoxLayout()
        tertiary_layout.setContentsMargins(0, 0, 0, 0)
        
        quad_layout = QHBoxLayout()
        quad_layout.setContentsMargins(0, 0, 0, 0)

        # first row
        browse_btn = QPushButton('Browse')
        browse_btn.clicked.connect(self.select_files)
        clr_btn = QPushButton('Clear')
        clr_btn.clicked.connect(self.clear)
        load_btn = QPushButton('Load | Merge')
        load_btn.clicked.connect(self.load_merge)

        tertiary_layout.addWidget(browse_btn)
        tertiary_layout.addWidget(clr_btn)
        tertiary_layout.addWidget(load_btn)
        tertiary_layout.addStretch()

        rb = RowBox()
        rb.count_changed.connect(self.update_buttons)

        secondary_layout.addWidget(rb)
        secondary_layout.addLayout(tertiary_layout)

        # second row

        merge_label = QLabel('No Data Loaded')
        merge_save_button = QPushButton('Export')
        merge_save_button.clicked.connect(self.export_merged)
        merge_del_button = QPushButton('Delete')
        merge_del_button.clicked.connect(self.remove_merged)

        quad_layout.addWidget(merge_label)
        quad_layout.addStretch()
        quad_layout.addWidget(merge_save_button)
        quad_layout.addWidget(merge_del_button)

        nav_row = NavRow(0, 5)
        nav_row.enable_next(False)
        nav_row.next_.connect(self.done.emit)

        primary_layout.addLayout(secondary_layout)
        primary_layout.addLayout(quad_layout)
        primary_layout.addWidget(nav_row)

        # internal state

        self.merged = None
        self.merged_label = merge_label
        self.export_button = merge_save_button
        self.in_files = rb
        self.merge_button = load_btn
        self.del_button = merge_del_button
        self.nav_row = nav_row
        self.update_buttons()

    def update_buttons(self):
        self.export_button.setEnabled(self.merged is not None)
        self.merge_button.setEnabled(self.in_files.count() > 0)
        self.del_button.setEnabled(self.merged is not None)

    def update_merged(self):
        if self.merged is None:
            self.merged_label.setText('No Data Loaded')
        else:
            self.merged_label.setText('The merged Dataset has {} rows and {} columns'.format(self.merged.shape[0], self.merged.shape[1]))

    def select_files(self):
        open_dialog = QFileDialog()
        open_dialog.setFileMode(QFileDialog.ExistingFiles)
        open_dialog.setNameFilters(['Excel Sheet (*.xlsx)', 'CSV Table (*.csv)'])
        if open_dialog.exec_():
            files = open_dialog.selectedFiles()
            self.in_files.add_items(files)
        self.update_buttons()

    def load_merge(self):
        if self.in_files.count() > 1:
            item, ok = QInputDialog.getItem(self, "Merge Mode", "Merge Mode: ", ['Horizontal', 'Vertical'], 0, False)
            if item and ok:
                # perform merge
                if item == 'Horizontal':
                    self.merged = pd.concat(self.in_files.get_dataframes(), axis = 1)
                elif item == 'Vertical':
                    self.merged = pd.concat(self.in_files.get_dataframes(), axis = 0)
                self.update_buttons()
                self.update_merged()
                self.nav_row.enable_next(True)
            else:
                pass
        else:
            self.merged = self.in_files.get_dataframes()[0]
            self.update_buttons()
            self.update_merged()
            self.nav_row.enable_next(True)

    def clear(self):
        self.in_files.clear()
        self.update_buttons()

    def export_merged(self):
        # cool abbreviation
        filename = QFileDialog.getSaveFileName(self, "Save file", "", ".xlsx")
        print(filename)
        if filename[0].endswith('.xlsx'):
            filename = filename[0]
        else:
            filename = filename[0]+filename[1]
        print(filename)
        self.merged.to_excel(filename, index=False)

    def remove_merged(self):
        self.merged = None
        self.update_buttons()
        self.update_merged()
        self.nav_row.enable_next(False)

    def get_merged(self):
        return self.merged

if __name__ == '__main__':
    app = QApplication([])
    main_window = FirstPage()
    main_window.show()
    app.exec_()
