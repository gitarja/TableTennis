
from PyQt5 import QtWidgets

def showErrorMessage(message):
    error_dialog = QtWidgets.QMessageBox()
    error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
    error_dialog.setText(message)
    error_dialog.exec()