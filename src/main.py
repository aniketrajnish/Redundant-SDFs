import sys
from PyQt5 import QtWidgets
from gui import VizWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    viz_window = VizWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()