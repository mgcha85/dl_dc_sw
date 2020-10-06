from ViewController import ViewController
from PyQt5.QtWidgets import *
import sys


class Main:
    def __init__(self):
        self.viewController = ViewController()


if __name__ == '__main__':
    print('메인이 실행되었습니다.')
    app = QApplication(sys.argv)
    myApp = Main()
    myApp.viewController.show()
    app.exec_()
