from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy

# Qt5_Backends::FigureCanvas继承自QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use('Qt5Agg')


def plot(figure):
    """ 传入UnitPlot的回调函数 """
    import numpy as np

    x = np.linspace(-3, 5, 3000)
    y = x ** 2
    axes = figure.add_subplot(111)
    axes.plot(x, y)


class UnitPlot(QWidget):
    def __init__(self, parent, name=""):
        super().__init__(parent)

        self.figure = plt.figure()
        # self.axes = self.figure.add_subplot(211)
        # self.axes = self.figure.add_axes([0.1,0.1,0.8,0.8])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        # 创建Matplotlib自带的工具栏
        # self.toolbar = NavigationToolbar(self.canvas, self)
        # self.toolbar.hide()

        # set the layout
        layout = QVBoxLayout()
        # layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        if name:
            plt.title(name)

    def show(self, func_plot):
        # 绘制图形
        func_plot(self.figure)
        self.canvas.draw()

    def home(self):
        self.toolbar.home()
    def zoom(self):
        self.toolbar.zoom()
    def pan(self):
        self.toolbar.pan()
