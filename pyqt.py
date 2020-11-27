
###############################################################################
# Name:         pyqt5_ext
# Purpose:      A simple packaging of Qt5 API
# Author:       Bright Li
# Modified by:
# Created:      2020-11-27
# Version:      [1.1.2]
# RCS-ID:       $$
# Copyright:    (c) Bright Li
# Licence:
###############################################################################

import os
import math
import copy

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QPoint, QRectF

from PyQt5.QtWidgets import (QLayout, QGridLayout, QHBoxLayout, QVBoxLayout, QGroupBox
                            , QLabel, QCheckBox, QLineEdit, QRadioButton, QSlider, QSpinBox
                            , QPushButton, QButtonGroup, QScrollArea, QTabWidget)

from PyQt5.QtWidgets import QMenu, QAction, QStatusBar
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtWidgets import QMessageBox

from PyQt5.QtGui import (QPixmap, QImage, QIcon, QPalette, QCursor, QPainter
                        , QPen, QBrush, QColor, QPainterPath)

#####################################################################
# Main Entrance
#####################################################################

# class QApp:
def run_qtapp(MainWndClass):
    app = QApplication([])
    try:
        mwnd = MainWndClass(None)
        # qss渲染
        mwnd.setProperty("class", "bkg")  # for qss
        qss_background_color = """
#mwnd, .bkg, .bkg::handle {
background-color: rgb(185, 194, 202);
}"""
        mwnd.setStyleSheet(qss_background_color)

        mwnd.show()
        app.exec_()
    except Exception as e:
        import traceback
        print("[!]", e, "\nMore Detail:\n" + "="*69)
        traceback.print_exc()
        print("="*69)
        # mwnd.close()
        app.exit()

"""
class MainWnd(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        loadUi("ui/wx_mwnd.ui", self)
"""

#####################################################################
# UI_tool
#####################################################################

from PyQt5 import uic

#####################################################################
# %% Clone From: utils.debug_v0.1.4
import sys
from importlib import import_module

def get_caller_path():
    """ 获取caller调用者模块的路径 """
    path_file = sys._getframe(2).f_code.co_filename  # inspect.stack()[2][1]
    return path_file

def path2module(path, package=None):
    assert not os.path.isabs(path), "请使用相对路径载入项目模块"
    without_ext, _ = os.path.splitext(path)
    path_posix = without_ext.replace("\\", ".")
    str_module = path_posix.replace("/", ".")
    return import_module(str_module, package)
#####################################################################

def _loadUi_by_Mixin(uifile, instance):
    """ uifile需要相对路径导入 """
    # path = without_ext.replace("ui/", "ui2py.")  # 默认规则：将res/ui目录改为res/ui2py
    module = path2module(uifile)
    try:
        Ui_Form = getattr(module, "Ui_Form")
    except AttributeError:
        # 提取模块中唯一的class对象
        raise NotImplementedError("请勿更改ui文件中“Ui_Form”的命名")

    # 通过Mixin的方式多继承Ui_Form
    WidgetClass = instance.__class__
    WidgetClass.__bases__ += (Ui_Form,)
    instance.setupUi(instance)

def loadUi(uifile_relto_curfile, instance):
    """ uifile: 相对当前模块的rpath
        instance: 固定传入self即可 """
    path_caller = get_caller_path()
    caller_dir_abs = os.path.dirname(path_caller)
    if caller_dir_abs:
        caller_dir_rel = os.path.relpath(caller_dir_abs, os.getcwd())
        uifile = os.path.join(caller_dir_rel, uifile_relto_curfile)
    else:
        uifile = uifile_relto_curfile
    uic.loadUi(uifile, instance)

#####################################################################
# wx_unit.py
#####################################################################

class UnitBase(QWidget):
    def isChecked(self):
        pass
    def get_value(self):
        pass
    def set_slot(self, func_slot):
        pass

class UnitLineEdit(UnitBase):
    def __init__(self, parent, name, val_default=0, isCheckbox=True, isChecked=False):
        super().__init__(parent)

        self.edit = QLineEdit(self)
        self.edit.setText(val_default if isinstance(val_default, str) else str(val_default))

        if isCheckbox:
            self.name = QCheckBox(name, self)
            self.name.setChecked(isChecked)
            self.edit.setEnabled(isChecked)
            self.name.stateChanged.connect(self.edit.setEnabled)
        else:
            self.name = QLabel(name, self)

        layout = QHBoxLayout(self)
        layout.addWidget(self.name)
        layout.addWidget(self.edit)

    def isChecked(self):
        if isinstance(self.name, QCheckBox):
            return self.name.isChecked()
        else:
            return True

    def set_slot(self, func_slot):
        self.edit.editingFinished.connect(func_slot)

    def get_value(self):
        text = self.get_text()
        return float(text)

    def get_text(self):
        return self.edit.text()

class UnitRadio(QGroupBox):  # UnitBase
    def __init__(self, parent, name, choices, val_init=0, choices_id=None):
        """ choices = [
            ["row0", "row0_1", "row0_2"],
            ["row1", "row1_1", "row1_2"]]
        """
        super().__init__(parent)

        self.setTitle(name)
        layout = QGridLayout(self)
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)  # 独占

        radio_id = -1
        for row, list_row in enumerate(choices):
            for column, str_item in enumerate(list_row):
                btn_radio = QRadioButton(str_item, self)
                layout.addWidget(btn_radio, row, column)
                if choices_id is None:
                    radio_id += 1
                else:
                    radio_id = choices_id[row][column]
                self.btn_group.addButton(btn_radio, radio_id)

        self.set_value(val_init)

    def set_slot(self, func_slot):
        self.btn_group.buttonClicked.connect(func_slot)

    def get_value(self):
        return self.btn_group.checkedId()  # checkedButton()

    def get_text(self):
        return self.btn_group.checkedButton().text()

    def set_value(self, index):
        self.btn_group.button(index).setChecked(True)

class UnitCheckBox(QGroupBox):  # UnitBase
    def __init__(self, parent, name, choices, val_init=0, choices_id=None):
        """ choices = [
            ["row0", "row0_1", "row0_2"],
            ["row1", "row1_1", "row1_2"]]
        """
        super().__init__(parent)

        self.setTitle(name)
        layout = QGridLayout(self)
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(False)

        radio_id = -1
        for row, list_row in enumerate(choices):
            for column, str_item in enumerate(list_row):
                btn_radio = QCheckBox(str_item, self)
                layout.addWidget(btn_radio, row, column)
                if choices_id is None:
                    radio_id += 1
                else:
                    radio_id = choices_id[row][column]
                self.btn_group.addButton(btn_radio, radio_id)

        self.set_value(val_init)

    def set_slot(self, func_slot):
        self.btn_group.buttonClicked.connect(func_slot)

    def get_value(self):
        # return self.btn_group.checkedId()  # checkedButton()
        list_ret = []
        for btn in self.btn_group.buttons():
            if btn.isChecked():
                list_ret.append(self.btn_group.id(btn))
        return list_ret

    def get_text(self):
        list_ret = []
        for btn in self.btn_group.buttons():
            if btn.isChecked():
                list_ret.append(btn.text())
        return list_ret

    def set_value(self, checked_id):
        """ checked_id: int or a list of id """
        if isinstance(checked_id, int):
            checked_id = [checked_id]
        for btn in self.btn_group.buttons():
            btn.setChecked(btn in checked_id)

class UnitSlider(UnitBase):
    def __init__(self, parent, name, val_range=None, val_default=0, showValue=True, isCheckbox=True, isChecked=False):
        super().__init__(parent)

        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)

        if isCheckbox:
            self.name = QCheckBox(name, self)
            self.name.setChecked(isChecked)
            self.slider.setEnabled(isChecked)
            self.name.stateChanged.connect(self.slider.setEnabled)
        else:
            self.name = QLabel(name, self)

        if val_range:
            self.slider.setRange(*val_range)
        self.slider.setValue(val_default)

        if showValue:
            self.label = QLabel(str(val_default), self)
            self.slider.valueChanged.connect(lambda x: self.label.setText(str(x)))

        layout = QHBoxLayout(self)
        layout.addWidget(self.name)
        layout.addWidget(self.slider)
        layout.addWidget(self.label)

    def isChecked(self):
        if isinstance(self.name, QCheckBox):
            return self.name.isChecked()
        else:
            return True

    def set_slot(self, func_slot):
        self.slider.valueChanged.connect(func_slot)

    def get_value(self):
        return self.slider.value()

class UnitSpinbox(UnitBase):
    def __init__(self, parent, name, val_range=None, val_default=0, isCheckbox=True, isChecked=False):
        super().__init__(parent)

        self.spinbox = QSpinBox(self)
        if val_range:
            self.spinbox.setRange(*val_range)
        self.spinbox.setValue(val_default)

        if isCheckbox:
            self.name = QCheckBox(name, self)
            self.name.setChecked(isChecked)
            self.spinbox.setEnabled(isChecked)
            self.name.stateChanged.connect(self.spinbox.setEnabled)
        else:
            self.name = QLabel(name, self)

        layout = QHBoxLayout(self)
        layout.addWidget(self.name)
        layout.addWidget(self.spinbox)

    def isChecked(self):
        if isinstance(self.name, QCheckBox):
            return self.name.isChecked()
        else:
            return True

    def set_slot(self, func_slot):
        self.spinbox.valueChanged.connect(func_slot)

    def get_value(self):
        return self.spinbox.value()

#####################################################################
# Canvas
#####################################################################

def distance(p):
    return math.sqrt(p.x() * p.x() + p.y() * p.y())

try:
    import numpy as np
    def distancetoline(point, line):
        p1, p2 = line
        p1 = np.array([p1.x(), p1.y()])
        p2 = np.array([p2.x(), p2.y()])
        p3 = np.array([point.x(), point.y()])
        if np.dot((p3 - p1), (p2 - p1)) < 0:
            return np.linalg.norm(p3 - p1)
        if np.dot((p3 - p2), (p1 - p2)) < 0:
            return np.linalg.norm(p3 - p2)
        return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

    HasImportNumpy = True
except ImportError:
    HasImportNumpy = False

#####################################################################

R, G, B = SHAPE_COLOR = 0, 255, 0  # green
DEFAULT_LINE_COLOR = QColor(R, G, B, 128)                # bf hovering
DEFAULT_FILL_COLOR = QColor(R, G, B, 128)                # hovering
DEFAULT_SELECT_LINE_COLOR = QColor(255, 255, 255)        # selected
DEFAULT_SELECT_FILL_COLOR = QColor(R, G, B, 155)         # selected
DEFAULT_VERTEX_FILL_COLOR = QColor(R, G, B, 255)         # hovering
DEFAULT_HVERTEX_FILL_COLOR = QColor(255, 255, 255, 255)  # hovering

class Shape(object):

    P_SQUARE, P_ROUND = 0, 1

    MOVE_VERTEX, NEAR_VERTEX = 0, 1

    # The following class variables influence the drawing of all shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    def __init__(self, label=None, line_color=None, shape_type=None,
                 flags=None, group_id=None):
        self.label = label
        self.group_id = group_id
        self.points = []
        self.fill = False
        self.selected = False
        self.shape_type = shape_type
        self.flags = flags
        self.other_data = {}

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

        self.shape_type = shape_type

    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = 'polygon'
        if value not in ['polygon', 'rectangle', 'point',
           'line', 'circle', 'linestrip']:
            raise ValueError('Unexpected shape_type: {}'.format(value))
        self._shape_type = value

    def close(self):
        self._closed = True

    def addPoint(self, point):
        if self.points and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)

    def canAddPoint(self):
        return self.shape_type in ['polygon', 'linestrip']

    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    def insertPoint(self, i, point):
        self.points.insert(i, point)

    def removePoint(self, i):
        self.points.pop(i)

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def getRectFromLine(self, pt1, pt2):
        x1, y1 = pt1.x(), pt1.y()
        x2, y2 = pt2.x(), pt2.y()
        return QRectF(x1, y1, x2 - x1, y2 - y1)

    def paint(self, painter):
        if self.points:
            color = self.select_line_color \
                if self.selected else self.line_color
            pen = QPen(color)
            # Try using integer sizes for smoother drawing(?)
            pen.setWidth(max(1, int(round(2.0 / self.scale))))
            painter.setPen(pen)

            line_path = QPainterPath()
            vrtx_path = QPainterPath()

            if self.shape_type == 'rectangle':
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.getRectFromLine(*self.points)
                    line_path.addRect(rectangle)
                for i in range(len(self.points)):
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "circle":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.getCircleRectFromLine(self.points)
                    line_path.addEllipse(rectangle)
                for i in range(len(self.points)):
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "linestrip":
                line_path.moveTo(self.points[0])
                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    self.drawVertex(vrtx_path, i)
            else:
                line_path.moveTo(self.points[0])
                # Uncommenting the following line will draw 2 paths
                # for the 1st vertex, and make it non-filled, which
                # may be desirable.
                # self.drawVertex(vrtx_path, 0)

                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    self.drawVertex(vrtx_path, i)
                if self.isClosed():
                    line_path.lineTo(self.points[0])

            painter.drawPath(line_path)
            painter.drawPath(vrtx_path)
            painter.fillPath(vrtx_path, self._vertex_fill_color)
            if self.fill:
                color = self.select_fill_color \
                    if self.selected else self.fill_color
                painter.fillPath(line_path, color)

    def drawVertex(self, path, i):
        d = self.point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        if self._highlightIndex is not None:
            self._vertex_fill_color = self.hvertex_fill_color
        else:
            self._vertex_fill_color = self.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def nearestVertex(self, point, epsilon):
        min_distance = float('inf')
        min_i = None
        for i, p in enumerate(self.points):
            dist = distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        assert HasImportNumpy

        min_distance = float('inf')
        post_i = None
        for i in range(len(self.points)):
            line = [self.points[i - 1], self.points[i]]
            dist = distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def containsPoint(self, point):
        return self.makePath().contains(point)

    def getCircleRectFromLine(self, line):
        """Computes parameters to draw with `QPainterPath::addEllipse`"""
        if len(line) != 2:
            return None
        (c, point) = line
        r = line[0] - line[1]
        d = math.sqrt(math.pow(r.x(), 2) + math.pow(r.y(), 2))
        rectangle = QRectF(c.x() - d, c.y() - d, 2 * d, 2 * d)
        return rectangle

    def makePath(self):
        if self.shape_type == 'rectangle':
            path = QPainterPath()
            if len(self.points) == 2:
                rectangle = self.getRectFromLine(*self.points)
                path.addRect(rectangle)
        elif self.shape_type == "circle":
            path = QPainterPath()
            if len(self.points) == 2:
                rectangle = self.getCircleRectFromLine(self.points)
                path.addEllipse(rectangle)
        else:
            path = QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        self._highlightIndex = None

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value


#####################################################################

CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor

class Canvas(QWidget):

    zoomRequest = pyqtSignal(int, QPoint)
    scrollRequest = pyqtSignal(int, int)
    newShape = pyqtSignal()
    selectionChanged = pyqtSignal(list)
    shapeMoved = pyqtSignal()
    drawingPolygon = pyqtSignal(bool)
    interactMouseClicked = pyqtSignal()

    edgeSelected = pyqtSignal(bool, object)
    vertexSelected = pyqtSignal(bool)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = 'polygon'

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop('epsilon', 10.0)
        self.double_click = kwargs.pop('double_click', 'close')
        if self.double_click not in [None, 'close']:
            raise ValueError(
                'Unexpected value for double_click event: {}'
                .format(self.double_click)
            )
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape()
        self.prevPoint = QPoint()
        self.prevMovePoint = QPoint()
        self.offsets = QPoint(), QPoint()
        self.scale = 1.0
        self.pixmap = QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self._painter = QPainter()
        self._cursor = CURSOR_DEFAULT
        self.interacting = False

        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QMenu(), QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in ['polygon', 'rectangle', 'circle',
           'line', 'point', 'linestrip']:
            raise ValueError('Unsupported createMode: %s' % value)
        self._createMode = value

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) >= 10:
            self.shapesBackups = self.shapesBackups[-9:]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.repaint()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            # if QT5:
            pos = self.transformPos(ev.localPos())
            # else:
            #     pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        self.prevMovePoint = pos
        self.restoreCursor()

        # Polygon drawing.
        if self.drawing():
            self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif len(self.current) > 1 and self.createMode == 'polygon' and\
                    self.closeEnough(pos, self.current[0]):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ['polygon', 'linestrip']:
                self.line[0] = self.current[-1]
                self.line[1] = pos
            elif self.createMode == 'rectangle':
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == 'circle':
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"
            elif self.createMode == 'line':
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == 'point':
                self.line.points = [self.current[0]]
                self.line.close()
            self.repaint()
            self.current.highlightClear()
            return

        # Polygon copy moving.
        if Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = \
                    [s.copy() for s in self.selectedShapes]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon / self.scale)
            index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click & drag to move point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s'") % shape.label)
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.edgeSelected.emit(self.hEdge is not None, self.hShape)
        self.vertexSelected.emit(self.hVertex is not None)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        point = self.prevMovePoint
        if shape is None or point is None:
            return
        index = shape.nearestVertex(point, self.epsilon)
        shape.removePoint(index)
        # shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = None
        self.hEdge = None
        self.movingShape = True  # Save changes

    def mousePressEvent(self, ev):
        # if QT5:
        pos = self.transformPos(ev.localPos())
        # else:
        #     pos = self.transformPos(ev.posF())

        if self.interacting:
            if ev.button() == Qt.LeftButton:
                if self.outOfPixmap(pos):
                    QMessageBox.warning(self, "Error", "选取点位超出图像区域，请重新选取")
                    return

                self.selected_pos = [int(pos.x()), int(pos.y())]
                self.interactMouseClicked.emit()
                return

        if ev.button() == Qt.LeftButton:
            if self.drawing():
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == 'polygon':
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                            self.drawingPolygon.emit(False)  #?? 非规范操作：手动封闭并执行后续
                    elif self.createMode in ['rectangle', 'circle', 'line']:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == 'linestrip':
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    self.current = Shape(shape_type=self.createMode)
                    self.current.addPoint(pos)
                    if self.createMode == 'point':
                        self.finalise()
                    else:
                        if self.createMode == 'circle':
                            self.current.shape_type = 'circle'
                        self.line.points = [pos, pos]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            else:
                group_mode = (int(ev.modifiers()) == Qt.ControlModifier)
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == Qt.RightButton and self.editing():
            group_mode = (int(ev.modifiers()) == Qt.ControlModifier)
            self.selectShapePoint(pos, multiple_selection_mode=group_mode)
            self.prevPoint = pos
            self.repaint()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton:
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos())) \
                    and self.selectedShapesCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapesCopy = []
                self.repaint()
        elif ev.button() == Qt.LeftButton and self.selectedShapes:
            self.overrideCursor(CURSOR_GRAB)

        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if (self.shapesBackups[-1][index].points !=
                    self.shapes[index].points):
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.repaint()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and self.current and len(self.current) > 2

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if (self.double_click == 'close' and self.canCloseShape() and
                len(self.current) > 3):
            self.current.popPoint()
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.calculateOffsets(shape, point)
                    self.setHiding()
                    if multiple_selection_mode:
                        if shape not in self.selectedShapes:
                            self.selectionChanged.emit(
                                self.selectedShapes + [shape])
                    else:
                        self.selectionChanged.emit([shape])
                    return
        self.deSelectShape()

    def calculateOffsets(self, shape, point):
        rect = shape.boundingRect()
        x1 = rect.x() - point.x()
        y1 = rect.y() - point.y()
        x2 = (rect.x() + rect.width() - 1) - point.x()
        y2 = (rect.y() + rect.height() - 1) - point.y()
        self.offsets = QPoint(x1, y1), QPoint(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QPoint(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QPoint(min(0, self.pixmap.width() - o2.x()),
                                 min(0, self.pixmap.height() - o2.y()))
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def copySelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QPoint(2.0, 2.0)
        self.offsets = QPoint(), QPoint()
        self.prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.HighQualityAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)
        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and \
                    self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if (self.fillDrawing() and self.createMode == 'polygon' and
                self.current is not None and len(self.current.points) >= 2):
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.fill_color.setAlpha(64)
            drawing_shape.paint(p)

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QPoint(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        assert self.current
        self.current.close()
        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [(0, 0),
                  (size.width() - 1, 0),
                  (size.width() - 1, size.height() - 1),
                  (0, size.height() - 1)]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QPoint(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QPoint(min(max(0, x2), max(x3, x4)), y3)
        return QPoint(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QPoint((x3 + x4) / 2, (y3 + y4) / 2)
                d = distance(m - QPoint(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        # if QT5:
        mods = ev.modifiers()
        delta = ev.angleDelta()
        if Qt.ControlModifier == int(mods):
            # with Ctrl/Command key
            # zoom
            self.zoomRequest.emit(delta.y(), ev.pos())
        else:
            # scroll
            self.scrollRequest.emit(delta.x(), Qt.Horizontal)
            self.scrollRequest.emit(delta.y(), Qt.Vertical)
        # else:
        #     if ev.orientation() == Qt.Vertical:
        #         mods = ev.modifiers()
        #         if Qt.ControlModifier == int(mods):
        #             # with Ctrl/Command key
        #             self.zoomRequest.emit(ev.delta(), ev.pos())
        #         else:
        #             self.scrollRequest.emit(
        #                 ev.delta(),
        #                 Qt.Horizontal
        #                 if (Qt.ShiftModifier == int(mods))
        #                 else Qt.Vertical)
        #     else:
        #         self.scrollRequest.emit(ev.delta(), Qt.Horizontal)
        ev.accept()

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == Qt.Key_Escape and self.current:
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
        elif key == Qt.Key_Return and self.canCloseShape():
            self.finalise()

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        if self.createMode in ['polygon', 'linestrip']:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ['rectangle', 'line', 'circle']:
            self.current.points = self.current.points[0:1]
        elif self.createMode == 'point':
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.repaint()

    def loadPixmap(self, pixmap):
        self.pixmap = pixmap
        self.shapes = []
        self.repaint()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.repaint()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.repaint()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()


#####################################################################

class UndoCommand:
    def execute(self):
        """ 执行的操作 """
    def rollback(self):
        """ 回滚时执行的操作 """
    def exec_init(self):
        """ 仅在第一次执行时操作，包括一些特殊的初始化操作 """

class UndoIndexError(IndexError):
    """ 撤销越界 """

class UndoStack:
    # 用于存储、管理UndoCommand操作
    def __init__(self):
        Stack = list

        self.stack_undo = Stack()  # elem: UndoCommand
        self.stack_redo = Stack()

    def _debug(self):
        total = len(self.stack_undo)
        for index, undo_cmd in enumerate(self.stack_undo):
            logger.debug(f"stack memory [{index +1}/{total}] >> {undo_cmd}")

    def undo(self):
        if not self.stack_undo:
            raise UndoIndexError("已撤回至初始状态")
            # logger.warning("已撤回至初始状态")
            # return
        undo_cmd = self.stack_undo.pop()
        ret = undo_cmd.rollback()
        self.stack_redo.append(undo_cmd)
        # self._debug()
        return ret

    def redo(self):
        if not self.stack_redo:
            raise UndoIndexError("已恢复至最终状态")
            # logger.warning("已恢复至最终状态")
            # return
        redo_cmd = self.stack_redo.pop()
        ret = redo_cmd.execute()
        self.stack_undo.append(redo_cmd)
        # self._debug()
        return ret

    def commit(self, undo_cmd):
        """ 若取消提交，return True """
        self.stack_redo.clear()
        undo_cmd.exec_init()
        ret = undo_cmd.execute()
        # 将command压栈
        self.stack_undo.append(undo_cmd)
        # print(">> UndoStack 压栈: ", undo_cmd)
        return ret

class ImgSnapCommand(UndoCommand):
    """ 功能仅限简单存储图像 """
    def __init__(self, ips_prev, ips_new):
        self.prev = ips_prev
        self.new = ips_new

    def execute(self):
        return self.new.copy()

    def rollback(self):
        return None if self.prev is None else self.prev.copy()

class ImgScriptCommand(ImgSnapCommand):
    """ 实现对操作函数的记录 """
    def __init__(self, ips_prev, scripts, ips_new):
        super().__init__(ips_prev, ips_new)
        self.scripts = scripts

class ImageContainer:
    """ 用于图像的收发管理 """
    def get_image(self):
        """ 返回一张ndarry图像 """
        return self.im_arr

    def set_image(self, im_arr):
        """ 设值Canvas图像 """
        self.im_arr = im_arr

class ImageManager(ImageContainer):
    """ 用于图像的多版本管理，包括：
        * stack: 每次执行图像操作的processing（而不是存储快照）
        * snap: 用于存储origin图像，用于连续测试时获取原始图像

        目标：尽可能使其接口接近ImagePlus，无缝对接canvas
    """
    def __init__(self):
        self.curr = None  # ips
        self.snap = None  # ips
        self.stack = UndoStack()

    def take_snap(self):
        self.snap = None if self.curr is None else self.curr.copy()

    def reset(self):
        self.curr = self.snap.copy()

    # def load_image(self, path_file):
    #     im_arr = imread(path_file)
    #     self.commit(im_arr)

    def get_image(self):
        return self.curr

    def get_snap(self):
        return self.snap

    def set_image(self, im_arr):
        """ 注意：Manager中set_image() 只是临时显示图像
            如确定存储图像，应配合使用commit()
        """
        assert isinstance(im_arr, np.ndarray), f"请传入np.ndarray的图像格式：【{type(im_arr)}】"
        self.curr = im_arr

    def commit(self, scripts=None):
        """ 确定当前图像take_snap()操作，并写入UndoStack """
        ips_prev = None if self.snap is None else self.snap.copy()
        # cmd = ImgSnapCommand(ips_prev, self.curr.copy())
        cmd = ImgScriptCommand(ips_prev, scripts, self.curr.copy())
        self.stack.commit(cmd)
        self.take_snap()

    def undo(self):
        """ 撤销操作 """
        self.curr = self.stack.undo()
        self.take_snap()

    def redo(self):
        self.curr = self.stack.redo()
        self.take_snap()

    def history(self):
        """ 显示undostack列表 """

    def dumps(self):
        """ 导出commit的脚本形式 """
        list_scripts = []
        for uno_cmd in self.stack.stack_undo:
            list_scripts.append(uno_cmd.scripts)
        return list_scripts

class QImageManager(QObject, ImageManager):
    """ 对于不以ImageManager作为直接的图像显示对象的Canvas容器而言，
        需要通过Qt信号通知canvas更新视图显示
    """
    updateImage = pyqtSignal()  # 通知canvas等Pixmap元素更新UI

    def set_image(self, im_arr):
        super().set_image(im_arr)
        self.updateImage.emit()

    def reset(self):
        super().reset()
        self.updateImage.emit()

    def undo(self):
        super().undo()
        self.updateImage.emit()

    def redo(self):
        super().redo()
        self.updateImage.emit()

#####################################################################

def delta2units(delta):
    return delta / 120  # 8 * 15

class ScrollArea(QScrollArea):  # ViewerBase
    """ 基本Canvas集成单元，本质上是一个包装器：
        - 通过Canvas显示和勾勒图像
        - 利用ImageContainer存储和管理/输出图像
    """
    MIN_ZOOM_RATIO = 0.05

    def __init__(self, parent):
        super().__init__(parent)
        self.setWidgetResizable(True)

        self.imgr = QImageManager()
        self.imgr.updateImage.connect(self.update_canvas)

        self.zoom_val = 100
        # self.setBackgroundRole()  # 背景色
        # self.setAlignment(Qt.AlignCenter)  # 居中对齐
        self.canvas = Canvas()
        self.setWidget(self.canvas)

    def _img_size(self):
        im = self.get_image()
        h, w = im.shape[:2]
        return (w, h)

    def update_canvas(self):
        # 更新canvas画面
        im_arr = self.imgr.get_image()
        if im_arr is None:
            self.canvas.resetState()
        else:
            pixmap = cv.ndarray2pixmap(im_arr)
            self.canvas.loadPixmap(pixmap)

    def get_container(self):
        return self.imgr

    def get_image(self):
        return self.imgr.get_image()

    def set_image(self, im_arr):
        self.imgr.set_image(im_arr)
        self.set_fit_window()

    def wheelEvent(self, event):
        delta = event.angleDelta()
        h_delta = delta.x()
        v_delta = delta.y()
        # if event.orientation() == Qt.Vertical: h_delta = 0

        mods = event.modifiers()
        if Qt.ControlModifier == int(mods) and v_delta:
            self.on_zoom(v_delta)
        else:
            v_delta and self.on_scroll(Qt.Vertical, v_delta)
            h_delta and self.on_scroll(Qt.Horizontal, h_delta)

        event.accept()

    #####################################################################
    def repaint(self):
        self.canvas.scale = 0.01 * self.zoom_val
        self.canvas.adjustSize()
        # logger.debug(f"当前的幕布尺寸：{self.canvas.size()}")
        self.canvas.update()

    def set_fit_origin(self):
        self.zoom_val = 100
        self.repaint()

    def set_fit_window(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2  # So that no scrollbars are generated.
        w1 = self.width() - e
        h1 = self.height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        # w2 = self.canvas.pixmap.width()
        # h2 = self.canvas.pixmap.height()
        w2, h2 = self._img_size()
        a2 = w2 / h2
        val = w1 / w2 if a2 >= a1 else h1 / h2
        self.zoom_val = int(val * 100)
        self.repaint()

    def set_fit_width(self):
        # The epsilon does not seem to work too well here.
        w = self.width() - 2
        # val = w / self.canvas.pixmap.width()
        val = w / self._img_size()[0]
        self.zoom_val = int(val * 100)
        self.repaint()

    def on_scroll(self, orientation, delta):
        # logger.debug(f"delta: {delta}, orientation: {orientation}")
        units = delta2units(delta)
        if orientation == Qt.Vertical:
            bar = self.verticalScrollBar()
        else:  # orientation == Qt.Horizontal:
            bar = self.horizontalScrollBar()

        bar.setValue(bar.value() - bar.singleStep() * units)

    def on_zoom(self, delta):
        # zoom in
        units = delta2units(delta)
        scale = 10
        zoom_val = self.zoom_val + scale * units
        if zoom_val < self.MIN_ZOOM_RATIO:  # 设定最小缩放比例
            return
        self.zoom_val = zoom_val
        # logger.debug(f"reset the zoom-scale to: {self.zoom_val}")
        self.adjust_bar_pos()  # 重定位当前的bar.position
        self.repaint()

    def adjust_bar_pos(self):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.verticalScrollBar()
        v_bar = self.horizontalScrollBar()

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        pos = QCursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.width()
        h = self.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

class ScrollCanvasBase(ScrollArea):
    """ 一般需要重写 """
    def __init__(self, parent):
        super().__init__(parent)
        # self.parent = parent

        # 拖拽文件
        self.setAcceptDrops(True)

        # 右键菜单
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.on_right_menu)

    def on_right_menu(self):
        self.contextMenu = QMenu()
        self.contextMenu.popup(QCursor.pos())

        act_fit_origin = self.contextMenu.addAction("图片默认尺寸")
        act_fit_origin.triggered.connect(self.set_fit_origin)

        act_fit_window = self.contextMenu.addAction("匹配窗口尺寸")
        act_fit_window.triggered.connect(self.set_fit_window)

        act_fit_width = self.contextMenu.addAction("匹配窗口宽度")
        act_fit_width.triggered.connect(self.set_fit_width)

        self.contextMenu.show()

    def dragEnterEvent(self, event):
        """ 只在进入Window的瞬间触发 """
        event.accept()  # 鼠标放开函数事件

    def dropEvent(self, event):
        path_file = event.mimeData().text().lstrip("file:///")

        _, ext = os.path.splitext(path_file)
        if ext not in [".png", ".jpg", ".bmp"]:
            QMessageBox.warning(self, "警告", "只支持 png/jpg/bmp 图片文件")
            return

        im_arr = cv.imread(path_file)
        self.imgr.set_image(im_arr)
        self.set_fit_window()

    # def set_ndarray(self, ndarray):
    #     pixmap = imgio.ndarray2pixmap(ndarray)
    #     self.canvas.setEnabled(False)
    #     self.canvas.loadPixmap(pixmap)
    #     self.canvas.setEnabled(True)
    #     self.set_fit_window()  # 默认适配窗口尺寸

    # def get_ndarray(self):
    #     pixmap = self.canvas.pixmap
    #     ndarray = imgio.pixmap2ndarray(pixmap)
    #     return ndarray

#####################################################################
# Dialog
#####################################################################

def dialog_file_select(parent, str_filter=None, canMutilSelect=False, onlyDir=False, saveSuffix=None):
    """ return a list of path (支持多选) """
    # caller = self.sender
    dialog = QFileDialog(parent)
    if canMutilSelect:
        dialog.setFileMode(QFileDialog.ExistingFiles)

    if onlyDir:
        dialog.setFileMode(QFileDialog.Directory)  # 只显示目录
        dialog.setOption(QFileDialog.ShowDirsOnly)
    # else:
    #     dialog.setFileMode(QFileDialog.AnyFile)

    if str_filter:
        dialog.setNameFilter(str_filter)  # "Images (*.png *.xpm *.jpg)"

    if saveSuffix:
        dialog.setDefaultSuffix(saveSuffix)

    if dialog.exec():
        list_path = dialog.selectedFiles()
        return list_path

def attach_widget(parent, widget, noInnerMargins=False, noOuterMargins=False):
    outer_layout = parent.layout()
    if outer_layout is None:
        outer_layout = QVBoxLayout(parent)
    if noOuterMargins:
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
    if noInnerMargins:
        inner_layout = widget.layout()
        if inner_layout:
            inner_layout.setContentsMargins(0, 0, 0, 0)
            # inner_layout.setSpacing(0)
    outer_layout.addWidget(widget)

def make_dialog(parent, func_wx_create, title=None, size=None, path_icon=None):
    dialog = QDialog(parent)
    if title:
        dialog.setWindowTitle(title)
    if path_icon:
        dialog.setWindowIcon(QIcon(path_icon))
    if size:
        dialog.resize(*size)
    QVBoxLayout(dialog)

    unit_widget = func_wx_create(parent=dialog)
    attach_widget(dialog, unit_widget, noOuterMargins=True)
    return dialog

class DialogWrapper(QDialog):
    def __init__(self, parent, title=None, size=None, path_icon=None):
        super().__init__(parent)
        self.parent = parent

        if title:
            self.setWindowTitle(title)
        if path_icon:
            self.setWindowIcon(QIcon(path_icon))
        if size:
            self.resize(*size)

        self.setLayout(QVBoxLayout(self))

    def load_plugin(self, str_module, __file__=None, **kwargs):
        from .debug import import_plugin

        wx_plugin = import_plugin(str_module, parent=self.parent, attach=self, **kwargs)
        # wx_plugin.setObjectName(self.PLUGIN_ID)
        attach_widget(self, wx_plugin, noOuterMargins=True)

    def load_instance(self, WidgetClass, **kwargs):
        wx_instance = WidgetClass(parent=self.parent, attach=self, **kwargs)
        attach_widget(self, wx_instance, noOuterMargins=True)

    def get_widget(self):
        wx_plugin = extract_widgets(self, index=0)
        return wx_plugin

    # 虽然可以使用特定的回调，但需要特殊处理传参。推荐使用猴子补丁，重写closeEvent()
    # def closeEvent(self, event):
    #     self.close_callback()

#####################################################################
# Widget
#####################################################################

qss_frame_color = """\
QWidget[whatsThis="frame"] {
    background-color: rgb(228, 231, 233);
    border-radius: 10px;
}"""

def extract_widgets(layout1wx, **kwargs):
    """
    output:
        - list: [("obj_name", widget), (xx, yy), ... ]
        - dict: {
                    "btn_id"    : QPushButton-Object,
                    "btn_id_2"  : QPushButton-Object,
                    ...
                }

    params:
        name    [str]: 根据widget的objectName，查找指定对象（递归，返回list）；
                       如未找到返回None；
        index   [int]: 根据该layout对象addWidget的顺序，提取widget对象；
                       如int参数越界，抛出异常
        type    [cls]: 查找第一个指定类型的widget对象，如未找到返回None
        pypes   (cls): 输入多个可选qt控件类型（Tuple），返回全部符合条件的对象
        all          : 返回layout中所有widget对象的映射字典
    """
    if "name" in kwargs:
        obj_name = kwargs["name"]
        qt_type = kwargs.get("type", QWidget)
        return layout1wx.findChild(qt_type, name=obj_name)  # return a list

    if isinstance(layout1wx, QLayout):
        layout = layout1wx
    else:
        layout = layout1wx.layout()
        if not layout:
            raise Exception("This containenr [{}] has NO layout.".format(layout1wx))

    if "index" in kwargs:
        # 如index越界，则程序执行异常
        index = kwargs["index"]
        if index < 0:
            index = layout.count() -1
        return layout.itemAt(index).widget()  # 直接输出(name,wx_obj)

    collection = []  # object: index

    args_type = kwargs.get("type")
    args_types = kwargs.get("types")
    args_all = "all" in kwargs

    for index in range(layout.count()):
        layout_item = layout.itemAt(index)
        sub_widget = layout_item.widget()  # 获取到widget对象
        if sub_widget is None:
            continue

        elif (args_type and isinstance(sub_widget, args_type)) or \
                 (args_types and sub_widget not in args_types) or \
                 args_all:
            # wx_name = sub_widget.objectName()
            collection.append(sub_widget)

    return collection


def clear_layout(layout):
    """
        itemAt(): 描述如何递归布局
        takeAt(): 描述如何移除布局中的元素
    """
    list_wx = extract_widgets(layout, all="anything here")
    for wx in list_wx:
        wx.deleteLater()

    # for index in range(layout.count()):
    while True:
        wx_item = layout.takeAt(0)
        if wx_item is None:
            break
        else:
            layout.removeItem(wx_item)
            del wx_item

def pixmap_label(qlabel, pixmap, size=None, scale_type=0):
    """ Qt.IgnoreAspectRatio:   0  # 填充
        Qt.KeepAspectRatio:    1  # 保持比例，按最长边对齐
        Qt.KeepAspectRatioByExpanding: 2  # 保持比例，最短边对齐
    """
    if size:
        pixmap.scaled(*size, scale_type)
    qlabel.setScaledContents(True)
    qlabel.setPixmap(pixmap)

def pixmap_button(button, path_pic=None):
    obj_name = "tmp"
    button.setObjectName(obj_name)
    # 使用border-image图像，图像整体缩放（若使用background-image，则使用默认尺寸层叠）
    button.setStyleSheet("#{}{{border-image:url({})}}".format(obj_name, path_pic))

"""
def set_bkg_image(qt_obj, path_pic=None, way="QSS", **kwargs):
    # 请注意，Qt对jpg的支持不如png，所以请尽量使用png图片格式
    if path_pic and not os.path.exists(path_pic):
        raise Exception(f"不存在图像文件路径【{path_pic}】")

    if way == "QSS":
        if "color" in kwargs:
            # 仅用以设置背景色
            qt_obj.setStyleSheet("background-color:{};".format(kwargs["color"]))
            return
        else:
            # 使用样式表，设置背景图片
            obj_name = qt_obj.objectName()
            if not obj_name:
                obj_name = "tmp"
                qt_obj.setObjectName(obj_name)
            # 可换用 background-image, border-image 或 image
            qt_obj.setStyleSheet("#{}{{border-image:url({})}}".format(obj_name, path_pic))

    elif way == "icon":
        # 针对 QAbstractButton 等图元类型设置背景图
        # bitmap = QPixmap(path_pic)
        icon = QIcon(path_pic)  # QIcon(bitmap)

        qt_obj.setIcon(icon)  # 限制：无法拉伸Icon图片，故只能让按钮适应图片
        qt_obj.setIconSize(qt_obj.size())

        # qt_obj.setFixedSize(bitmap.size())
        # qt_obj.setMask(bitmap.mask())

    elif way == "palette":
        # 默认使用画刷
        palette	= QPalette()
        palette.setBrush(QPalette.Background, # Window
                        QBrush(QPixmap(path_pic)))
        if not qt_obj.autoFillBackground():
            qt_obj.setAutoFillBackground(True)  # 设置图片填充
        qt_obj.setPalette(palette)

    else:
        raise Exception("Unknown param 'way' = [{}]".format(way))
"""

#####################################################################
# util
#####################################################################

def get_window_center(parent, window):
    pos = window.pos()
    x, y = pos.x(), pos.y()
    size = parent.size()
    width, height = size.width(), size.height()
    center = (x + width/2, y + height/2)
    return center

def show_dialog_at_center(parent, subwindow):
    pos = parent.pos()
    x, y = pos.x(), pos.y()
    size = parent.size()
    width, height = size.width(), size.height()
    center = (x + width/2, y + height/2)

    sub_size = subwindow.size()
    width, height = sub_size.width(), sub_size.height()

    pos_related_parent = center[0] - width /2, center[1] - height /2
    subwindow.move(*pos_related_parent)

    subwindow.show()

#####################################################################
# Menu
#####################################################################

def make_action(parent, text, func_slot=None, path_icon=None,
                shortcut=None, tip=None):
    """ parent could be a menu or toolbar """
    # exitAction = QAction(QIcon('exit.png'), '&Exit', self)
    action = parent.addAction(text)  # '&Exit'
    if func_slot:
        action.triggered.connect(func_slot)
    if shortcut:
        action.setShortcut(shortcut)
    if path_icon:
        action.setIcon(QIcon(path_icon))
    if tip:
        action.setStatusTip(tip)
    return action

def make_submenu(menu, text, path_icon=None):
    # exitAction = QAction(QIcon('exit.png'), '&Exit', self)
    submenu = menu.addMenu(text)  # '&Exit'
    if path_icon:
        submenu.setIcon(QIcon(path_icon))
    return submenu
