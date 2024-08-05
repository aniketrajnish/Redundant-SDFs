import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, Qt
from OpenGL.GL import *
from PyQt5.QtWidgets import QWidget
from raymarcher import Raymarcher

class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)
        self.raymarcher = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16) # 60 FPS
        self.time = 0
        self.selected_shape = 0

    def initializeGL(self):
        self.raymarcher = Raymarcher()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        self.time += 0.016
        self.raymarcher.render(self.width(), 
                               self.height(), 
                               self.time, 
                               self.selected_shape)
        
class VizWindow(QMainWindow):
    def __init__(self):
        super(VizWindow, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.init_window()
        self.init_layout()
        self.init_widgets()

    def init_window(self):
        self.setWindowTitle('Visualizing Shapes')
        self.setGeometry(100, 100, 1200, 800)  # Increased window size
        self.show()

    def init_layout(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout(self.central_widget)
        self.central_widget.setLayout(self.main_layout)

    def init_widgets(self):
        self.init_gl_widget()
        self.init_btn_widget()

    def init_gl_widget(self):
        self.gl_widget = GLWidget()
        self.main_layout.addWidget(self.gl_widget, 1)  # Give more space to GL widget

    def init_btn_widget(self):
        btn_widget = QWidget()
        btn_layout = QGridLayout(btn_widget)
        shapes = [
            "Sphere", "Torus", "Capped Torus", "Link", "Cone", "Infinite Cone", "Plane",
            "Hexagonal Prism", "Triangular Prism", "Capsule", "Infinite Cylinder", "Box",
            "Round Box", "Rounded Cylinder", "Capped Cone", "Box Frame", "Solid Angle",
            "Cut Sphere", "Cut Hollow Sphere", "Death Star", "Round Cone", "Ellipsoid",
            "Rhombus", "Octahedron", "Pyramid", "Triangle", "Quad", "Fractal"
        ]

        for i, shape in enumerate(shapes):
            btn = QPushButton(shape)
            btn.clicked.connect(lambda checked, idx=i: self.select_shape(idx))
            btn_layout.addWidget(btn, i // 4, i % 4)

        scroll_area = QScrollArea()
        scroll_area.setWidget(btn_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(300) 

        self.main_layout.addWidget(scroll_area)

    def select_shape(self, shape_idx):
        self.gl_widget.selected_shape = shape_idx

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VizWindow()
    sys.exit(app.exec_())