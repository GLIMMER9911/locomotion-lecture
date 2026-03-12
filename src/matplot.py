import numpy as np
import time
from typing import Optional, Dict, Any, Tuple
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import sys
import weakref


class MultiChartRealTimePlotManager:
    def __init__(self):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.plotters: Dict[str, Dict[str, Any]] = {}
        self.max_points = 1000
        self.default_figsize = (1000, 800)  
        self.parent_win: Optional[QtWidgets.QMainWindow] = None  
        self.main_layout: Optional[QtWidgets.QGridLayout] = None

    def _init_parent_window(self):
        if self.parent_win is not None:
            return
        self.parent_win = QtWidgets.QMainWindow()
        self.parent_win.setWindowTitle("Real-Time Monitor")
        self.parent_win.resize(*self.default_figsize)
        
        central_widget = QtWidgets.QWidget()
        self.parent_win.setCentralWidget(central_widget)
        self.main_layout = QtWidgets.QGridLayout(central_widget)
        self.parent_win.show()

    def _create_base_plotter(self, plotter_name: str, 
                             title: str, x_label: str, y_label: str,
                             row: int, col: int):
        self._init_parent_window()
        if plotter_name in self.plotters:
            return
        
        plot_widget = pg.PlotWidget()
        plot_widget.setTitle(title, size="11pt")
        plot_widget.setLabel('bottom', x_label, size="9pt")
        plot_widget.setLabel('left', y_label, size="9pt")
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_widget.setObjectName(plotter_name)
        
        self.main_layout.addWidget(plot_widget, row, col)
        
        self.plotters[plotter_name] = {
            "plot_widget": weakref.ref(plot_widget),
            "series": {},
            "start_time": time.time(),
            "valid": True
        }

    def addNewFigurePlotter(self, plotter_name: str, title: str = "rt data", 
                            x_label: str = "time", y_label: str = "value",
                            row: int = 0, col: int = 0):  
        self._create_base_plotter(plotter_name, title, x_label, y_label, row, col)

    def addPlotToPlotter(self, plotter_name: str, series_name: str, 
                         color: Optional[str] = None, linestyle: str = '-', 
                         linewidth: float = 1.5, label: Optional[str] = None):
        if plotter_name not in self.plotters or not self.plotters[plotter_name]["valid"]:
            return
        
        plotter = self.plotters[plotter_name]
        if series_name in plotter["series"]:
            return
        
        plot_widget = plotter["plot_widget"]() if plotter["plot_widget"] else None
        if not plot_widget:
            plotter["valid"] = False
            return
        
        pen = pg.mkPen(color=color or 'cyan', width=linewidth)
        if linestyle == ':':
            pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        
        curve = plot_widget.plot(name=label or series_name, pen=pen)
        plot_widget.addLegend(size=(80, 40))
        
        plotter["series"][series_name] = {
            "curve": weakref.ref(curve),
            "x_data": [],
            "y_data": []
        }

    def updateDataToPlotter(self, plotter_name: str, series_name: str, new_y: float):
        if plotter_name not in self.plotters:
            return
        
        plotter = self.plotters[plotter_name]
        if not plotter["valid"]:
            return
        
        if series_name not in plotter["series"]:
            self.addPlotToPlotter(plotter_name, series_name)
            if series_name not in plotter["series"]:
                return
        
        plot_widget = plotter["plot_widget"]() if plotter["plot_widget"] else None
        curve = plotter["series"][series_name]["curve"]() if plotter["series"][series_name]["curve"] else None
        if not plot_widget or not curve:
            plotter["valid"] = False
            return
        
        current_x = time.time() - plotter["start_time"]
        series = plotter["series"][series_name]
        
        series["x_data"].append(current_x)
        series["y_data"].append(new_y)
        
        if len(series["x_data"]) > self.max_points:
            series["x_data"] = series["x_data"][-self.max_points:]
            series["y_data"] = series["y_data"][-self.max_points:]
        
        curve.setData(series["x_data"], series["y_data"])
        self.app.processEvents()

    def closeAll(self):
        if self.parent_win:
            self.parent_win.close()
        self.app.quit()


if __name__ == "__main__":
    plot_manager = MultiChartRealTimePlotManager()
    
    plot_manager.addNewFigurePlotter("acc.x", title="acc.x", row=0, col=0)
    plot_manager.addNewFigurePlotter("speed.x", title="speed.x", row=0, col=1)
    plot_manager.addNewFigurePlotter("nowspeed.x", title="nowspeed.x", row=0, col=2)
    plot_manager.addNewFigurePlotter("delta", title="delta", row=1, col=0)
    # plot_manager.addNewFigurePlotter("delta", title="delta.y", row=1, col=1)
    # plot_manager.addNewFigurePlotter("delta", title="delta.z", row=1, col=2)
    
    plot_manager.addPlotToPlotter("acc.x", "acc.x", color="g")
    plot_manager.addPlotToPlotter("speed.x", "speed.x", color="r")
    plot_manager.addPlotToPlotter("nowspeed.x", "nowspeed.x", color="b")
    plot_manager.addPlotToPlotter("delta", "delta.x", color="y")
    plot_manager.addPlotToPlotter("delta", "delta.y", color="m")
    plot_manager.addPlotToPlotter("delta", "delta.z", color="c")
    
    try:
        while True:
            plot_manager.updateDataToPlotter("acc.x", "acc.x", np.random.randn()*0.5+4)
            plot_manager.updateDataToPlotter("speed.x", "speed.x", np.random.randn()*10+300)
            plot_manager.updateDataToPlotter("nowspeed.x", "nowspeed.x", np.random.randn()*5+730)
            plot_manager.updateDataToPlotter("delta", "delta.x", np.random.randn()*0.2+4)
            plot_manager.updateDataToPlotter("delta", "delta.y", np.random.randn()*0.2+4)
            plot_manager.updateDataToPlotter("delta", "delta.z", np.random.randn()*0.2+4)
            time.sleep(0.1)
    except KeyboardInterrupt:
        plot_manager.closeAll()
