from modules.ui.BaseCloudTabView import BaseCloudTabView
from modules.ui.CloudTabController import CloudTabController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLineEdit, QStyledItemDelegate, QWidget

_STOCK_COLORS = {
    "High": "#66bb6a",
    "Medium": "#ffb300",
    "Low": "#ff7043",
    "Available": "#66bb6a",
    "Out of stock": "#9e9e9e",
}


class _GpuStockDelegate(QStyledItemDelegate):
    """Paint availability and price right-aligned beside the GPU type."""

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        marker = index.data(Qt.ItemDataRole.UserRole)
        if not marker:
            return
        color_key = index.data(Qt.ItemDataRole.UserRole + 1)
        painter.save()
        painter.setPen(QColor(_STOCK_COLORS.get(color_key, "#9e9e9e")))
        painter.drawText(
            option.rect.adjusted(0, 0, -6, 0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            marker,
        )
        painter.restore()


class PySide6CloudTabView(BaseCloudTabView, QWidget, metaclass=QtABCMeta):

    def __init__(self, master, controller: CloudTabController, ui_state):
        QWidget.__init__(self, master)
        BaseCloudTabView.__init__(self, pyside6_components, controller)

        self.ui_state = ui_state

        scroll, frame = pyside6_components.scrollable_frame(self)
        pyside6_components._layout(self).addWidget(scroll, 0, 0)
        lo = pyside6_components._layout(frame)
        lo.setColumnStretch(1, 1)
        lo.setColumnStretch(3, 1)
        lo.setColumnStretch(5, 1)
        self.frame = frame

        self.build_content(frame, controller, ui_state)

        self.gpu_types_menu.setItemDelegate(_GpuStockDelegate(self.gpu_types_menu))
        self.gpu_types_menu.view().setMinimumWidth(520)

    def _on_set_gpu_types(self):
        combo = self.gpu_types_menu
        previous = combo.currentText()
        infos = self.controller.get_gpu_availability()

        # Keep the visible item text as the plain GPU id because the UI-state
        # binding saves currentText() into cloud.gpu_type.
        combo.blockSignals(True)
        combo.clear()
        for index, info in enumerate(infos):
            combo.addItem(info["id"])
            marker = None
            color_key = None
            if "available" in info:
                available = int(info["available"])
                marker = f"{available} offer{'s' if available != 1 else ''}"
                color_key = "Available" if available else "Out of stock"
            elif "stock_status" in info:
                marker = info["stock_status"] or "Out of stock"
                color_key = marker

            price = info.get("price")
            if price is not None:
                marker = f"{marker}  ${float(price):.2f}/h" if marker else f"${float(price):.2f}/h"
            if marker:
                combo.setItemData(index, marker, Qt.ItemDataRole.UserRole)
                combo.setItemData(index, color_key, Qt.ItemDataRole.UserRole + 1)
                combo.setItemData(index, marker, Qt.ItemDataRole.ToolTipRole)

        selected_index = combo.findText(previous)
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

        if selected_index < 0 and combo.count() > 0:
            combo.setCurrentIndex(0)

    def _set_visible(self, widget, visible: bool):
        widget.setVisible(visible)

    def _mask_secret(self, widget):
        widget.setEchoMode(QLineEdit.EchoMode.Password)

    def _make_reattach_frame(self, frame):
        reattach_frame = QWidget(frame)
        pyside6_components._layout(frame).addWidget(reattach_frame, 9, 3)
        pyside6_components._layout(reattach_frame).setColumnStretch(0, 1)
        return reattach_frame

    def _make_create_frame(self, frame):
        create_frame = QWidget(frame)
        pyside6_components._layout(frame).addWidget(create_frame, 1, 5)
        pyside6_components._layout(create_frame).setColumnStretch(1, 1)
        return create_frame
