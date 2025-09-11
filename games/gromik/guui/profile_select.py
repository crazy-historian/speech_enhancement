from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QListWidget, QMessageBox,
    QLineEdit, QDialog, QDialogButtonBox, QLabel, QApplication, QHBoxLayout
)
from .profiles import load_profiles, add_profile, delete_profile, rename_profile
from .main_window import MainWindow
import sys


class TextInputDialog(QDialog):
    def __init__(self, title: str, label: str, initial: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(label))
        self.name_edit = QLineEdit(self)
        self.name_edit.setText(initial)
        layout.addWidget(self.name_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_text(self):
        return self.name_edit.text().strip()


class ProfileSelectWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Выбор профиля (pichik)")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        self.list_widget.itemDoubleClicked.connect(self.open_selected_profile)

        # Панель кнопок
        btns = QHBoxLayout()
        btn_create = QPushButton("Создать профиль")
        btn_rename = QPushButton("Переименовать профиль")
        btn_delete = QPushButton("Удалить профиль")
        btn_open   = QPushButton("Выбрать профиль →")
        btns.addWidget(btn_create)
        btns.addWidget(btn_rename)
        btns.addWidget(btn_delete)
        btns.addWidget(btn_open)
        layout.addLayout(btns)

        btn_create.clicked.connect(self.create_profile)
        btn_rename.clicked.connect(self.rename_selected)
        btn_delete.clicked.connect(self.delete_selected)
        btn_open.clicked.connect(self.open_selected_profile)


        self.reload()

    def reload(self):
        self.list_widget.clear()
        data = load_profiles()
        for p in data.get("profiles", []):
            self.list_widget.addItem(p.get("name"))

    def _current_name(self):
        it = self.list_widget.currentItem()
        return it.text() if it else None

    def create_profile(self):
        dlg = TextInputDialog("Создать профиль", "Имя профиля:", "", self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            name = dlg.get_text()
            try:
                add_profile(name)
            except ValueError as e:
                QMessageBox.warning(self, "Ошибка", str(e)); return
            self.reload()

    def rename_selected(self):
        name = self._current_name()
        if not name:
            QMessageBox.information(self, "Переименование", "Выберите профиль.")
            return
        dlg = TextInputDialog("Переименовать профиль", "Новое имя:", name, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_name = dlg.get_text()
            try:
                rename_profile(name, new_name)
            except ValueError as e:
                QMessageBox.warning(self, "Ошибка", str(e)); return
            self.reload()

    def delete_selected(self):
        name = self._current_name()
        if not name:
            QMessageBox.information(self, "Удаление профиля", "Выберите профиль.")
            return
        reply = QMessageBox.question(
            self, "Подтверждение",
            f"Точно удалить профиль «{name}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            delete_profile(name)
            self.reload()

    def open_selected_profile(self):
        name = self._current_name()
        if not name:
            QMessageBox.information(self, "Открыть профиль", "Выберите профиль.")
            return
        self.hide()
        self.main = MainWindow(profile_name=name)
        self.main.show()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ProfileSelectWindow()
    w.show()
    sys.exit(app.exec())
