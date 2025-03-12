import sys
from PyQt6.QtWidgets import QApplication
from src.gui.main_window import ROMApp  # Asegúrate de que este archivo es el correcto

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ROMApp()
    window.show()
    sys.exit(app.exec())
