from APIKeyManager import load_api_key, save_api_key
from PyQt5.QtWidgets import (QVBoxLayout, QDialog, QDialogButtonBox, QLabel, QLineEdit)

class OpenAIApiKeyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set OpenAI API Key")
        self.layout = QVBoxLayout(self)

        self.label = QLabel("Enter your OpenAI API Key:")
        self.input = QLineEdit()
        self.input.setEchoMode(QLineEdit.Password)
        self.input.setText(load_api_key())

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.input)
        self.layout.addWidget(self.buttons)

    def get_api_key(self):
        return self.input.text()
    