from PyQt5.QtCore import pyqtSignal, QObject

class TranscriptionSignal(QObject):
    update_transcript = pyqtSignal(str)
    update_summary = pyqtSignal(str)