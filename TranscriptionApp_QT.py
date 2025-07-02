import sys
import threading
import os
# from dotenv import load_dotenv, set_key
# from Constants import ENV_FILE
from APIKeyManager import load_api_key, save_api_key
from Dialogs.InitialInstructionsDialog import InitialInstructionsDialog
from Dialogs.OpenAIApiKeyDialog import OpenAIApiKeyDialog
from TranscriptionSignal import TranscriptionSignal
from TranscriptionWorker import run_transcription
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QDialog, QPushButton, QTextEdit, QMessageBox, QWidget, 
    QMenuBar, QHBoxLayout, QAction, QLabel, QCheckBox, QDoubleSpinBox
)


class TranscriptionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meeting Transcriber")
        self.setGeometry(1100, 100, 700, 1000)

        self.layout = QVBoxLayout()

        # Menu Bar
        self.menubar = QMenuBar(self)
        # Settings Menu
        self.settings_menu = self.menubar.addMenu("Settings")
        # Api Key
        self.api_key_action = QAction("OpenAI API Key", self)
        self.api_key_action.triggered.connect(self.show_api_key_dialog)
        self.settings_menu.addAction(self.api_key_action)
        # Help Menu
        self.help_menu = self.menubar.addMenu("Help")
        self.instructions_action = QAction("Instructions on Initial Setup", self)
        self.instructions_action.triggered.connect(self.show_instructions_dialog)
        self.help_menu.addAction(self.instructions_action)

        # Header row layout for title and API status
        header_layout = QHBoxLayout()
        # self.title_label = QLabel("📜🪶 Transcription")
        self.title_label = QLabel("🎧 Transcription")
        self.title_label.setAlignment(Qt.AlignLeft)
        self.api_status_label = QLabel("🔴 API Key not set")
        self.api_status_label.setAlignment(Qt.AlignRight)
        self.update_api_status()
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.api_status_label)

        # Transcriotion
        self.transcription_text = QTextEdit(self)
        self.transcription_text.setReadOnly(True)
        self.transcription_text.setPlaceholderText("Transcription will appear here...")

        # Summary
        # self.summary_title_label = QLabel("✨ Summary")
        self.summary_title_label = QLabel("🤖 Summary")
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("Summary will appear here...")

        # Bottom buttons
        # Transciption Controls
        self.start_btn = QPushButton("Start Transcription", self)
        self.stop_btn = QPushButton("Stop Transcription", self)
        self.stop_btn.setEnabled(False)

        # Summary Controls
        # # Create a row layout for the summarize button and toggle
        self.summarize_btn = QPushButton("Summarize Chunk Now", self)
        self.summarize_btn.clicked.connect(self.manual_summarize)
        # Checkbox
        self.auto_summary_checkbox = QCheckBox("Auto Summary:")
        self.auto_summary_checkbox.setChecked(True)
        self.auto_summary_checkbox.stateChanged.connect(self.toggle_auto_summary)
        # Spinbox to control summary interval in minutes
        self.summary_interval_spinbox = QDoubleSpinBox()
        self.summary_interval_spinbox.setRange(0.1, 60.0)
        self.summary_interval_spinbox.setSingleStep(1.0)
        self.summary_interval_spinbox.setValue(1.0)  # default = one minute
        self.summary_interval_spinbox.setPrefix("Every ")
        self.summary_interval_spinbox.setSuffix(" min")
        self.summary_interval_spinbox.setToolTip("How many minutes between auto summaries")
        # Connect the state to enable/disable the spinbox based on teh checkbox state
        self.auto_summary_checkbox.stateChanged.connect(
            lambda state: self.summary_interval_spinbox.setEnabled(state == Qt.Checked)
        )
        self.summary_interval_spinbox.setEnabled(self.auto_summary_checkbox.isChecked())
        # Layout row: summarize button on left, checkbox + spinbox column on right
        summary_controls_row = QHBoxLayout()
        # Left: summarize now button
        summary_controls_row.addWidget(self.summarize_btn)
        # Right: auto-summary + interval stacked vertically
        auto_summary_column = QVBoxLayout()
        auto_summary_column.addWidget(self.auto_summary_checkbox)
        auto_summary_column.addWidget(self.summary_interval_spinbox)
        summary_controls_row.addStretch()
        summary_controls_row.addLayout(auto_summary_column)
        
        # Layout
        self.layout.setMenuBar(self.menubar)
        self.layout.addLayout(header_layout)
        self.layout.addWidget(self.transcription_text)
        self.layout.addWidget(self.summary_title_label)
        self.layout.addWidget(self.summary_text)
        # self.layout.addLayout(summary_controls_row)
        self.layout.addWidget(self.start_btn)
        self.layout.addWidget(self.stop_btn)
        self.layout.addLayout(summary_controls_row)
        self.setLayout(self.layout)

        # Signals and button interactions
        self.signals = TranscriptionSignal()
        self.signals.update_transcript.connect(self.append_transcript)
        self.signals.update_summary.connect(self.set_summary)

        self.start_btn.clicked.connect(self.start_transcription)
        self.stop_btn.clicked.connect(self.stop_transcription)

        # Program State
        self.auto_summary_enabled = True
        self.transcription_thread = None
        self.running = False
        

    def append_transcript(self, text):
        self.transcription_text.append(text)

    def set_summary(self, text):
        self.summary_text.append(text)
        self.summary_text.append('\n')


    def update_api_status(self):
        key = load_api_key()
        if key:
            self.api_status_label.setText("🟢 API Key set")
        else:
            self.api_status_label.setText("🔴 API Key not set")

    def show_api_key_dialog(self):
        dialog = OpenAIApiKeyDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            save_api_key(dialog.get_api_key())
            self.update_api_status()
    
    def show_instructions_dialog(self):
        dialog = InitialInstructionsDialog(self)
        dialog.exec_()

    def start_transcription(self):
        api_key = load_api_key()
        if not api_key:
            QMessageBox.warning(
                self,
                "Missing API Key",
                "You must set your OpenAI API key before starting transcription.\nGo to Settings > OpenAI API Key."
            )
            return

        self.running = True
        interval_seconds = lambda: self.summary_interval_spinbox.value() * 60
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.transcription_thread = threading.Thread(
            target=run_transcription,
            args=(
                self.signals.update_transcript,
                self.signals.update_summary,
                lambda: self.running,
                lambda: self.auto_summary_enabled,
                interval_seconds 
            )
        )
        self.transcription_thread.start()

    def stop_transcription(self):
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.transcription_thread:
            self.transcription_thread.join()
            self.transcription_thread = None

    def manual_summarize(self):
        # Reuse the same logic as in the transcription_worker
        import TranscriptionWorker
        summary = TranscriptionWorker.summarize_buffer()
        if summary:
            self.set_summary(summary)

    def toggle_auto_summary(self, state):
        self.auto_summary_enabled = (state == Qt.Checked)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TranscriptionApp()
    window.show()
    sys.exit(app.exec_())
