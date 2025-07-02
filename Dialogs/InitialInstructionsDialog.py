from PyQt5.QtWidgets import (QVBoxLayout, QDialog, QTextEdit)

class InitialInstructionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Instructions on Initial Setup")
        self.resize(600, 400)

        layout = QVBoxLayout()
        self.instructions_text = QTextEdit()
        self.instructions_text.setReadOnly(True)

        # You can format this in rich HTML or convert from Markdown manually
        self.instructions_text.setHtml("""
        <h2>🛠️ Initial Setup Instructions</h2>
        <p>To use this app, follow these steps:</p>
        <ol>
            <li><b>Install dependencies:</b> Make sure <code>whisper</code>, <code>sounddevice</code>, <code>openai</code>, and <code>PyQt5</code> are installed.</li>
            <li><b>Audio routing:</b> On Windows, install <a href="https://vb-audio.com/Cable/">VB-Cable</a> or <a href="https://vb-audio.com/Voicemeeter/">VoiceMeeter Banana</a> to capture system audio.</li>
            <li><b>API Key:</b> Use <i>Settings → OpenAI API Key</i> to enter your key before starting transcription.</li>
            <li><b>Press “Start Transcription”</b> to begin recording and summarizing.</li>
        </ol>
        <p>Auto summary and interval can be configured using the checkboxes and spinbox in the UI.</p>
        <p>Summaries use the GPT-3.5 model from OpenAI.</p>
        """)

        layout.addWidget(self.instructions_text)
        self.setLayout(layout)