from PyQt5.QtWidgets import QDialog, QVBoxLayout, QGroupBox, QProgressBar, QLabel, QPushButton
from PyQt5.QtCore import QTimer, pyqtSignal

class ProgressWindow(QDialog):
    stop_requested = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Batch Progress")
        self.setGeometry(200, 200, 400, 200)

        # Progress section inside the new window
        progress_group = QGroupBox("Batch Progress")
        progress_layout = QVBoxLayout()

        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setValue(0)  # Initialize to 0
        progress_layout.addWidget(self.batch_progress_bar)

        self.batch_status_label = QLabel("Ready")
        progress_layout.addWidget(self.batch_status_label)

        self.batch_queue_label = QLabel("Queue: 0 jobs waiting")
        progress_layout.addWidget(self.batch_queue_label)

        self.batch_stop_btn = QPushButton("Stop Batch")
        self.batch_stop_btn.setEnabled(False)
        progress_layout.addWidget(self.batch_stop_btn)

        # Set the layout of the progress window
        progress_group.setLayout(progress_layout)
        dialog_layout = QVBoxLayout(self)
        dialog_layout.addWidget(progress_group)
        self.setLayout(dialog_layout)

        # Connect stop button
        self.batch_stop_btn.clicked.connect(self.stop_requested.emit)

    def set_running(self, running: bool):
        """Enable/disable the stop button depending on batch state."""
        self.batch_stop_btn.setEnabled(bool(running))

    def set_progress(self, value: int):
        try:
            self.batch_progress_bar.setValue(int(value))
        except Exception:
            pass

    def set_status(self, text: str):
        self.batch_status_label.setText(str(text))

    def set_queue(self, queue_size: int):
        self.batch_queue_label.setText(f"Queue: {int(queue_size)} jobs waiting")
