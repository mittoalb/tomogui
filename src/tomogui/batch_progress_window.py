from PyQt5.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QLabel, QPushButton

class ProgressWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Progress")
        self.setFixedSize(400, 200)

        # Layout
        layout = QVBoxLayout()

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)  # Set max value dynamically later
        layout.addWidget(self.progress_bar)

        # Queue Count Label
        self.queue_label = QLabel("Queue: 0 waiting", self)
        layout.addWidget(self.queue_label)

        # Stop Queue Button
        self.stop_button = QPushButton("Stop Queue", self)
        self.stop_button.clicked.connect(self.stop_queue)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

    def update_progress(self, value, queue_count):
        """Update progress bar and queue count."""
        self.progress_bar.setValue(value)
        self.queue_label.setText(f"Queue: {queue_count} waiting")

    def stop_queue(self):
        """Handle stopping the queue."""
        # Add logic to stop the queue here
        print("Queue stopped!")
        self.close()