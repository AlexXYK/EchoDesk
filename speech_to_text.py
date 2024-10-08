import torch
import sounddevice as sd
import numpy as np
from transformers import pipeline, WhisperProcessor
from PyQt5 import QtWidgets, QtCore, QtGui
from pynput import keyboard
import sys
import os
import threading
import queue
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class PulsingCircle(QtWidgets.QWidget):
    start_pulse_signal = QtCore.pyqtSignal(str)
    stop_pulse_signal = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(30, 30)
        self.color = QtGui.QColor(139, 0, 0)  # Dark red
        self.pulsing = False
        self.pulse_timer = QtCore.QTimer(self)
        self.pulse_timer.timeout.connect(self.pulse)
        self.pulse_value = 0
        self.pulse_direction = 1
        self.start_pulse_signal.connect(self._start_pulsing)
        self.stop_pulse_signal.connect(self._stop_pulsing)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush(self.color)
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(self.rect())

    def pulse(self):
        self.pulse_value += self.pulse_direction * 10
        if self.pulse_value >= 100 or self.pulse_value <= 0:
            self.pulse_direction *= -1
        self.color.setRed(139 + self.pulse_value)
        self.update()

    def start_pulsing(self, color='red'):
        self.start_pulse_signal.emit(color)

    def stop_pulsing(self):
        self.stop_pulse_signal.emit()

    @QtCore.pyqtSlot(str)
    def _start_pulsing(self, color):
        self.pulsing = True
        self.color = QtGui.QColor(color)
        self.pulse_timer.start(50)

    @QtCore.pyqtSlot()
    def _stop_pulsing(self):
        self.pulsing = False
        self.pulse_timer.stop()
        self.color = QtGui.QColor(139, 0, 0)  # Reset to dark red
        self.update()

    def set_size(self, size):
        sizes = {"sm": 20, "md": 30, "lg": 40}
        new_size = sizes.get(size, 30)
        self.setFixedSize(new_size, new_size)

class SpeechToTextApp(QtWidgets.QWidget):
    update_ui_signal = QtCore.pyqtSignal(bool, bool)

    def __init__(self):
        super().__init__()
        self.model_name = "openai/whisper-small"
        self.language = "en"
        self.minimal_ui_size = "md"
        self.initUI()
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = pipeline("automatic-speech-recognition", 
                              model=self.model_name, 
                              tokenizer=self.processor.tokenizer,
                              feature_extractor=self.processor.feature_extractor)
        self.recording = False
        self.samplerate = 16000
        self.audio_queue = queue.Queue()  # Queue to handle streaming audio
        self.offset = None
        self.mic_active = False
        self.transcription_thread = None
        self.transcription_lock = threading.Lock()  # Lock to avoid concurrent transcription
        self.minimized = False

        # Set initial visibility of language selector
        self.language_selector_label.setVisible(False)
        self.language_selector.setVisible(False)
        self.language_selector.setEnabled(False)

        self.update_ui_signal.connect(self._update_ui)

    def initUI(self):
        self.setGeometry(100, 100, 350, 300)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setMinimumSize(250, 200)
        
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        self.main_widget = QtWidgets.QWidget(self)
        self.main_widget.setStyleSheet("background-color: #2d2d2d; border-radius: 10px;")
        layout = QtWidgets.QVBoxLayout(self.main_widget)
        layout.setSpacing(10)

        top_layout = QtWidgets.QHBoxLayout()
        
        self.minimize_button = QtWidgets.QPushButton('-', self.main_widget)
        self.minimize_button.setFixedSize(24, 24)
        self.minimize_button.setStyleSheet("background-color: transparent; border: none; color: #ffffff; font-weight: bold; font-size: 16px;")
        self.minimize_button.clicked.connect(self.toggle_ui)
        
        close_button = QtWidgets.QPushButton('X', self.main_widget)
        close_button.setFixedSize(24, 24)
        close_button.setStyleSheet("background-color: transparent; border: none; color: #ffffff; font-weight: bold; font-size: 16px;")
        close_button.clicked.connect(self.close_app)
        
        top_layout.addWidget(self.minimize_button)
        top_layout.addStretch()
        top_layout.addWidget(close_button)
        layout.addLayout(top_layout)

        self.label = QtWidgets.QLabel('Press Ctrl+Alt+S to start/stop recording', self.main_widget)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("color: #ffffff; font-size: 14px;")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.model_selector = QtWidgets.QComboBox(self.main_widget)
        self.model_selector.addItems(["openai/whisper-tiny", "openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v3"])
        self.model_selector.setCurrentText(self.model_name)
        self.model_selector.setStyleSheet("background-color: #3e3e3e; color: #ffffff; border-radius: 5px; padding: 5px;")
        self.model_selector.currentIndexChanged.connect(self.change_model)
        layout.addWidget(self.model_selector)

        self.language_selector_label = QtWidgets.QLabel('Input Language:', self.main_widget)
        self.language_selector_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        layout.addWidget(self.language_selector_label)

        self.language_selector = QtWidgets.QComboBox(self.main_widget)
        self.language_selector.addItems(["en", "es", "fr", "de", "it", "zh", "ja", "ko", "ru"])
        self.language_selector.setCurrentText(self.language)
        self.language_selector.setStyleSheet("background-color: #3e3e3e; color: #ffffff; border-radius: 5px; padding: 5px;")
        self.language_selector.currentIndexChanged.connect(self.change_language)
        layout.addWidget(self.language_selector)

        self.minimal_ui_size_label = QtWidgets.QLabel('Minimal UI Size:', self.main_widget)
        self.minimal_ui_size_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        layout.addWidget(self.minimal_ui_size_label)

        self.minimal_ui_size_selector = QtWidgets.QComboBox(self.main_widget)
        self.minimal_ui_size_selector.addItems(["sm", "md", "lg"])
        self.minimal_ui_size_selector.setCurrentText(self.minimal_ui_size)
        self.minimal_ui_size_selector.setStyleSheet("background-color: #3e3e3e; color: #ffffff; border-radius: 5px; padding: 5px;")
        self.minimal_ui_size_selector.currentIndexChanged.connect(self.change_minimal_ui_size)
        layout.addWidget(self.minimal_ui_size_selector)

        self.progress_label = QtWidgets.QLabel('', self.main_widget)
        self.progress_label.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_label.setStyleSheet("color: #ffffff; font-size: 12px;")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)

        layout.addStretch()

        self.main_layout.addWidget(self.main_widget)

        # Create a minimal UI widget
        self.minimal_widget = PulsingCircle(self)
        self.minimal_widget.hide()

        # Add resize handles
        self.right_handle = QtWidgets.QWidget(self)
        self.right_handle.setStyleSheet("background-color: transparent;")
        self.right_handle.setCursor(QtCore.Qt.SizeHorCursor)
        self.right_handle.setFixedWidth(5)

        self.bottom_handle = QtWidgets.QWidget(self)
        self.bottom_handle.setStyleSheet("background-color: transparent;")
        self.bottom_handle.setCursor(QtCore.Qt.SizeVerCursor)
        self.bottom_handle.setFixedHeight(5)

        self.bottom_right_handle = QtWidgets.QWidget(self)
        self.bottom_right_handle.setStyleSheet("background-color: transparent;")
        self.bottom_right_handle.setCursor(QtCore.Qt.SizeFDiagCursor)
        self.bottom_right_handle.setFixedSize(5, 5)

    def toggle_ui(self):
        if not self.minimized:
            self.main_widget.hide()
            self.minimal_widget.show()
            minimal_size = self.minimal_widget.size()
            self.setFixedSize(minimal_size)  # Set fixed size to prevent resizing in minimal mode
            self.right_handle.hide()
            self.bottom_handle.hide()
            self.bottom_right_handle.hide()
            self.minimized = True
        else:
            self.expand_ui()

    def expand_ui(self):
        self.minimal_widget.hide()
        self.main_widget.show()
        self.setFixedSize(350, 300)  # Set initial size
        self.setMinimumSize(250, 200)  # Set minimum size
        self.setMaximumSize(16777215, 16777215)  # Reset maximum size (essentially removing it)
        self.right_handle.show()
        self.bottom_handle.show()
        self.bottom_right_handle.show()
        self.minimized = False

    def change_minimal_ui_size(self):
        self.minimal_ui_size = self.minimal_ui_size_selector.currentText()
        self.minimal_widget.set_size(self.minimal_ui_size)
        if self.minimized:
            minimal_size = self.minimal_widget.size()
            self.setFixedSize(minimal_size)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.right_handle.setGeometry(self.width() - 5, 0, 5, self.height())
        self.bottom_handle.setGeometry(0, self.height() - 5, self.width(), 5)
        self.bottom_right_handle.setGeometry(self.width() - 5, self.height() - 5, 5, 5)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.offset = event.pos()
            if self.right_handle.geometry().contains(event.pos()):
                self.resizing = 'right'
            elif self.bottom_handle.geometry().contains(event.pos()):
                self.resizing = 'bottom'
            elif self.bottom_right_handle.geometry().contains(event.pos()):
                self.resizing = 'bottom_right'
            else:
                self.resizing = None
        elif event.button() == QtCore.Qt.RightButton and self.minimized:
            self.expand_ui()

    def mouseMoveEvent(self, event):
        if self.offset is not None and event.buttons() == QtCore.Qt.LeftButton:
            if self.resizing:
                if self.resizing == 'right':
                    width = max(self.minimumWidth(), event.pos().x())
                    self.resize(width, self.height())
                elif self.resizing == 'bottom':
                    height = max(self.minimumHeight(), event.pos().y())
                    self.resize(self.width(), height)
                elif self.resizing == 'bottom_right':
                    width = max(self.minimumWidth(), event.pos().x())
                    height = max(self.minimumHeight(), event.pos().y())
                    self.resize(width, height)
            else:
                self.move(self.pos() + event.pos() - self.offset)

    def mouseReleaseEvent(self, event):
        self.offset = None
        self.resizing = None

    def change_model(self):
        self.model_name = self.model_selector.currentText()
        self.label.setText(f'Loading {self.model_name} model...')
        QtWidgets.QApplication.processEvents()  # Update label immediately
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = pipeline("automatic-speech-recognition", 
                              model=self.model_name, 
                              tokenizer=self.processor.tokenizer,
                              feature_extractor=self.processor.feature_extractor)
        self.label.setText('Model loaded. Press Ctrl+Alt+S to start/stop recording')

        # Handle language selector visibility based on model
        if self.model_name == "openai/whisper-large-v3":
            self.language_selector_label.setVisible(True)
            self.language_selector.setVisible(True)
            self.language_selector.setEnabled(True)
        else:
            self.language_selector_label.setVisible(False)
            self.language_selector.setVisible(False)
            self.language_selector.setCurrentText("en")
            self.language_selector.setEnabled(False)

    def change_language(self):
        self.language = self.language_selector.currentText()
        self.label.setText(f'Language set to {self.language}')
        QtWidgets.QApplication.processEvents()  # Update label immediately
        if self.model_name == "openai/whisper-large-v3-turbo":
            self.model = pipeline("automatic-speech-recognition", model=self.model_name)

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.recording = True
        self.mic_active = True
        self.audio_queue.queue.clear()  # Clear any existing audio data
        default_device = get_default_input_device()
        self.stream = sd.InputStream(callback=self.audio_callback, 
                                     channels=1, 
                                     samplerate=self.samplerate,
                                     device=default_device)
        self.stream.start()
        self.transcription_thread = threading.Thread(target=self.run_transcription)
        self.transcription_thread.start()
        self.update_mic_button()
        self.label.setText('Recording... Press Ctrl+Alt+S to stop.')

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.mic_active = False
        self.update_mic_button()
        self.stream.stop()
        self.stream.close()
        self.label.setText('Finalizing transcription...')
        self.progress_label.setText('Please wait, processing remaining audio...')
        
        # Wait for the transcription thread to finish
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join()

        self.update_mic_button()  # Update UI after transcription is done

    def run_transcription(self):
        audio_buffer = np.array([], dtype=np.float32)
        chunk_duration = 5  # Process 5 seconds of audio at a time
        chunk_samples = int(self.samplerate * chunk_duration)

        while self.recording or not self.audio_queue.empty():
            try:
                while len(audio_buffer) < chunk_samples and (self.recording or not self.audio_queue.empty()):
                    try:
                        audio_chunk = self.audio_queue.get(timeout=1)
                        audio_buffer = np.concatenate((audio_buffer, audio_chunk))
                    except queue.Empty:
                        break  # Break if no more audio data is available

                if len(audio_buffer) > 0:
                    with self.transcription_lock:
                        QtCore.QMetaObject.invokeMethod(self, 'update_progress', 
                                                        QtCore.Qt.QueuedConnection, 
                                                        QtCore.Q_ARG(str, 'Processing audio...'))
                        
                        # Ensure audio_buffer is a single-channel numpy array
                        if audio_buffer.ndim > 1:
                            audio_buffer = audio_buffer[:, 0]
                        
                        # Transcribe using the raw audio buffer
                        transcription = self.model(audio_buffer)
                        
                        QtCore.QMetaObject.invokeMethod(self, 'transcription_done', 
                                                        QtCore.Qt.QueuedConnection, 
                                                        QtCore.Q_ARG(str, transcription['text']))

                    # Clear the processed audio
                    audio_buffer = np.array([], dtype=np.float32)

            except Exception as e:
                print(f"Error during transcription: {str(e)}")
                QtCore.QMetaObject.invokeMethod(self, 'update_progress', 
                                                QtCore.Qt.QueuedConnection, 
                                                QtCore.Q_ARG(str, f'Error: {str(e)}'))

        QtCore.QMetaObject.invokeMethod(self, 'update_progress', 
                                        QtCore.Qt.QueuedConnection, 
                                        QtCore.Q_ARG(str, 'Transcription finished'))

    @QtCore.pyqtSlot(str)
    def update_progress(self, message):
        self.progress_label.setText(message)

    @QtCore.pyqtSlot(str)
    def transcription_done(self, text):
        self.insert_text(text)
        self.label.setText('Press Ctrl+Alt+S to start/stop recording')

    def update_mic_button(self):
        is_transcribing = self.transcription_thread is not None and self.transcription_thread.is_alive()
        self.update_ui_signal.emit(self.mic_active, is_transcribing)

    @QtCore.pyqtSlot(bool, bool)
    def _update_ui(self, mic_active, transcribing):
        if self.minimized:
            if mic_active:
                self.minimal_widget.start_pulsing('red')
            elif transcribing:
                self.minimal_widget.start_pulsing('orange')
            else:
                self.minimal_widget.stop_pulsing()

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            # Ensure we're working with a single-channel numpy array
            audio_data = indata.copy()
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]  # Take the first channel if multi-channel
            self.audio_queue.put(audio_data)

    def insert_text(self, text):
        # Function to insert text into the active input field
        controller = keyboard.Controller()
        controller.type(text)

    def close_app(self):
        QtWidgets.QApplication.quit()

def get_default_input_device():
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    if isinstance(default_input, str):
        for i, dev in enumerate(devices):
            if dev['name'] == default_input:
                return i
    return default_input

if __name__ == '__main__':
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")
    elif sys.platform == 'linux':
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    
    app = QtWidgets.QApplication(sys.argv)
    ex = SpeechToTextApp()
    ex.show()

    listener = keyboard.GlobalHotKeys({'<ctrl>+<alt>+s': ex.toggle_recording})
    listener.start()

    sys.exit(app.exec_())