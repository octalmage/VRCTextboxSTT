import speech_recognition as sr
import numpy as np
import pyaudio
from queue import Queue
from config import listener_config
import traceback
import logging

log = logging.getLogger(__name__)

class ListenHandler(object):
    def __init__(self, config: listener_config) -> None:
        self.config: listener_config = config
        self.rec = sr.Recognizer()
        self.source = sr.Microphone(sample_rate=16000)
        self.data_queue = Queue()
        self.stop_listening = None
        self.set_config(config)

    def set_config(self, config: listener_config):
        self.config = config

        self.rec.dynamic_energy_threshold = bool(self.config.dynamic_energy_threshold)
        self.rec.energy_threshold = self.config.energy_threshold
        self.rec.pause_threshold = self.config.pause_threshold
        resolved_index = self._reconcile_device_index()
        self.source.device_index = int(resolved_index) if resolved_index is not None else None

    def _reconcile_device_index(self):
        """Resolve the saved microphone after device renumbering."""
        saved_index = self.config.microphone_index
        saved_name = getattr(self.config, "microphone_name", None)
        if saved_index is None:
            return None
        if not saved_name:
            return saved_index
        try:
            p = pyaudio.PyAudio()
            try:
                numdev = p.get_host_api_info_by_index(0).get("deviceCount")
                current_name = None
                if 0 <= saved_index < numdev:
                    info = p.get_device_info_by_host_api_device_index(0, saved_index)
                    if info.get("maxInputChannels") > 0:
                        current_name = info.get("name")
                if current_name == saved_name:
                    return saved_index
                for i in range(numdev):
                    try:
                        info = p.get_device_info_by_host_api_device_index(0, i)
                        if info.get("maxInputChannels") > 0 and info.get("name") == saved_name:
                            log.info(f"Microphone index for '{saved_name}' changed: {saved_index} -> {i}")
                            return i
                    except Exception:
                        continue
            finally:
                p.terminate()
        except Exception:
            log.error("Error reconciling microphone device index: ")
            log.error(traceback.format_exc())
        return saved_index

    def listen_once(self) -> np.ndarray:
        """
        Listens once and returns the audio data as a np array.
        """

        with self.source:
            try:
                _audio = self.rec.listen(self.source, timeout=self.config.timeout_time)
            except sr.WaitTimeoutError:
                return None
            except Exception:
                log.error("Error listening: ")
                log.error(traceback.format_exc())
                return None

            return _audio.get_raw_data()
        
    def start_listen_background(self, no_phrase_time_limit = False) -> None:
        """
        Listens in the background and puts the audio data into a queue.
        """
        def record_callback(_, audio:sr.AudioData) -> None:
            try:
                _data = audio.get_raw_data()
                self.data_queue.put(_data)
            except Exception:
                log.error("Error in record callback data: ")
                log.error(traceback.format_exc())

        try:
            self.stop_listening = self.rec.listen_in_background(self.source, record_callback, phrase_time_limit=self.config.phrase_time_limit if not no_phrase_time_limit else None)
        except Exception:
            log.error("Error starting background listener: ")
            log.error(traceback.format_exc())

    def stop_listen_background(self) -> None:
        try:
            self.stop_listening()
        except Exception:
            log.error("Error stopping listening: ")
            log.error(traceback.format_exc())
        try:
            self.clear_queue()
        except Exception:
            log.error("Error clearing queue: ")
            log.error(traceback.format_exc())

    def raw_to_np(self, raw_data:bytes) -> np.ndarray:
        """Convert raw audio from signed 16-bit integer to signed 16-bit float and return as np array."""
        try:
            return np.frombuffer(raw_data, np.int16).flatten().astype(np.float32) / 32768.0
        except Exception:
            log.error("Error converting raw data to np: ")
            log.error(traceback.format_exc())
            return None
    
    def clear_queue(self) -> None:
        try:
            self.data_queue.queue.clear()
        except Exception:
            log.error("Error clearing queue: ")
            log.error(traceback.format_exc())
    
    def get_queue_data(self) -> bytes:
        _data = bytes()
        while not self.data_queue.empty():
            _data += self.data_queue.get()
        return _data
    
    def get_queue_empty(self) -> bool:
        return self.data_queue.empty()

    def get_energy_threshold(self) -> int:
        try:
            with self.source:
                _last = self.rec.energy_threshold
                self.rec.adjust_for_ambient_noise(self.source, 5)
                _value = round(self.rec.energy_threshold) + 100
                self.rec.energy_threshold = _last
                return _value
        except Exception:
            log.error("Error getting energy threshold: ")
            log.error(traceback.format_exc())
            return 200
