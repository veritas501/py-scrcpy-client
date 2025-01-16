import functools
import os
import re
import socket
import struct
import threading
import time
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union
from pathlib import Path

import numpy as np
from adbutils import AdbConnection, AdbDevice, AdbError, Network, adb
from av.codec import CodecContext
from av.error import InvalidDataError

from .const import (
    EVENT_DISCONNECT,
    EVENT_FRAME,
    EVENT_INIT,
    LOCK_SCREEN_ORIENTATION_UNLOCKED,
)
from .control import ControlSender


SCRCPY_SERVER_FILENAME = "scrcpy-server-v3.1"
SCRCPT_SERVER_VERSION = "3.1"
SCRCPY_SERVER_FILEPATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), SCRCPY_SERVER_FILENAME
)


def need_server(func):
    @functools.wraps(func)  # 保留原始函数的元信息
    def wrapper(this, *args, **kwargs):
        # 在调用原始函数之前，先调用 push_server
        this.push_server()
        # 调用原始函数
        return func(this, *args, **kwargs)

    return wrapper


class Client:
    def __init__(
        self,
        device: Optional[Union[AdbDevice, str, any]] = None,
        max_width: int = 0,
        bitrate: int = 0,
        max_fps: int = 0,
        flip: bool = False,
        block_frame: bool = False,
        stay_awake: bool = True,
        lock_screen_orientation: int = LOCK_SCREEN_ORIENTATION_UNLOCKED,
        connection_timeout: int = 3000,
        encoder_name: Optional[str] = None,
        codec_name: Optional[str] = None,
        new_display: Union[str, bool] = False,
        start_app: Optional[str] = None,
    ):
        """
        Create a scrcpy client, this client won't be started until you call the start function

        Args:
            device: Android device, select first one if none, from serial if str
            max_width: frame width that will be broadcast from android server
            bitrate: bitrate
            max_fps: maximum fps, 0 means not limited (supported after android 10)
            flip: flip the video
            block_frame: only return nonempty frames
            stay_awake: keep Android device awake
            lock_screen_orientation: lock screen orientation, LOCK_SCREEN_ORIENTATION_*
            connection_timeout: timeout for connection, unit is ms
            encoder_name: encoder name, enum: [OMX.google.h264.encoder, OMX.qcom.video.encoder.avc, c2.qti.avc.encoder, c2.android.avc.encoder], default is None (Auto)
            codec_name: codec name, enum: [h264, h265, av1], default is None (Auto)
        """
        # Check Params
        assert max_width >= 0, "max_width must be greater than or equal to 0"
        assert bitrate >= 0, "bitrate must be greater than or equal to 0"
        assert max_fps >= 0, "max_fps must be greater than or equal to 0"
        assert -1 <= lock_screen_orientation <= 3, (
            "lock_screen_orientation must be LOCK_SCREEN_ORIENTATION_*"
        )
        assert connection_timeout >= 0, (
            "connection_timeout must be greater than or equal to 0"
        )
        assert encoder_name in [
            None,
            "OMX.google.h264.encoder",
            "OMX.qcom.video.encoder.avc",
            "c2.qti.avc.encoder",
            "c2.android.avc.encoder",
        ]
        assert codec_name in [None, "h264", "h265", "av1"]

        # Check new_display format
        if new_display:
            if isinstance(new_display, bool) and new_display:
                # Use default value, no need to check
                new_display = ""
            else:
                # Check format [<width>x<height>][/<dpi>]
                parts = new_display.split("/")
                if len(parts) > 2:
                    raise ValueError(
                        "Invalid new_display format. Expected format: [<width>x<height>][/<dpi>]"
                    )

                # Check resolution part
                if parts[0]:
                    width, height = map(int, parts[0].split("x"))
                    assert width > 0 and height > 0, (
                        "Invalid resolution format. Expected format: <width>x<height>"
                    )

                # Check DPI part
                if len(parts) == 2:
                    dpi = int(parts[1])
                    assert dpi > 0, "DPI must be a positive integer"

        # Params
        self.flip = flip
        self.max_width = max_width
        self.bitrate = bitrate
        self.max_fps = max_fps
        self.block_frame = block_frame
        self.stay_awake = stay_awake
        self.lock_screen_orientation = lock_screen_orientation
        self.connection_timeout = connection_timeout
        self.encoder_name = encoder_name
        self.codec_name = codec_name
        self.new_display = new_display
        self.start_app = start_app

        # Connect to device
        if device is None:
            device = adb.device_list()[0]
        elif isinstance(device, str):
            device = adb.device(serial=device)

        self.device = device
        self.listeners = dict(frame=[], init=[], disconnect=[])

        # User accessible
        self.last_frame: Optional[np.ndarray] = None
        self.resolution: Optional[Tuple[int, int]] = None
        self.device_name: Optional[str] = None
        self.control = ControlSender(self)

        # Need to destroy
        self.alive = False
        self.__server_stream: Optional[AdbConnection] = None
        self.__video_socket: Optional[socket.socket] = None
        self.control_socket: Optional[socket.socket] = None
        self.control_socket_lock = threading.Lock()

        # Available if start with threaded or daemon_threaded
        self.stream_loop_thread = None

        # remote server path
        self.remote_server_path: Optional[Text] = None

    def __init_server_connection(self) -> None:
        """
        Connect to android server, there will be two sockets, video and control socket.
        This method will set: video_socket, control_socket, resolution variables
        """
        for _ in range(self.connection_timeout // 100):
            try:
                self.__video_socket = self.device.create_connection(
                    Network.LOCAL_ABSTRACT, "scrcpy"
                )
                break
            except AdbError:
                sleep(0.1)
                pass
        else:
            raise ConnectionError("Failed to connect scrcpy-server after 3 seconds")

        dummy_byte = self.__video_socket.recv(1)
        if not len(dummy_byte) or dummy_byte != b"\x00":
            raise ConnectionError("Did not receive Dummy Byte!")

        self.control_socket = self.device.create_connection(
            Network.LOCAL_ABSTRACT, "scrcpy"
        )
        self.device_name = self.__video_socket.recv(64).decode("utf-8").rstrip("\x00")
        if not len(self.device_name):
            raise ConnectionError("Did not receive Device Name!")

        res = self.__video_socket.recv(4)
        self.resolution = struct.unpack(">HH", res)
        self.__video_socket.setblocking(False)

    def push_server(self, remote_path="/data/local/tmp") -> str:
        remote_filename = (Path(remote_path) / SCRCPY_SERVER_FILENAME).as_posix()
        if not self.device.sync.exists(remote_filename):
            self.device.sync.push(SCRCPY_SERVER_FILEPATH, remote_filename)
        self.remote_server_path = remote_filename
        return remote_filename

    @need_server
    def start_server(self) -> None:
        """
        Start scrcpy server on android device
        """

        commands = [
            f"CLASSPATH={self.remote_server_path}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            str(SCRCPT_SERVER_VERSION),
            "log_level=info",
            "tunnel_forward=true",
            "send_frame_meta=false",
            "audio=false",
            "control=true",
            "show_touches=false",
            "stay_awake=true",
            "power_off_on_close=false",
            "clipboard_autosync=false",
            "cleanup=false",
            "screen_off_timeout=2147483000",
        ]

        if self.max_width:
            commands.append(f"max_size={self.max_width}")
        if self.max_fps:
            commands.append(f"max_fps={self.max_fps}")
        if self.bitrate:
            commands.append(f"video_bit_rate={self.bitrate}")
        if self.encoder_name:
            commands.append(f"video_encoder={self.encoder_name}")
        if self.codec_name:
            commands.append(f"video_codec={self.codec_name}")
        else:
            commands.append("video_codec=h264")
        if self.new_display or self.new_display == "":
            commands.append(f"new_display={self.new_display}")

        self.__server_stream: AdbConnection = self.device.shell(
            commands,
            stream=True,
        )

        # Wait for server to start
        self.__server_stream.read(10)

    @need_server
    def list_codec_and_encoder(self) -> List[Dict[Text, Text]]:
        """list valid codec and encoder"""
        commands = [
            f"CLASSPATH={self.remote_server_path}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            str(SCRCPT_SERVER_VERSION),
            "list_encoders=true",
            "cleanup=false",
        ]
        res = self.device.shell(commands)
        pattern = r"--video-codec=(\w+)\s+--video-encoder=([\w\.]+)"
        matches = re.findall(pattern, res)
        return [{"codec": codec, "encoder": encoder} for codec, encoder in matches]

    @need_server
    def list_apps(self) -> Dict[Text, Text]:
        """list apps"""
        commands = [
            f"CLASSPATH={self.remote_server_path}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            str(SCRCPT_SERVER_VERSION),
            "list_apps=true",
            "cleanup=false",
        ]
        res = self.device.shell(commands)
        pattern = r"\s+[-*]\s+(.*?)\s+(([a-z_][a-z0-9_]*)(\.[a-z_][a-z0-9_]*)*)"
        matches = re.findall(pattern, res)
        return {m[1]: m[0] for m in matches}

    @need_server
    def start(self, threaded: bool = False, daemon_threaded: bool = False) -> None:
        """
        Start listening video stream

        Args:
            threaded: Run stream loop in a different thread to avoid blocking
            daemon_threaded: Run stream loop in a daemon thread to avoid blocking
        """
        assert self.alive is False
        print("self.start_app",self.start_app)
        self.start_server()
        self.__init_server_connection()
        self.alive = True
        self.__send_to_listeners(EVENT_INIT)

        if self.start_app:
            self.control.start_app(self.start_app)

        if threaded or daemon_threaded:
            self.stream_loop_thread = threading.Thread(
                target=self.__stream_loop, daemon=daemon_threaded
            )
            self.stream_loop_thread.start()
        else:
            self.__stream_loop()

    def stop(self) -> None:
        """
        Stop listening (both threaded and blocked)
        """
        self.alive = False
        if self.__server_stream is not None:
            try:
                self.__server_stream.close()
            except Exception:
                pass

        if self.control_socket is not None:
            try:
                self.control_socket.close()
            except Exception:
                pass

        if self.__video_socket is not None:
            try:
                self.__video_socket.close()
            except Exception:
                pass

    def snapshot(self):
        """take device screen snapshot"""
        return self.last_frame

    def __stream_loop(self) -> None:
        """
        Core loop for video parsing
        """
        codec = CodecContext.create("h264", "r")
        while self.alive:
            try:
                raw_h264 = self.__video_socket.recv(0x4000)
                if raw_h264 == b"":
                    raise ConnectionError("Video stream is disconnected")
                packets = codec.parse(raw_h264)
                for packet in packets:
                    frames = codec.decode(packet)
                    for frame in frames:
                        if not frame:
                            continue
                        frame = frame.to_ndarray(format="bgr24")
                        if self.flip:
                            frame = frame[:, ::-1, :]
                            frame = np.ascontiguousarray(frame)
                        self.last_frame = frame
                        self.resolution = (frame.shape[1], frame.shape[0])
                        self.__send_to_listeners(EVENT_FRAME, frame)
            except (BlockingIOError, InvalidDataError):
                time.sleep(0.01)
                if not self.block_frame:
                    self.__send_to_listeners(EVENT_FRAME, None)
            except (ConnectionError, OSError) as e:  # Socket Closed
                if self.alive:
                    self.__send_to_listeners(EVENT_DISCONNECT)
                    self.stop()
                    raise e

    def add_listener(self, cls: str, listener: Callable[..., Any]) -> None:
        """
        Add a video listener

        Args:
            cls: Listener category, support: init, frame
            listener: A function to receive frame np.ndarray
        """
        self.listeners[cls].append(listener)

    def remove_listener(self, cls: str, listener: Callable[..., Any]) -> None:
        """
        Remove a video listener

        Args:
            cls: Listener category, support: init, frame
            listener: A function to receive frame np.ndarray
        """
        self.listeners[cls].remove(listener)

    def __send_to_listeners(self, cls: str, *args, **kwargs) -> None:
        """
        Send event to listeners

        Args:
            cls: Listener type
            *args: Other arguments
            *kwargs: Other arguments
        """
        for fun in self.listeners[cls]:
            fun(*args, **kwargs)
