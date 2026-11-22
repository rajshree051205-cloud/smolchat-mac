"""SmolChat for macOS – entry point."""

from __future__ import annotations

import logging
import os
import signal
import sys

from PySide6.QtCore import QMetaObject, Qt, QTimer, QUrl
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle

from app.backend import Backend, IconProvider
from app.config import APP_NAME, APP_VERSION, ORG_NAME, AppPaths, configure_logging, resource_root
from app.instance_lock import InstanceLock, activation_name


def _notify_running_instance(name: str) -> bool:
    """Ask the instance that owns this data folder to bring its window forward."""
    sock = QLocalSocket()
    sock.connectToServer(name)
    if not sock.waitForConnected(1000):
        return False
    sock.write(b"raise\n")
    sock.waitForBytesWritten(1000)
    sock.disconnectFromServer()
    return True


def _listen_for_activation(name: str, windows: list) -> QLocalServer:
    """Raise the main window whenever a second launch pings us.

    Called right after the data lock is taken, before the slow start-up work, so a
    second launch never finds the lock held but nobody listening. ``windows`` is
    filled with the main window once it exists; pings before that are ignored.
    """
    QLocalServer.removeServer(name)  # safe: we hold the lock, so any socket file is stale
    server = QLocalServer()
    server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    if not server.listen(name):
        logging.getLogger(__name__).warning("Activation socket failed: %s", server.errorString())

    def on_connection() -> None:
        conn = server.nextPendingConnection()
        if conn is None:
            return
        conn.disconnected.connect(conn.deleteLater)
        if windows:
            windows[0].show()
            windows[0].raise_()
            windows[0].requestActivate()

    server.newConnection.connect(on_connection)
    return server


def main() -> int:
    paths = AppPaths.default()
    paths.ensure()
    configure_logging(paths)

    # The Basic style is fully customisable; the native macOS style is not.
    QQuickStyle.setStyle("Basic")
    os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")

    app = QGuiApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationVersion(APP_VERSION)
    icon = QIcon()
    for size in (157, 314, 627, 1254):
        icon.addFile(str(resource_root() / "resources" / "assets" / "logo" / f"logo-{size}.png"))
    app.setWindowIcon(icon)  # also the Dock icon on macOS

    # Only one instance may use a data folder (SMOLCHAT_HOME gives separate folders).
    # The OS drops the lock when the process exits, so a crash never leaves it stuck.
    lock = InstanceLock(paths.data)
    socket_name = activation_name(paths.data)
    if not lock.acquire():
        if not _notify_running_instance(socket_name):
            print(f"SmolChat is already running for the data folder {paths.data}", file=sys.stderr)
        return 0
    main_window: list = []
    activation_server = _listen_for_activation(socket_name, main_window)

    backend = Backend(paths)

    def apply_color_scheme() -> None:
        # Native chrome (title bar, file dialogs) follows the app theme;
        # "system" hands control back to macOS.
        hints = app.styleHints()
        theme = backend.settings.theme
        if theme == "system":
            hints.unsetColorScheme()
        else:
            hints.setColorScheme(Qt.ColorScheme.Light if theme == "light" else Qt.ColorScheme.Dark)

    apply_color_scheme()
    backend.settingsChanged.connect(apply_color_scheme)
    app.styleHints().colorSchemeChanged.connect(lambda *_: backend.on_system_scheme_changed())

    engine = QQmlApplicationEngine()
    engine.addImageProvider("icon", IconProvider())
    engine.addImageProvider("snapshot", Backend.snapshots)
    engine.rootContext().setContextProperty("backend", backend)
    engine.load(QUrl.fromLocalFile(str(resource_root() / "ui" / "main.qml")))
    if not engine.rootObjects():
        return 1
    main_window.append(engine.rootObjects()[0])

    app.aboutToQuit.connect(backend.shutdown)
    # Start background initialisation once the event loop (and splash) is running.
    QTimer.singleShot(0, backend.start)

    # Development aid: SMOLCHAT_SCREENSHOT=out.png [SMOLCHAT_SCREENSHOT_DELAY=ms]
    # saves a capture of the main window and quits.
    shot = os.environ.get("SMOLCHAT_SCREENSHOT")
    if shot:
        window = engine.rootObjects()[0]

        panel = os.environ.get("SMOLCHAT_SCREENSHOT_PANEL")  # "library" | "settings"
        if panel:
            QTimer.singleShot(
                1500, lambda: QMetaObject.invokeMethod(window, f"open{panel.capitalize()}Panel")
            )

        prompt = os.environ.get("SMOLCHAT_SCREENSHOT_PROMPT")
        if prompt:
            QTimer.singleShot(1500, lambda: backend.sendMessage(prompt, []))

        def capture() -> None:
            window.grabWindow().save(shot)
            app.quit()

        QTimer.singleShot(int(os.environ.get("SMOLCHAT_SCREENSHOT_DELAY", "2500")), capture)

    # Let Ctrl+C in a terminal quit the app.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    tick = QTimer()
    tick.start(250)
    tick.timeout.connect(lambda: None)

    code = app.exec()
    del engine  # tear down QML before the backend it binds to
    activation_server.close()
    lock.release()
    return code


if __name__ == "__main__":
    sys.exit(main())
