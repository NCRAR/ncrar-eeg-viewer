import argparse
import configparser
import logging
from pathlib import Path

from enaml.application import deferred_call
from enaml.qt.QtCore import QStandardPaths


def config_file():
    config_path = Path(QStandardPaths.standardLocations(QStandardPaths.AppConfigLocation)[0])
    config_file =  config_path / 'ncrar-eeg-viewer' / 'config.ini'
    config_file.parent.mkdir(exist_ok=True, parents=True)
    return config_file


def get_config():
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'current_path': ''}
    config.read(config_file())
    return config


def write_config(config):
    with config_file().open('w') as fh:
        config.write(fh)


def main():
    import enaml
    from enaml.qt.qt_application import QtApplication
    logging.basicConfig(level='INFO')

    from ncrar_eeg_viewer.presenter import Presenter
    with enaml.imports():
        from ncrar_eeg_viewer.gui import Main

    parser = argparse.ArgumentParser("ncrear-eeg-viewer")
    parser.add_argument("path", nargs='?')
    args = parser.parse_args()

    app = QtApplication()
    config = get_config()

    presenter = Presenter()
    view = Main(presenter=presenter)

    view.show()
    app.start()
    app.stop()


if __name__ == "__main__":
    main()
