import argparse
import configparser
import logging
from pathlib import Path

from enaml.application import deferred_call
from enaml.qt.QtCore import QStandardPaths
import enamlx


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

    enamlx.install()

    logging.basicConfig(level='INFO')

    from ncrar_eeg_viewer.presenter import Presenter
    with enaml.imports():
        from ncrar_eeg_viewer.gui import Main

    parser = argparse.ArgumentParser("ncrear-eeg-viewer")
    parser.add_argument("path", nargs='?')
    parser.add_argument('--settings')
    parser.add_argument('--analysis')
    args = parser.parse_args()

    app = QtApplication()
    config = get_config()

    presenter = Presenter()
    view = Main(presenter=presenter)

    if args.settings is not None:
        presenter.load_config(args.settings)
    if args.analysis is not None:
        presenter.load_analysis(args.analysis)
    if args.path is not None:
        presenter.load_file(args.path)

    view.show()
    app.start()
    app.stop()


if __name__ == "__main__":
    main()
