"""Logger module.

This module instantiates a global logger singleton.
"""
from malib.logger.histogram import Histogram
from malib.logger.logger import Logger, LogOutput
from malib.logger.simple_outputs import StdOutput, TextOutput
from malib.logger.tabular_input import TabularInput
from malib.logger.csv_output import CsvOutput  # noqa: I100
from malib.logger.snapshotter import Snapshotter
from malib.logger.tensor_board_output import TensorBoardOutput

logger = Logger()
tabular = TabularInput()
snapshotter = Snapshotter()

__all__ = [
    'Histogram', 'Logger', 'CsvOutput', 'StdOutput', 'TextOutput', 'LogOutput',
    'Snapshotter', 'TabularInput', 'TensorBoardOutput', 'logger', 'tabular',
    'snapshotter'
]
