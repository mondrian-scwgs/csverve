import gzip
from typing import List, Dict, Union, Any

import pandas as pd  # type: ignore
from csverve.errors import CsverveInputError
from csverve.errors import CsverveParseError


class IrregularCsverveInput(object):
    def __init__(self, filepath: str, dtypes: Dict[str, str]) -> None:
        """
        CSV file and all related metadata.

        @param filepath: Path of CSV.
        @param dtypes: dictionary of pandas dtypes by column, keys = column name, value = dtype.
        """
        self.filepath: str = filepath

        self.sep = self.get_seperator()
        self.columns = self.get_columns()

        self.dtypes: Dict[str, str] = dtypes

    @property
    def __file_type(self) -> str:
        if self.filepath.endswith('gz'):
            return 'gzip'
        elif self.filepath.endswith('csv'):
            return 'plain-text'
        else:
            raise CsverveInputError('Unsupported file type: {}'.format(self.filepath))

    @property
    def yaml_file(self) -> str:
        """
        Append '.yaml' to CSV path.

        @return: YAML metadata path.
        """
        return self.filepath + '.yaml'

    def get_seperator(self) -> str:
        """
        Detect whether file is tab or comma separated from header.
        @return: '\t', or ',', or raise error if unable to detect separator.
        """
        opener: Any = gzip.open if self.__file_type == 'gzip' else open
        with opener(self.filepath, 'rt') as inputfile:
            header: str = inputfile.readline().strip()

        tab = '\t' in header
        comma = ',' in header

        if (tab and comma) or (not tab and not comma):
            raise CsverveParseError("Unable to detect separator from {}".format(header))

        return '\t' if tab else ','

    def get_columns(self) -> List[str]:
        """
        Detect whether file is tab or comma separated from header.
        @return: '\t', or ',', or raise error if unable to detect separator.
        """
        opener: Any = gzip.open if self.__file_type == 'gzip' else open
        with opener(self.filepath, 'rt') as inputfile:
            header: str = inputfile.readline().strip()

        columns: List[str] = header.split(self.sep)

        return columns

    def read_csv(self, chunksize: int = None) -> pd.DataFrame:
        """
        Read CSV.

        @param chunksize: Number of rows to read at a time (optional, applies to large datasets).
        @return: pandas DataFrame.
        """
        try:
            data: pd.DataFrame = pd.read_csv(
                self.filepath, chunksize=chunksize,
                sep=self.sep, names=self.columns, dtype=self.dtypes
            )
        except pd.errors.EmptyDataError:
            data = pd.DataFrame(columns=self.columns)

        if chunksize:
            for df in data:
                for col in data.columns.values:
                    assert col in self.dtypes, col
                yield df
        else:
            for col in data.columns.values:
                assert col in self.dtypes, col
            return data
