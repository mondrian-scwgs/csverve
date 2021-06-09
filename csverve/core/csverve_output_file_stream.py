import gzip
import shutil
from typing import List, Dict, TextIO

from csverve.core import CsverveOutput


class CsverveOutputFileStream(CsverveOutput):
    def __init__(
        self,
        filepath: str,
        dtypes: Dict[str, str],
        columns: List[str],
        write_header: bool = True,
        na_rep: str = 'NaN',
        sep: str = ',',
    ) -> None:
        """
        CSV file and all related metadata.

        @param filepath: CSV file path.
        @param dtypes: Dictionary of pandas dtypes, where key = column name, value = dtype.
        @param header: boolean, True = write header, False = don't write header.
        @param na_rep: replace NaN with this value.
        @param columns: List of column names.
        """

        super().__init__(
            filepath, dtypes, columns,
            write_header=write_header, na_rep=na_rep, sep=sep
        )

    def _write_header_to_file(self, writer: TextIO) -> None:
        """
        Write header.
        @param writer: TextIO.
        @return:
        """
        assert self.columns
        header: str = ','.join(self.columns)
        header = header + '\n'
        writer.write(header)

    def write_data_streams(self, csvfiles: List[str]) -> None:
        """
        Write data streams.
        @param csvfiles: List of CSV files paths.
        @return:
        """
        assert self.columns
        assert self.dtypes
        with gzip.open(self.filepath, 'wt') as writer:

            if self.write_header:
                self._write_header_to_file(writer)

            for csvfile in csvfiles:
                with gzip.open(csvfile, 'rt') as data_stream:
                    shutil.copyfileobj(
                        data_stream, writer, length=16 * 1024 * 1024
                    )

        self.write_yaml()

    def rewrite_csv(self, csvfile: str) -> None:
        """
        Rewrite CSV.
        @param csvfile: Filepath of CSV file.
        @return:
        """

        assert self.columns
        assert self.dtypes
        with gzip.open(self.filepath, 'wt') as writer:
            if self.write_header:
                self._write_header_to_file(writer)

            with gzip.open(csvfile, 'rt') as data_stream:
                shutil.copyfileobj(
                    data_stream, writer, length=16 * 1024 * 1024
                )

        self.write_yaml()
