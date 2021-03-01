from typing import List, Set, Dict, Tuple, Optional, Union, TextIO, Any, Hashable
from csverve import helpers
import pandas as pd # type: ignore
import collections
import logging
import shutil
import yaml
import gzip
import os


"""
    Type hints: https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html


    YAML format example:

    columns:
    - dtype: int
      name: prediction_id
    - dtype: str
      name: chromosome_1
    - dtype: str
      name: strand_1
    header: true
    sep: "\t"
"""


class CsverveMergeDtypesEmptyMergeSet(Exception):
    pass


class CsverveConcatNaNIntDtypeException(Exception):
    pass


class CsverveMergeCommonColException(Exception):
    pass


class DtypesMergeException(Exception):
    pass


class CsverveConcatException(Exception):
    pass


class CsverveAnnotateError(Exception):
    pass


class CsverveMergeException(Exception):
    pass


class CsverveMergeColumnMismatchException(Exception):
    pass


class CsverveParseError(Exception):
    pass


class CsverveInputError(Exception):
    pass


class CsverveWriterError(Exception):
    pass


class CsverveTypeMismatch(Exception):
    def __init__(self, column, expected_dtype, dtype):
        self.column = column
        self.dtype = dtype
        self.expected_dtype = expected_dtype

    def __str__(self):
        message = 'mismatching types for col {}. types were {} and {}'
        message = message.format(self.column, self.expected_dtype, self.dtype)
        return message


def pandas_to_std_types() -> Dict[str, str]:
    std_dict = {
        "bool": "bool",
        "int64": "int",
        "int": "int",
        "Int64": "int",
        "float64": "float",
        "float": "float",
        "object": "str",
        "str": "str",
        "category": "str",
        "NaN": "NaN",
    }

    return collections.defaultdict(lambda: "str", std_dict)


class IrregularCsverveInput(object):
    def __init__(self, filepath: str, dtypes: Dict[str, str], na_rep: str = 'NaN') -> None:
        """
        CSV file and all related metadata.

        @param filepath: Path of CSV.
        @param dtypes: dictionary of pandas dtypes by column, keys = column name, value = dtype.
        @param na_rep: replace NaN with this value.
        """
        self.header: bool
        self.sep: str
        self.dtypes: Dict[str, str]
        self.columns: List[str]

        self.filepath: str = filepath
        self.na_rep: str = na_rep
        metadata: Tuple[bool, str, Dict[str, str], List[str]] = self.__generate_metadata()
        self.header, self.sep, self.dtypes, self.columns = metadata
        self.dtypes = dtypes

    @property
    def yaml_file(self) -> str:
        """
        Append '.yaml' to CSV path.

        @return: YAML metadata path.
        """
        return self.filepath + '.yaml'

    def __get_compression_type_pandas(self) -> Union[None, str]:
        """
        Get CSV compression type.

        @return .csv -> None
                .gz -> gzip
                .h5 -> error, not supported
                .hdf5 -> error, not supported
                other exts -> None:
        """
        filepath: str = self.filepath
        if filepath.endswith('.tmp'):
            filepath = filepath[:-4]

        ext: str
        _, ext = os.path.splitext(filepath)

        if ext == ".csv":
            return None
        elif ext == ".gz":
            return "gzip"
        elif ext == ".h5" or ext == ".hdf5":
            raise CsverveInputError("HDF is not supported")
        else:
            logging.getLogger("mondrian.utils.csv").warn(
                "Couldn't detect output format. extension {}".format(ext)
            )
            return None

    def get_dtypes_from_df(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Get dtypes of pandas DataFrame.

        @param df: pandas DataFrame.
        @return: dictionary of pandas dtypes by column, keys = column name, value = dtype.
        """
        type_converter: Dict[str, str] = pandas_to_std_types()

        typeinfo = {}
        for column, dtype in df.dtypes.items():
            if column in ['chr', 'chrom', 'chromosome']:
                typeinfo[column] = 'str'
                typeinfo[column] = 'str'
            else:
                if df.empty:
                    typeinfo[column] = self.na_rep
                else:
                    typeinfo[column] = type_converter[str(dtype)]

        return typeinfo

    def __detect_sep_from_header(self, header: str) -> str:
        """
        Detect whether file is tab or comma separated from header.

        @param header: header line.
        @return: '\t', or ',', or raise error if unable to detect separator.
        """
        if '\t' in header and ',' in header:
            raise CsverveParseError("Unable to detect separator from {}".format(header))

        if '\t' not in header and ',' not in header:
            raise CsverveParseError("Unable to detect separator from {}".format(header))

        if '\t' in header:
            return '\t'
        elif ',' in header:
            return ','
        else:
            raise Exception()

    def __generate_metadata(self) -> Tuple[bool, str, Dict[str, str], List[str]]:
        """
        Get metadata of CSV file.

        @return: header, separator, pandas dtypes, columns
        """
        with helpers.getFileHandle(self.filepath, 'rt') as inputfile:
            header_line: str = inputfile.readline().strip()
            sep: str = self.__detect_sep_from_header(header_line)
            columns: List[str] = header_line.split(sep)
            header: bool = True
            dtypes: Dict[str, str] = self.__generate_dtypes(sep=sep)
            return header, sep, dtypes, columns

    def __generate_dtypes(self, columns: List[str] = None, sep: str = ',') -> Dict[str, str]:
        """
        Generate dtypes.

        @param columns List of columns names.
        @param sep: Separator.
        @return: Dictionary of pandas dtypes by column, keys = column name, value = dtype.
        """
        compression: Union[None, str] = self.__get_compression_type_pandas()

        data: pd.DataFrame = pd.read_csv(
            self.filepath, compression=compression, chunksize=10 ** 6,
            sep=sep
        )
        data = next(data)

        if columns:
            data.columns = columns

        typeinfo: Dict[str, str] = self.get_dtypes_from_df(data)
        return typeinfo

    def read_csv(self, chunksize: int = None) -> pd.DataFrame:
        """
        Read CSV.

        @param chunksize: Number of rows to read at a time (optional, applies to large datasets).
        @return: pandas DataFrame.
        """
        def return_gen(df_iterator):
            for df in df_iterator:
                for col in df.columns.values:
                    assert col in self.dtypes, col
                yield df

        dtypes: Dict[str, str] = {k: v for k, v in self.dtypes.items() if v != "NA"}
        # if header exists then use first line (0) as header
        header = 0 if self.header else None
        names = None if self.header else self.columns

        compression: Union[None, str] = self.__get_compression_type_pandas()

        try:
            data: pd.DataFrame = pd.read_csv(
                self.filepath, compression=compression, chunksize=chunksize,
                sep=self.sep, header=header, names=names, dtype=dtypes)
        except pd.errors.EmptyDataError:
            data = pd.DataFrame(columns=self.columns)

        if chunksize:
            return return_gen(data)
        else:
            for col in data.columns.values:
                assert col in self.dtypes, col
            return data


class CsverveInput(object):
    def __init__(self, filepath: str, na_rep: str = 'NaN') -> None:
        """
        CSV file and all related metadata.

        @param filepath: Path of CSV.
        @param na_rep: Replace NaN with this value.
        """
        self.header: bool
        self.dtypes: Dict[str, str]
        self.columns: List[str]
        self.sep: str

        self.filepath: str = filepath
        self.na_rep: str = na_rep
        metadata: Tuple[bool, Dict[str, str], List[str], str] = self.__parse_metadata()
        self.header, self.dtypes, self.columns, self.sep = metadata
        self.__confirm_compression_type_pandas()

    def cast_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast dataframe dtypes.

        @param df: Pandas DataFrame.
        @return: Pandas DataFrame.
        """
        for column_name in df.columns.values:
            dtype = self.dtypes[column_name]
            df[column_name] = df[column_name].astype(dtype)
        return df

    @property
    def yaml_file(self) -> str:
        """
        Append '.yaml' to CSV path.

        @return: YAML metadata path.
        """
        return self.filepath + '.yaml'

    def __confirm_compression_type_pandas(self) -> None:
        """
        Confirm compression type of CSV file is .gz, is not, raise error.

        @return:
        """

        filepath: str = self.filepath
        if filepath.endswith('.tmp'):
            filepath = filepath[:-4]

        ext: str
        _, ext = os.path.splitext(filepath)

        if not ext == ".gz":
            raise CsverveInputError("{} is not supported".format(ext))

    def __parse_metadata(self) -> Tuple[bool, Dict[str, str], List[str], str]:
        """
        Parse metadata.

        @return: header, pandas dtypes, columns, separator
        """

        with open(self.filepath + '.yaml') as yamlfile:
            yamldata = yaml.safe_load(yamlfile)

        header: bool = yamldata['header']
        sep: str = yamldata['sep']

        dtypes: Dict[str, str] = {}
        columns: List[str] = []
        for coldata in yamldata['columns']:
            colname: str = coldata['name']

            dtypes[colname] = coldata['dtype']

            columns.append(colname)

        return header, dtypes, columns, sep

    def __verify_data(self, df: pd.DataFrame) -> None:
        """
        Verify columns of DataFrame match those of class property.

        @param df: Pandas DataFrame.
        @return:
        """
        if not set(list(df.columns.values)) == set(self.columns):
            raise CsverveParseError("metadata mismatch in {}".format(self.filepath))

    def read_csv(self, chunksize: int = None) -> pd.DataFrame:
        """
        Read CSV.

        @param chunksize: Number of rows to read at a time (optional, applies to large datasets).
        @return: pandas DataFrame.
        """

        def return_gen(df_iterator):
            for df in df_iterator:
                self.__verify_data(df)
                yield df

        dtypes: Dict[str, str] = {k: v for k, v in self.dtypes.items() if v != "NA"}
        # if header exists then use first line (0) as header
        header: Union[int, None] = 0 if self.header else None
        names: Union[None, List[str]] = None if self.header else self.columns

        try:
            data: pd.DataFrame = pd.read_csv(
                self.filepath, compression='gzip', chunksize=chunksize,
                sep=self.sep, header=header, names=names, dtype=dtypes)
        except pd.errors.EmptyDataError:
            data = pd.DataFrame(columns=self.columns)
            data = self.cast_dataframe(data)

        if chunksize:
            return return_gen(data)
        else:
            self.__verify_data(data)
            return data


class CsverveOutput(object):
    def __init__(
        self,
        filepath: str,
        dtypes: Dict[str, str],
        header: bool = True,
        na_rep: str = 'NaN',
        columns: List[str] = None,
    ) -> None:
        """
        CSV file and all related metadata.

        @param filepath: CSV file path.
        @param dtypes: Dictionary of pandas dtypes, where key = column name, value = dtype.
        @param header: boolean, True = write header, False = don't write header.
        @param na_rep: replace NaN with this value.
        @param columns: List of column names.
        """

        self.filepath: str = filepath
        self.header: bool = header
        self.dtypes: Dict[str, str] = dtypes
        self.na_rep: str = na_rep
        self.columns: Optional[List[str]] = columns
        self.__confirm_compression_type_pandas()
        self.sep: str = ','

    @property
    def yaml_file(self) -> str:
        """
        Append '.yaml' to CSV path.

        @return: YAML metadata path.
        """
        return self.filepath + '.yaml'

    @property
    def header_line(self) -> str:
        """
        Return header line as string using designated separator.

        @return:
        """
        assert self.columns
        return self.sep.join(self.columns) + '\n'

    def __confirm_compression_type_pandas(self) -> None:
        """
        Confirm compression type of CSV file is .gz, is not, raise error.

        @return:
        """

        filepath: str = self.filepath
        if filepath.endswith('.tmp'):
            filepath = filepath[:-4]

        ext: str
        _, ext = os.path.splitext(filepath)

        if not ext == ".gz":
            raise CsverveWriterError("{} is not supported".format(ext))

    def write_yaml(self) -> None:
        """
        Write .yaml file.

        @return:
        """

        type_converter: Dict[str, str] = pandas_to_std_types()

        yamldata: Dict[str, Any] = {'header': self.header, 'sep': self.sep, 'columns': []}

        assert self.columns is not None
        for column in self.columns:
            data = {'name': column, 'dtype': type_converter[self.dtypes[column]]}
            yamldata['columns'].append(data)

        with open(self.yaml_file, 'wt') as f:
            yaml.safe_dump(yamldata, f, default_flow_style=False)

    def __get_dtypes_from_df(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Get pandas dtypes from pandas DataFrame.

        @param df: Pandas DataFrame.
        @return: Dictionary of pandas dtypes, where key = column name, value = dtype.
        """

        dtypes: Dict[str, str] = df.dtypes
        dtypes_converter: Dict[str, str] = pandas_to_std_types()
        dtypes = {k: dtypes_converter[str(v)] for k, v in dtypes.items()}
        return dtypes

    def __verify_df(self, df: pd.DataFrame) -> None:
        """
        Verify columns of DataFrame match those of class property.

        @param df: Pandas DataFrme.
        @return:
        """

        if self.columns:
            assert set(list(df.columns.values)) == set(self.columns)
        else:
            self.columns = df.columns.values

    def __cast_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast dataframe dtypes.

        @param df: Pandas DataFrame.
        @return: Pandas DataFrame.
        """

        for column_name in df.columns.values:
            dtype: str = self.dtypes[column_name]

            if str(dtype) == 'bool' and df[column_name].isnull().any():
                raise Exception('NaN found in bool column:{}'.format(column_name))

            df[column_name] = df[column_name].astype(dtype)

        return df

    def __write_df(self, df: pd.DataFrame, header: bool = True, mode: str = 'w') -> None:
        """
        Write pandas DataFrame to gzip CSV.

        @param df: Pandas DataFrame.
        @param header: bool value to write header or not.
        @param mode: Pandas write CSV move, 'w' by default.
        @return:
        """

        df = self.__cast_df(df)
        if self.columns:
            assert self.columns == list(df.columns.values)
        else:
            self.columns = list(df.columns.values)

        df.to_csv(
            self.filepath, sep=self.sep, na_rep=self.na_rep,
            index=False, compression='gzip', mode=mode, header=header
        )

    def __write_df_chunks(self, dfs: List[pd.DataFrame], header: bool = True) -> None:
        """
        Write DataFrames in chunks.

        @param dfs: List of pandas DataFrames.
        @param header: bool, write header or not.
        @return:
        """

        for i, df in enumerate(dfs):
            if i == 0 and self.header:
                self.__write_df(df, header=header, mode='w')
            else:
                self.__write_df(df, header=False, mode='a')

    def write_df(self, df: pd.DataFrame, chunks: bool = False) -> None:
        """
        Write out dataframe to CSV.

        @param df: Pandas DataFrames.
        @param chunks: bool.
        @return:
        """

        if chunks:
            self.__write_df_chunks(df, header=self.header)
        else:
            self.__write_df(df, self.header)

        self.write_yaml()

    def write_header(self, writer: TextIO) -> None:
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

            if self.header:
                self.write_header(writer)

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
            if self.header:
                self.write_header(writer)

            with gzip.open(csvfile, 'rt') as data_stream:
                shutil.copyfileobj(
                    data_stream, writer, length=16 * 1024 * 1024
                )

        self.write_yaml()

    def write_text(self, text: List[str]) -> None:
        """
        Write text.

        @param text: List of lines.
        @return:
        """

        assert self.columns
        assert self.dtypes

        with gzip.open(self.filepath, 'wt') as writer:

            if self.header:
                self.write_header(writer)

            for line in text:
                writer.write(line)

        self.write_yaml()


def write_metadata(infile: str, dtypes: Dict[str, str]) -> None:
    """
    Create meta YAML for a gzipped CSV file. Must include dtypes for all columns.

    @param infile: Path to gzipped CSV file.
    @param dtypes: Dictionary of pandas dtypes, where key = column name, value = dtype.
    @return:
    """

    csvinput: IrregularCsverveInput = IrregularCsverveInput(infile, dtypes)

    csvoutput: CsverveOutput = CsverveOutput(
        infile, csvinput.dtypes, header=csvinput.header,
        columns=csvinput.columns
    )
    csvoutput.write_yaml()


def merge_dtypes(dtypes_all: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Merge pandas dtypes.

    @param dtypes_all: List of dtypes dictionaries, where key = column name, value = pandas dtype.
    @return: Merged dtypes dictionary.
    """

    if dtypes_all == []:
        raise CsverveMergeDtypesEmptyMergeSet("must provide dtypes to merge")

    merged_dtypes: Dict[str, str] = {}

    for dtypes in dtypes_all:
        for k, v in dtypes.items():
            if k in merged_dtypes:
                if merged_dtypes[k] != v:
                    raise DtypesMergeException("dtypes not mergeable")
            else:
                merged_dtypes[k] = v

    return merged_dtypes


def concatenate_csv(inputfiles: List[str], output: str, write_header: bool = True) -> None:
    """
    Concatenate gzipped CSV files, dtypes in meta YAML files must be the same.

    @param inputfiles: List of gzipped CSV file paths, or a dictionary where the keys are file paths.
    @param output: Path of resulting concatenated gzipped CSV file and meta YAML.
    @param write_header: boolean, True = write header, False = don't write header.
    @return:
    """

    if inputfiles == [] or inputfiles == {}:
        raise CsverveConcatException("nothing provided to concat")

    if isinstance(inputfiles, dict):
        inputfiles = inputfiles.values()

    inputs: List[CsverveInput] = [CsverveInput(infile) for infile in inputfiles]

    dtypes: Dict[str, str] = merge_dtypes([csvinput.dtypes for csvinput in inputs])

    headers: List[bool] = [csvinput.header for csvinput in inputs]

    columns: List[List[str]] = [csvinput.columns for csvinput in inputs]

    low_memory: bool = True
    if any(headers):
        low_memory = False

    if not all(columns[0] == elem for elem in columns):
        low_memory = False

    if low_memory:
        columns_ = columns[0]
        concatenate_csv_files_quick_lowmem(inputfiles, output, dtypes, columns_, write_header=write_header)

    else:
        concatenate_csv_files_pandas(inputfiles, output, dtypes, write_header=write_header)


def concatenate_csv_files_pandas(
    in_filenames: Union[List[str], Dict[str, str]],
    out_filename: str,
    dtypes: Dict[str, str],
    write_header: bool = True
) -> None:
    """
    Concatenate gzipped CSV files.

    @param in_filenames: List of gzipped CSV file paths, or a dictionary where the keys are file paths.
    @param out_filename: Path of resulting concatenated gzipped CSV file and meta YAML.
    @param dtypes: Dictionary of pandas dtypes, where key = column name, value = dtype.
    @param write_header: boolean, True = write header, False = don't write header.
    @return:
    """

    if isinstance(in_filenames, dict):
        in_filenames = list(in_filenames.values())

    data: List[CsverveInput] = [
        CsverveInput(in_filename).read_csv() for in_filename in in_filenames
    ]
    data_: pd.DataFrame = pd.concat(data, ignore_index=True)
    csvoutput: CsverveOutput = CsverveOutput(out_filename, dtypes, header=write_header)
    csvoutput.write_df(data_)


def concatenate_csv_files_quick_lowmem(
    inputfiles: List[str],
    output: str,
    dtypes: Dict[str, str],
    columns: List[str],
    write_header: bool = True
) -> None:
    """
    Concatenate gzipped CSV files.

    @param inputfiles: List of gzipped CSV file paths.
    @param output: Path of resulting concatenated gzipped CSV file and meta YAML.
    @param dtypes: Dictionary of pandas dtypes, where key = column name, value = dtype.
    @param columns: List of column names for newly concatenated gzipped CSV file.
    @param write_header: boolean, True = write header, False = don't write header.
    @return:
    """

    csvoutput: CsverveOutput = CsverveOutput(
        output, dtypes, header=write_header, columns=columns
    )
    csvoutput.write_data_streams(inputfiles)


# annotation_dtypes shouldnt be default, if it is None, it breaks
def annotate_csv(
    infile: str,
    annotation_data,
    outfile,
    annotation_dtypes,
    on="cell_id",
    write_header: bool = True,
):
    """
    TODO: fill this in

    @param infile:
    @param annotation_data:
    @param outfile:
    @param annotation_dtypes:
    @param on:
    @param write_header:
    @return:
    """

    csvinput = CsverveInput(infile)
    metrics_df = csvinput.read_csv()

    ann = pd.DataFrame(annotation_data).T

    ann[on] = ann.index

    # get annotation rows that correspond to rows in on
    reformed_annotation = ann[ann[on].isin(metrics_df[on])]

    # do nothing if the annotation df is empty
    if reformed_annotation.empty:  # so we dont add NaNs
        return write_dataframe_to_csv_and_yaml(metrics_df, outfile,
                                               csvinput.dtypes,
                                               write_header=write_header)

    metrics_df = metrics_df.merge(reformed_annotation, on=on, how='outer')

    csv_dtypes = csvinput.dtypes

    for col, dtype in csv_dtypes.items():
        if col in annotation_dtypes:
            assert dtype == annotation_dtypes[col]

    csv_dtypes.update(annotation_dtypes)

    output = CsverveOutput(outfile, csv_dtypes, header=write_header)
    output.write_df(metrics_df)


def simple_annotate_csv(
    in_f: str,
    out_f: str,
    col_name: str,
    col_val: str,
    col_dtype: str,
    write_header: bool = False,
) -> None:
    """
    Simplified version of the annotate_csv method.
    Add column with the same value for all rows.

    @param in_f:
    @param out_f:
    @param col_name:
    @param col_val:
    @param col_dtype:
    @param write_header:
    @return:
    """
    csvinput = CsverveInput(in_f)
    metrics_df = csvinput.read_csv()
    metrics_df[col_name] = col_val

    csv_dtypes = csvinput.dtypes
    csv_dtypes[col_name] = col_dtype

    output = CsverveOutput(out_f, csv_dtypes, header=write_header)
    output.write_df(metrics_df)


def add_col_from_dict(
    infile,
    col_data,
    outfile,
    dtypes,
    write_header=True
):
    """
    TODO: fill this in
    Add column to gzipped CSV.

    @param infile:
    @param col_data:
    @param outfile:
    @param dtypes:
    @param write_header:
    @return:
    """

    csvinput = CsverveInput(infile)
    csv_dtypes = csvinput.dtypes
    csvinput = csvinput.read_csv()

    for col_name, col_value in col_data.items():
        csvinput[col_name] = col_value

    for col, dtype in csv_dtypes.items():
        if col in dtypes:
            assert dtype == dtypes[col]

    output = CsverveOutput(outfile, dtypes, header=write_header)
    output.write_df(csvinput)


def rewrite_csv_file(
    filepath: str,
    outputfile: str,
    write_header: bool = True,
    dtypes: Dict[str, str] = None,
) -> None:
    """
    Generate header less csv files.

    @param filepath: File path of CSV.
    @param outputfile: File path of header less CSV to be generated.
    @param write_header: boolean, True = write header, False = don't write header.
    @param dtypes: Dictionary of pandas dtypes, where key = column name, value = dtype.
    @return:
    """

    if os.path.exists(filepath + '.yaml'):
        csvinput: Union[CsverveInput, IrregularCsverveInput] = CsverveInput(filepath)
    else:
        assert dtypes
        csvinput = IrregularCsverveInput(filepath, dtypes)

    if csvinput.header:
        df = csvinput.read_csv()

        csvoutput = CsverveOutput(
            outputfile, header=write_header, columns=csvinput.columns,
            dtypes=csvinput.dtypes
        )
        csvoutput.write_df(df)

    else:
        csvoutput = CsverveOutput(
            outputfile, header=write_header, columns=csvinput.columns,
            dtypes=csvinput.dtypes
        )
        csvoutput.rewrite_csv(filepath)


def merge_csv(
    in_filenames: Union[List[str], Dict[str, str]],
    out_filename: str,
    how: str,
    on: str,
    write_header: bool = True
) -> None:
    """
    Create one gzipped CSV out of multiple gzipped CSVs.

    @param in_filenames: Dictionary containing file paths as keys
    @param out_filename: Path to newly merged CSV
    @param how: How to join DataFrames (inner, outer, left, right).
    @param on: Column(s) to join on, comma separated if multiple.
    @param write_header: boolean, True = write header, False = don't write header
    @return:
    """
    if isinstance(in_filenames, dict):
        in_filenames = list(in_filenames.values())

    data: List[CsverveInput] = [CsverveInput(infile) for infile in in_filenames]

    dfs: List[str] = [csvinput.read_csv() for csvinput in data]

    dtypes: List[Dict[str, str]] = [csvinput.dtypes for csvinput in data]

    data_: pd.DataFrame = merge_frames(dfs, how, on)

    dtypes_: Dict[str, str] = merge_dtypes(dtypes)

    columns: List[str] = list(data_.columns.values)

    csvoutput: CsverveOutput = CsverveOutput(out_filename, dtypes_, header=write_header, columns=columns)
    csvoutput.write_df(data_)


def _validate_merge_cols(frames: List[pd.DataFrame], on: Union[List[str], str]) -> None:
    """
    Make sure frames look good, raise relevant exceptions.

    @param frames: list of pandas DataFrames to merge
    @param on: list of common columns in frames on which to merge
    @return:
    """

    if on == []:
        raise CsverveMergeException("unable to merge if given nothing to merge on")

    # check that columns to be merged have identical values
    standard = frames[0][on]
    for frame in frames:
        if not standard.equals(frame[on]):
            raise CsverveMergeColumnMismatchException("columns on which to merge must be identical")

    # check that columns to be merged have same dtypes
    for shared_col in on:
        if len(set([frame[shared_col].dtypes for frame in frames])) != 1:
            raise CsverveMergeColumnMismatchException("columns on which to merge must have same dtypes")

    common_cols = set.intersection(*[set(frame.columns) for frame in frames])
    cols_to_check = list(common_cols - set(on))

    for frame1, frame2 in zip(frames[:-1], frames[1:]):
        if not frame1[cols_to_check].equals(frame2[cols_to_check]):
            raise CsverveMergeCommonColException("non-merged common cols must be identical")


def merge_frames(frames: List[pd.DataFrame], how: str, on: str) -> pd.DataFrame:
    """
    Takes in a list of pandas DataFrames, and merges into a single DataFrame.
    #TODO: add handling if empty list is given

    @param frames: List of pandas DataFrames.
    @param how: How to join DataFrames (inner, outer, left, right).
    @param on: Column(s) to join on, comma separated if multiple.
    @return: merged pandas DataFrame.
    """

    if ',' in on:
        on_split = on.split(',')
    else:
        on_split = [on]

    _validate_merge_cols(frames, on)

    if len(frames) == 1:
        return frames[0]

    else:
        left: pd.DataFrame = frames[0]
        right: pd.DataFrame = frames[1]
        cols_to_use: List[str] = list(right.columns.difference(left.columns))
        cols_to_use += on
        cols_to_use = list(set(cols_to_use))

        merged_frame: pd.DataFrame = pd.merge(
            left, right[cols_to_use], how=how, on=on,
        )
        for i, frame in enumerate(frames[2:]):
            merged_frame = pd.merge(
                merged_frame, frame, how=how, on=on,
            )
        return merged_frame


def write_dataframe_to_csv_and_yaml(
    df: pd.DataFrame,
    outfile: str,
    dtypes: Dict[str, str],
    write_header: bool = True
) -> None:
    """
    Output pandas dataframe to a CSV and meta YAML files.

    @param df: pandas DataFrame.
    @param outfile: Path of CSV & YAML file to be written to.
    @param dtypes: dictionary of pandas dtypes by column, keys = column name, value = dtype.
    @param write_header: boolean, True = write header, False = don't write header
    @return:
    """
    csvoutput: CsverveOutput = CsverveOutput(outfile, dtypes, header=write_header)
    csvoutput.write_df(df)


def read_csv_and_yaml(infile: str, chunksize: int = None) -> pd.DataFrame:
    """
    Read in CSV file and return as a pandas DataFrame.

    Assumes a YAML meta file in the same path with the same name, with a .yaml extension.
    YAML file structure is atop this file.

    @param infile: Path to CSV file.
    @param chunksize: Number of rows to read at a time (optional, applies to large datasets).
    @return: pandas DataFrame.
    """
    return CsverveInput(infile).read_csv(chunksize=chunksize)


def get_metadata(input: str) -> Tuple[bool, Dict[str, str], List[str]]:
    """
    Get CSV file's header, dtypes and columns.

    Assumes a YAML meta file in the same path with the same name, with a .yaml extension.
    YAML file structure is atop this file.

    Example: CSV = /path/to/file.csv
            YAML = /path/to/file.csv.yaml


    @param input: Path to CSV file.
    @return: header (bool), dtypes (dict), columns (list).
    """
    csvinput: CsverveInput = CsverveInput(input)
    return csvinput.header, csvinput.dtypes, csvinput.columns


