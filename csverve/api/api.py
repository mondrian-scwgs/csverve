import os
from typing import List, Dict, Union

import csverve.utils as utils
import pandas as pd  # type: ignore
from csverve.core import CsverveInput
from csverve.core import CsverveOutputDataFrame
from csverve.core import CsverveOutputFileStream
from csverve.core import IrregularCsverveInput
from csverve.errors import CsverveConcatException
from csverve.utils import concatenate_csv_files_quick_lowmem, concatenate_csv_files_pandas


def get_columns(infile):
    return CsverveInput(infile).columns


def get_dtypes(infile):
    return CsverveInput(infile).dtypes


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
        df = csvinput.read_csv()

        csvoutput_df = CsverveOutputDataFrame(
            df, outputfile, write_header=write_header,
            dtypes=csvinput.dtypes
        )
        csvoutput_df.write_df()
    else:
        assert dtypes
        csvinput = IrregularCsverveInput(filepath, dtypes)

        csvoutput_fs = CsverveOutputFileStream(
            outputfile, write_header=write_header, columns=csvinput.columns,
            dtypes=csvinput.dtypes
        )
        csvoutput_fs.rewrite_csv(filepath)


def merge_csv(
        in_filenames: Union[List[str], Dict[str, str]],
        out_filename: str,
        how: str,
        on: List[str],
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

    merged_data: pd.DataFrame = utils.merge_frames(dfs, how, on)

    dtypes_: Dict[str, str] = utils.merge_dtypes(dtypes)

    csvoutput: CsverveOutputDataFrame = CsverveOutputDataFrame(
        merged_data, out_filename, dtypes_, write_header=write_header
    )
    csvoutput.write_df()


def concatenate_csv(inputfiles: List[str], output: str, write_header: bool = True) -> None:
    """
    Concatenate gzipped CSV files, dtypes in meta YAML files must be the same.

    @param inputfiles: List of gzipped CSV file paths, or a dictionary where the keys are file paths.
    @param output: Path of resulting concatenated gzipped CSV file and meta YAML.
    @param write_header: boolean, True = write header, False = don't write header.
    @return:
    """
    if isinstance(inputfiles, dict):
        inputfiles = list(inputfiles.values())

    if inputfiles == []:
        raise CsverveConcatException("nothing provided to concat")

    inputs: List[CsverveInput] = [CsverveInput(infile) for infile in inputfiles]

    dtypes: Dict[str, str] = utils.merge_dtypes([csvinput.dtypes for csvinput in inputs])

    headers: List[bool] = [csvinput.header for csvinput in inputs]

    columns: List[List[str]] = [csvinput.columns for csvinput in inputs]

    low_memory: bool = True
    if any(headers):
        low_memory = False

    if not all(columns[0] == elem for elem in columns):
        low_memory = False

    if low_memory:
        concatenate_csv_files_quick_lowmem(inputfiles, output, dtypes, columns[0], write_header=write_header)
    else:
        concatenate_csv_files_pandas(inputfiles, output, dtypes, write_header=write_header)


def annotate_csv(
        infile: str,
        annotation_df: pd.DataFrame,
        outfile,
        annotation_dtypes,
        on="cell_id",
        write_header: bool = True,
):
    """
    TODO: fill this in
    @param infile:
    @param annotation_df:
    @param outfile:
    @param annotation_dtypes:
    @param on:
    @param write_header:
    @return:
    """

    csvinput = CsverveInput(infile)
    metrics_df = csvinput.read_csv()


    # get annotation rows that correspond to rows in on
    reformed_annotation = annotation_df[annotation_df[on].isin(metrics_df[on])]

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

    output = CsverveOutputDataFrame(metrics_df, outfile, csv_dtypes, write_header=write_header)
    output.write_df()


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

    output = CsverveOutputDataFrame(metrics_df, out_f, csv_dtypes, write_header=write_header)
    output.write_df()


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

    dtypes = utils.merge_dtypes([csv_dtypes, dtypes])
    output = CsverveOutputDataFrame(
        csvinput, outfile, dtypes, write_header=write_header
    )
    output.write_df()


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
    csvoutput: CsverveOutputDataFrame = CsverveOutputDataFrame(
        df, outfile, dtypes, write_header=write_header
    )
    csvoutput.write_df()


def read_csv(infile: str, chunksize: int = None, usecols=None) -> pd.DataFrame:
    """
    Read in CSV file and return as a pandas DataFrame.

    Assumes a YAML meta file in the same path with the same name, with a .yaml extension.
    YAML file structure is atop this file.

    @param infile: Path to CSV file.
    @param chunksize: Number of rows to read at a time (optional, applies to large datasets).
    @return: pandas DataFrame.
    """
    return CsverveInput(infile).read_csv(chunksize=chunksize, usecols=usecols)
