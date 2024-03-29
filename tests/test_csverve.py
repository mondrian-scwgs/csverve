#!/usr/bin/env python

"""Tests for `csverve` package."""

import itertools
import os
import random
import unittest

import csverve.api as api
import csverve.utils as utils
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from csverve import cli
from csverve.api.api import concatenate_csv_files_pandas
from csverve.api.api import concatenate_csv_files_quick_lowmem
from csverve.core.csverve_input import CsverveInput
from csverve.errors import CsverveConcatException
from csverve.errors import CsverveMergeCommonColException
from csverve.errors import CsverveMergeDtypesEmptyMergeSet
from csverve.errors import CsverveMergeException
from csverve.errors import CsverveWriterError
from csverve.errors import DtypesMergeException

from . import test_helpers as helpers


###############################################
#                fixtures                     #
################################################


@pytest.fixture
def n_dtypes():
    return random.randint(2, 10)


@pytest.fixture
def n_frames():
    return random.randint(3, 10)


@pytest.fixture
def n_rows():
    return random.randint(3, 10)


################################################
#                  tests                       #
################################################


class TestAnnotateCsv(helpers.AnnotationHelpers):
    """
    class to test annotate_csv
    """

    def test_annotate_csv(self, tmpdir, n_rows):
        """
        basic sanity check - test annotating normal csv
        :param tmpdir: temporary directory to write in
        :param n_rows: number of rows in test csvs
        """
        dtypes = {v: "int" for v in 'ABCD'}
        dtypes["cell_id"] = "str"
        ann_dtypes = {v: "int" for v in 'ERF'}

        csv, annotation, annotated = self.base_annotation_test(tmpdir, n_rows,
                                                               dtypes, ann_dtypes)

        assert self.validate_annotation_test(csv, annotation, annotated, "cell_id")

    def test_annotate_csv_annotate_not_on_cell_id(self, tmpdir, n_rows):
        """
        test annotating on different column that default "cell_id"
        :param tmpdir: temporary directory to write in
        :param n_rows: number of rows in test csvs
        """

        dtypes = {v: "int" for v in 'ABCD'}
        dtypes["TEST"] = "str"
        ann_dtypes = {v: "int" for v in 'ERF'}

        csv, annotation, annotated = self.base_annotation_test(tmpdir, n_rows,
                                                               dtypes, ann_dtypes,
                                                               on="TEST")

        assert self.validate_annotation_test(csv, annotation, annotated, "TEST")

    def test_annotate_csv_annotation_col_mismatch(self, tmpdir, n_rows):
        """
        test annotating csv where annotation_data differs in length from csv
        :param tmpdir: temporary directory to write in
        :param n_rows: number of rows in test csvs
        """

        dtypes = {v: "int" for v in 'ABCD'}
        dtypes["cell_id"] = "str"
        ann_dtypes = {v: "int" for v in 'ERF'}
        annotated = os.path.join(tmpdir, "annotated.csv.gz")

        csv, annotation = self.make_ann_test_inputs(tmpdir, n_rows, dtypes,
                                                    ann_dtypes)

        newdf = pd.Series({"E": 1, "R": 43, "F": 2, "name": "new_cell"}).to_frame().T
        newdf = newdf.set_index('name')

        annotation = pd.concat([annotation, newdf])

        api.annotate_csv(csv, annotation, annotated, ann_dtypes)

        self.validate_annotation_test(csv, annotation, annotated, "cell_id")

    def test_annotate_csv_annotation_col_dtype_mismatch(self, tmpdir, n_rows):
        """
        test annotating csv with inappropriate annotation_dtypes
        :param tmpdir: temporary directory to write in
        :param n_rows: number of rows in test csvs
        """

        dtypes = {v: "int" for v in 'ABCD'}
        dtypes["cell_id"] = "str"
        ann_dtypes = {v: "int" for v in 'ERF'}
        annotated = os.path.join(tmpdir, "annotated.csv.gz")

        csv, annotation = self.make_ann_test_inputs(tmpdir, n_rows, dtypes,
                                                    ann_dtypes)

        api.annotate_csv(csv, annotation, annotated, ann_dtypes)

        csv = CsverveInput(csv).read_csv().merge(annotation)

        assert self.dfs_exact_match(annotated, csv)

    def test_annotate_csv_no_write_header(self, tmpdir, n_rows):
        """
        test annotating csv without writing header
        :param tmpdir: temporary directory to write in
        :param n_rows: number of rows in test csvs
        """
        dtypes = {v: "int" for v in 'ABCD'}
        dtypes["cell_id"] = "str"
        ann_dtypes = {v: "int" for v in 'ERF'}

        csv, annotation, annotated = self.base_annotation_test(tmpdir, n_rows,
                                                               dtypes, ann_dtypes,
                                                               skip_header=True)

        assert self.validate_annotation_test(csv, annotation, annotated, "cell_id")


class TestConcatCsv(helpers.ConcatHelpers):
    """
    test class for csverve concat_csv
    """

    def test_concat_csv(self, tmpdir, n_rows):
        """
        basic sanity check - concat two csvs with same cols
        :param tmpdir: temporary directory to write in
        :param n_rows: number of rows in test csvs
        """
        dtypes = {v: "int" for v in 'ABCD'}
        concatenated = os.path.join(tmpdir, 'concat.csv.gz')

        dfs, csvs, ref = self.base_test_concat(n_rows, [dtypes, dtypes], write=True,
                                               get_ref=True, dir=tmpdir)

        api.concatenate_csv(csvs, concatenated)

        assert self.dfs_exact_match(ref, concatenated)

    def test_concat_csv_no_header(self, tmpdir, n_rows):
        """
        test concating csvs with no headers
        :param tmpdir: temporary directory to write in
        :param n_rows: number of rows in test csvs
        """

        dtypes = {v: "int" for v in 'ABCD'}
        concatenated = os.path.join(tmpdir, 'concat.csv.gz')

        dfs, csvs, ref = self.base_test_concat(n_rows, [dtypes, dtypes], write=True,
                                               get_ref=True, dir=tmpdir,
                                               skip_header=True)
        api.concatenate_csv(csvs, concatenated, skip_header=True)

        assert self.dfs_exact_match(ref, concatenated)

        concatenated = pd.read_csv(concatenated)  # ignore separate yaml

        assert all([col not in concatenated.columns.tolist()
                    for col in dtypes.keys()])

    def test_concat_csv_empty_inputs(self, tmpdir):
        """
        test concatenate csv with data-less input csvs
        :param tmpdir: tempdir to test in
        """
        dtypes = {v: "int" for v in 'ABCD'}
        concatenated = os.path.join(tmpdir, 'concat.csv.gz')

        dfs, csvs, ref = self.base_test_concat(0, [dtypes, dtypes], write=True,
                                               get_ref=True, dir=tmpdir)

        api.concatenate_csv(csvs, concatenated)

        assert self.dfs_exact_match(concatenated, ref)

    def test_concat_csv_nothing_to_concat(self, tmpdir):
        """
        test concat_csv with not inputs
        :param tmpdir:  tempdir to test in
        """
        concatenated = os.path.join(tmpdir, 'concat.csv.gz')

        assert self._raises_correct_error(api.concatenate_csv, [],
                                          concatenated,
                                          expected_error=CsverveConcatException)

    def test_concat_csv_one_file_to_concat(self, tmpdir, n_rows):
        """
        provide just 1 file to concat
        :param tmpdir: temp dir to test in
        :param n_rows: length of test dfs
        :return:
        """
        dtypes = {v: "int" for v in 'ABCD'}
        concatenated = os.path.join(tmpdir, 'concat.csv.gz')

        df, csv, ref = self.base_test_concat(n_rows, [dtypes], write=True,
                                             get_ref=True, dir=tmpdir)

        api.concatenate_csv(csv, concatenated)

        assert self.dfs_exact_match(ref, concatenated)

    def test_concat_csv_multiple_files_to_concat(self, tmpdir, n_rows, n_frames):
        """
        provide just 1 file to concat
        """
        dtypes = [{v: "int" for v in 'ABCD'} for _ in range(n_frames)]
        concatenated = os.path.join(tmpdir, 'concat.csv.gz')

        dfs, csvs, ref = self.base_test_concat(n_rows, dtypes, write=True,
                                               get_ref=True, dir=tmpdir)

        api.concatenate_csv(csvs, concatenated)

        assert self.dfs_exact_match(ref, concatenated)

    def test_concat_csv_different_cols(self, tmpdir, n_rows):
        """
        concat two dataframes with different columns
        """
        dtypes1 = {v: "float" for v in 'ABCD'}
        dtypes2 = {v: "float" for v in 'ABGF'}

        concatenated = os.path.join(tmpdir, 'concat.csv.gz')

        dfs, csvs, ref = self.base_test_concat(n_rows, [dtypes1, dtypes2], write=True,
                                               get_ref=True, dir=tmpdir)

        api.concatenate_csv(csvs, concatenated)

        assert self.dfs_exact_match(ref, concatenated)

    def test_concat_csv_different_dtypes(self, tmpdir, n_rows):
        """
        concat two dataframes same colnames with different dtypes
        """

        dtypes1 = {v: "float" for v in 'ABCD'}
        dtypes2 = {v: "int" for v in 'ABCD'}

        concatenated = os.path.join(tmpdir, 'concat.csv.gz')

        dfs, csvs = self.base_test_concat(n_rows, [dtypes1, dtypes2], write=True,
                                          get_ref=False, dir=tmpdir)

        assert self._raises_correct_error(api.concatenate_csv, csvs,
                                          concatenated,
                                          expected_error=DtypesMergeException)

    def test_concat_csv_with_nans(self, tmpdir, n_rows):
        """
        concat two csvs with NaNs
        """
        dtypes = {v: "float" for v in 'ABCD'}

        concatenated = os.path.join(tmpdir, 'concat.csv.gz')

        dfs = self.make_test_dfs([dtypes, dtypes], n_rows)
        csvs = [os.path.join(tmpdir, "0.csv.gz"),
                os.path.join(tmpdir, "1.csv.gz")]

        dfs[0].iloc[2, dfs[0].columns.get_loc("A")] = np.NaN
        dfs[1].iloc[2, dfs[1].columns.get_loc("A")] = np.NaN

        api.write_dataframe_to_csv_and_yaml(dfs[0], csvs[0], dtypes)
        api.write_dataframe_to_csv_and_yaml(dfs[1], csvs[1], dtypes)

        ref = pd.concat(dfs, ignore_index=True)
        api.concatenate_csv(csvs, concatenated)

        assert self.dfs_exact_match(ref, concatenated)

    def test_concat_csv_gen_nans_with_int_dtype(self, tmpdir, n_rows):
        """
        concat two csvs with NaNs
        """
        dtypes1 = {v: "int" for v in 'ABCD'}
        dtypes2 = {v: "int" for v in 'ABE'}

        concatenated = os.path.join(tmpdir, 'concat.csv.gz')

        dfs, csvs = self.base_test_concat(n_rows, [dtypes1, dtypes2], write=True,
                                          get_ref=False, dir=tmpdir)

        # check for value error - should raise internal pandas error
        assert self._raises_correct_error(api.concatenate_csv, csvs,
                                          concatenated,
                                          expected_error=CsverveWriterError)


class TestConcatCsvFilesPandas(helpers.ConcatHelpers):
    """
    test class for csverve concat_csv_files_pandas
    """

    def test_concat_csv_pandas(self, tmpdir, n_rows):
        """
        basic sanity check - concat 2 csvs with same cols
        """

        dtypes = {v: "int" for v in 'ABCD'}
        concatenated = os.path.join(tmpdir, 'concat.csv.gz')

        dfs, csvs, ref = self.base_test_concat(n_rows, [dtypes, dtypes],
                                               write=True, get_ref=True,
                                               dir=tmpdir)

        concatenate_csv_files_pandas(csvs, concatenated, dtypes, True)

        assert self.dfs_exact_match(ref, concatenated)

    def test_concat_csv_pandas_no_header(self, tmpdir, n_rows):
        dtypes = {v: "int" for v in 'ABCD'}
        concatenated = os.path.join(tmpdir, 'concat.csv.gz')

        dfs, csvs, ref = self.base_test_concat(n_rows, [dtypes, dtypes],
                                               write=True, get_ref=True,
                                               dir=tmpdir, skip_header=True)

        concatenate_csv_files_pandas(csvs, concatenated, dtypes, True)

        assert self.dfs_exact_match(ref, concatenated)


class TestConcatCsvFilesQuickLowMem(helpers.ConcatHelpers):
    """
    test class for csverve concat_csv_files_quick_lowmem
    """

    def test_quick_concat(self, tmpdir, n_rows):
        """
        sanity check - two easy csvs
        """
        dtypes = {v: "int" for v in 'ABCD'}
        concatenated = os.path.join(tmpdir, 'concat.csv.gz')

        dfs, csvs, ref = self.base_test_concat(n_rows, [dtypes, dtypes],
                                               write=True, get_ref=True,
                                               dir=tmpdir, skip_header=True)

        concatenate_csv_files_quick_lowmem(csvs, concatenated, dtypes, list(dtypes.keys()))

        assert self.dfs_exact_match(ref, concatenated)

    def test_quick_concat_scale(self, tmpdir, n_rows):
        """
        sanity check - two easy csvs
        """
        dtypes = {v: "int" for v in 'ABCD'}
        concatenated = os.path.join(tmpdir, 'concat.csv.gz')
        scale = 100

        dfs, csvs, ref = self.base_test_concat(n_rows * scale, [dtypes, dtypes],
                                               write=True, get_ref=True,
                                               dir=tmpdir, skip_header=True)

        concatenate_csv_files_quick_lowmem(csvs, concatenated, dtypes, list(dtypes.keys()))

        assert self.dfs_exact_match(ref, concatenated)


class TestWriteDataFrameToCsvAndYaml(helpers.WriteHelpers):
    """
    class to test writing of dfs to csv and yamls
    """

    def validate_write_to_csv_yaml_test(self, df, dtypes, filename,
                                        yaml_filename, skipped_header=False):
        """
        tests that write_data_frane_to_csv_and_yaml wrote data correctly
        :param df: input df
        :param dtypes: input dtypes of df
        :param filename: csv file written to
        :param yaml_filename: yaml file written to
        :param wrote_header: T/F wrote header out
        """
        assert os.path.exists(filename)
        assert os.path.exists(yaml_filename)
        assert self.write_file_successful(df, filename, skipped_header=skipped_header)
        assert self.metadata_write_successful(dtypes, yaml_filename)

    def base_write_to_csv_yaml_test(self, temp, dtypes, length, skip_header=False):
        """
        base test for write csv yaml
        """
        df = self.make_test_dfs([dtypes], length)

        csv = self.write_dfs(temp, df, [dtypes], skip_header)

        filename = csv[0]

        yaml_filename = filename + ".yaml"
        os.remove(yaml_filename)

        assert not os.path.exists(yaml_filename)

        api.write_dataframe_to_csv_and_yaml(df[0], filename, dtypes,
                                            skip_header=skip_header)

        return df[0], filename, yaml_filename

    def test_write_to_csv_yaml(self, tmpdir, n_rows):
        """
        basic sanity check - write normal df
        """
        skip_header = False
        dtypes = {v: "int" for v in "ABC"}
        df, filename, yaml_filename = self.base_write_to_csv_yaml_test(
            tmpdir,
            dtypes,
            n_rows,
            skip_header=skip_header
        )

        self.validate_write_to_csv_yaml_test(df, dtypes, filename, yaml_filename,
                                             skipped_header=skip_header)

    def test_write_to_csv_yaml_no_header(self, tmpdir, n_rows):
        """
        write single df without header
        """
        skip_header = False
        dtypes = {v: "int" for v in "ABC"}
        df, filename, yaml_filename = self.base_write_to_csv_yaml_test(
            tmpdir,
            dtypes,
            n_rows,
            skip_header=skip_header)

        self.validate_write_to_csv_yaml_test(
            df, dtypes, filename, yaml_filename,
            skipped_header=skip_header)

    def test_write_to_csv_yaml_empty(self, tmpdir):
        """
        write empty df
        """

        dtypes = {v: "int" for v in 'ABCD'}
        df = pd.DataFrame()
        filename = os.path.join(tmpdir, "df.csv.gz")
        yaml_filename = filename + ".yaml"

        api.write_dataframe_to_csv_and_yaml(df, filename, dtypes)

        assert os.path.exists(filename)
        assert os.path.exists(yaml_filename)


class TestMergeFrames(helpers.MergeHelpers):
    """
    test merge_frames
    """

    def merge_frames_directional_test(self, length, direction):
        """
        merge frames in a given direction; corresponds to "how"
        :param length: length of test dfs
        :param direction: direction to merge in (outter, inner etc.)
        """
        how = direction
        on = ["A"]
        suffs = ["", ""]
        dtypes1 = {v: "int" for v in "ABC"}
        dtypes2 = {v: "int" for v in "ADF"}
        dtypes = [dtypes1, dtypes2]

        dfs, ref = self.base_merge_test(length, how, on, suffs, dtypes)

        merged = utils.merge_frames(dfs, how=how, on=on)

        assert self.dfs_exact_match(ref, merged)

    # TODO: missing case where NaN generated with int column
    def test_merge_frames(self, n_rows):
        """
        basic sanity check - test merging of two frames on 1 col
        """
        self.merge_frames_directional_test(n_rows, "outer")

    def test_merge_frames_inner(self, n_rows):
        """
        test merging of 2 dfs on 1 col with inner merge
        """
        self.merge_frames_directional_test(n_rows, "inner")

    def test_merge_frames_left(self, n_rows):
        """
        test merging of 2 dfs on 1 col with left merge
        """

        self.merge_frames_directional_test(n_rows, "left")

    def test_merge_frames_right(self, n_rows):
        """
        test merging of 2 dfs on 1 col with right merge
        """

        self.merge_frames_directional_test(n_rows, "right")

    def test_merge_frames_multiple_cols(self, n_rows):
        """
        test merging of 2 dfs on multiple columns with right merge
        """

        how = "inner"
        on = ["A", "B"]
        suffs = ["", ""]
        dtypes1 = {v: "int" for v in "ABC"}
        dtypes2 = {v: "int" for v in "ABDF"}
        dtypes = [dtypes1, dtypes2]

        dfs, ref = self.base_merge_test(n_rows, how, on, suffs, dtypes)

        merged = utils.merge_frames(dfs, how=how, on=on)

        assert self.dfs_exact_match(ref, merged)

    def test_merge_frames_one_frame(self, n_rows):
        """
        provide just one df
        :param n_rows: number of rows in simulated df
        :return: assertion
        """

        how = "inner"
        on = ["A"]
        suffs = ["", ""]
        dtypes = {v: "int" for v in "ABC"}

        df = self.base_merge_test(n_rows, how, on, suffs, [dtypes],
                                  get_ref=False)

        merged = utils.merge_frames(df, how=how, on=on)

        assert self.dfs_exact_match(df[0], merged)

    def test_merge_frames_nothing_to_merge(self, n_rows):
        """
        test merging of 2 dfs on 0 col
        """
        dtypes1 = {v: "int" for v in 'CD'}
        dtypes2 = {v: "int" for v in 'EFGH'}

        dfs = self.make_mergeable_test_dfs([dtypes1, dtypes2], [], n_rows)

        assert self._raises_correct_error(utils.merge_frames, dfs,
                                          how="outer", on=[],
                                          expected_error=CsverveMergeException)

    def test_merge_frames_cols_to_merge_have_different_dtypes(self, n_rows):
        """
        test merging of 2 dfs on 1 col. Each dataframe has different dtype for col
        """

        dtypes1 = {v: "int" for v in 'ABCD'}
        dtypes2 = {v: "str" for v in 'AEFGH'}

        dfs = self.make_mergeable_test_dfs([dtypes1, dtypes2], [], n_rows)

        assert self._raises_correct_error(utils.merge_frames, dfs,
                                          how="outer", on=[],
                                          expected_error=CsverveMergeException)

    def test_merge_frames_no_merge_col_given(self, n_rows):
        """
        test merging of 2 dfs on no col
        """

        dtypes1 = {v: "int" for v in 'ACD'}
        dtypes2 = {v: "int" for v in 'ABG'}

        dfs = self.make_mergeable_test_dfs([dtypes1, dtypes2], [], n_rows)

        assert self._raises_correct_error(utils.merge_frames, dfs,
                                          how="outer", on=[],
                                          expected_error=CsverveMergeException)

    def test_merge_frames_with_nans(self, n_rows):
        """
        test merging of 2 dfs on 1 col which contains NaNs in each
        """
        dtypes1 = {v: "float" for v in 'ACD'}
        dtypes2 = {v: "float" for v in 'AEG'}
        how = "outer"
        on = ['A']

        dfs = self.make_mergeable_test_dfs([dtypes1, dtypes2], on, n_rows)

        dfs[0].iloc[2, dfs[0].columns.get_loc(on[0])] = np.NaN
        dfs[1].iloc[2, dfs[1].columns.get_loc(on[0])] = np.NaN

        ref = dfs[0].merge(dfs[1], how=how, on=on)

        merged = utils.merge_frames(dfs, how=how, on=on)

        assert self.dfs_exact_match(ref, merged)

    def test_merge_frames_inner_differing_vals_on_common_cols(self, n_rows):
        """
        test merging of 2 dfs on multiple columns with right merge
        """
        how = "inner"
        on = ["A"]
        dtypes1 = {v: "float" for v in "AC"}
        dtypes2 = {v: "float" for v in "ACF"}

        dfs = self.make_mergeable_test_dfs([dtypes1, dtypes2], on, n_rows)

        assert self._raises_correct_error(utils.merge_frames, dfs,
                                          how=how, on=on,
                                          expected_error=CsverveMergeCommonColException)

    def test_merge_frames_outer_differing_vals_on_common_cols(self, n_rows):
        """
        test merging of 2 dfs on multiple columns with right merge
        """
        how = "outer"
        on = ["A"]
        dtypes1 = {v: "float" for v in "AC"}
        dtypes2 = {v: "float" for v in "ACF"}

        dfs = self.make_mergeable_test_dfs([dtypes1, dtypes2], on, n_rows)

        assert self._raises_correct_error(utils.merge_frames, dfs,
                                          how=how, on=on,
                                          expected_error=CsverveMergeCommonColException)

    def test_merge_frames_inner_same_vals_on_common_cols(self, n_rows):
        """
        test merging of 2 dfs on multiple columns with right merge
        """
        how = "inner"
        on = ["A"]
        dtypes1 = {v: "float" for v in "AD"}
        dtypes2 = {v: "float" for v in "ACF"}

        dfs = self.make_mergeable_test_dfs([dtypes1, dtypes2], on, n_rows)

        merged = utils.merge_frames(dfs, how=how, on=on, lenient=True)

        ref = dfs[0].merge(dfs[1], how=how, on=on)

        assert self.dfs_exact_match(ref, merged)

    def test_merge_frames_outer_differing_vals_on_common_cols_lenient(self, n_rows):
        """
        test merging of 2 dfs on multiple columns with right merge
        """
        how = "outer"
        on = ["A"]
        dtypes1 = {v: "float" for v in "AC"}
        dtypes2 = {v: "float" for v in "ACF"}

        dfs = self.make_mergeable_test_dfs([dtypes1, dtypes2], on, n_rows)

        merged = utils.merge_frames(dfs, how=how, on=on, lenient=True)

        ref = dfs[0].merge(dfs[1][['A', 'F']], how=how, on=on)

        assert self.dfs_exact_match(ref, merged)


class TestMergeDtypes(helpers.MergeHelpers):

    def test_merge_dtypes(self):
        """
        basic sanity check - test merging of two dtype dicts
        """
        dtypes1 = {v: "float" for v in 'ACD'}
        dtypes2 = {v: "float" for v in 'ACDEF'}
        ref = {v: "float" for v in
               set(dtypes1.keys()).union(set(dtypes2.keys()))}

        merged_dtypes = utils.merge_dtypes([dtypes1, dtypes2])

        assert ref == merged_dtypes

    def test_merge_dtypes_none_given(self):
        """
        test merging of empty list of dtypes
        """
        assert self._raises_correct_error(utils.merge_dtypes, [],
                                          expected_error=CsverveMergeDtypesEmptyMergeSet)

    def test_merge_dtypes_one_given(self):
        """
        test merging of list of 1 dtype dict
        """
        dtypes1 = {v: "float" for v in 'ACD'}
        ref = dtypes1

        merged_dtypes = utils.merge_dtypes([dtypes1])

        assert ref == merged_dtypes

    def test_merge_dtypes_multiple_given(self, n_dtypes):
        """
        test merging of n_dtypes dtype dicts
        :param n_dtypes: number of dtypes to merge
        """
        dtypes = [{v: "int" for v in "".join(self._str_list(3, "A"))}
                  for _ in range(n_dtypes)]

        # taken from https://stackoverflow.com/questions/9819602/union-of-dict-objects-in-python
        ref = dict(itertools.chain.from_iterable(dct.items()
                                                 for dct in dtypes))

        merged_dtypes = utils.merge_dtypes(dtypes)

        assert ref == merged_dtypes

    @staticmethod
    def merge_dtype_test_types_different(dtype):
        """
        test merge dtypes that are different
        """
        dtypes1 = {v: dtype for v in 'ACD'}
        dtypes2 = {v: dtype for v in 'ACDEF'}
        ref = {v: dtype for v in
               set(dtypes1.keys()).union(set(dtypes2.keys()))}

        merged_dtypes = utils.merge_dtypes([dtypes1, dtypes2])

        assert ref == merged_dtypes

    def test_merge_dtypes_float(self):
        """
        merge dtypes with float types
        """
        self.merge_dtype_test_types_different("float")

    def test_merge_dtypes_str(self):
        """
        merge dtypes with str types
        """
        self.merge_dtype_test_types_different("str")

    def test_merge_dtypes_bool(self):
        """
        merge dtypes with bool types
        """
        self.merge_dtype_test_types_different("bool")

    @staticmethod
    def merge_dtype_test_types_mixed():
        """
        test merge dtypes that have mixed typing
        """
        types = ["int", "float", "bool", "str"]
        dtypes1 = {v: random.choice(types) for v in "ACD"}
        dtypes2 = {v: random.choice(types) for v in "ACD"}

        ref = {v: type for v in
               set(dtypes1.keys()).union(set(dtypes2.keys()))}

        merged_dtypes = utils.merge_dtypes([dtypes1, dtypes2])

        assert ref == merged_dtypes


class TestMergeCsv(helpers.MergeHelpers):
    """
    class to test merge_csv
    """

    def test_merge_csv(self, tmpdir, n_rows):
        """
        test merge_csv
        """
        dtypes1 = {v: "int" for v in 'ABCD'}
        dtypes2 = {v: "int" for v in 'AEFGH'}
        how = "outer"
        on = ["A"]
        suffs = ["", ""]
        merged = os.path.join(tmpdir, "merged.csv.gz")

        dfs, csvs, ref = self.base_merge_test(n_rows, how, on, suffs,
                                              [dtypes1, dtypes2],
                                              write=True, dir=tmpdir)

        api.merge_csv(csvs, merged, how=how, on=on)

        assert os.path.exists(merged)

        assert self.dfs_exact_match(ref, merged)

    def test_merge_csv_no_header(self, tmpdir, n_rows):
        """
        merge csvs with no headers
        """

        dtypes1 = {v: "int" for v in 'ABCD'}
        dtypes2 = {v: "int" for v in 'AEFGH'}
        how = "outer"
        on = ["A"]
        suffs = ["", ""]
        merged = os.path.join(tmpdir, "merged.csv.gz")

        dfs, csvs, ref = self.base_merge_test(n_rows, how, on, suffs,
                                              [dtypes1, dtypes2],
                                              write=True, dir=tmpdir,
                                              skip_header=True)

        api.merge_csv(csvs, merged, how=how, on=on)

        assert os.path.exists(merged)

        assert self.dfs_exact_match(ref, merged)


class TestCsverve(unittest.TestCase):
    """Tests for `csverve` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        _ = runner.invoke(cli.cli)
        # assert result.exit_code == 0
        # assert 'csverve.cli.main' in result.output
        # help_result = runner.invoke(cli.main, ['--help'])
        # assert help_result.exit_code == 0
        # assert '--help  Show this message and exit.' in help_result.output
