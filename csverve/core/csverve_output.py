from typing import List, Dict, Any

import yaml
from csverve.errors import CsverveWriterError
from csverve import utils


class CsverveOutput(object):
    def __init__(
        self,
        filepath: str,
        dtypes: Dict[str, str],
        columns: List[str],
        write_header: bool = True,
        na_rep: str = 'NaN',
        sep: str = ',',
    ) -> None:
        self.filepath: str = filepath
        self._verify_input()

        self.write_header: bool = write_header
        self.dtypes: Dict[str, str] = dtypes
        self.na_rep: str = na_rep
        self.sep: str = sep
        self._convert_dtypes()

        self.columns = columns

    def _convert_dtypes(self) -> None:
        self.dtypes = {col: utils.pandas_to_std_types(dtype) for col, dtype in self.dtypes.items()}

    @property
    def yaml_file(self) -> str:
        """
        Append '.yaml' to CSV path.

        @return: YAML metadata path.
        """
        return self.filepath + '.yaml'

    def _verify_input(self):
        """
        Verify gzip status and check for yaml

        @return:
        """
        if not self.filepath.endswith('.gz'):
            raise CsverveWriterError('output must be gzipped')

    def write_yaml(self) -> None:
        """
        Write .yaml file.

        @return:
        """

        yaml_columns = [{'name': col, 'dtype': self.dtypes[col]} for col in self.columns]

        yamldata: Dict[str, Any] = {
            'header': self.write_header,
            'sep': self.sep,
            'columns': yaml_columns
        }

        with open(self.yaml_file, 'wt') as f:
            yaml.safe_dump(yamldata, f, default_flow_style=False)
