"""Console script for csverve."""
import sys
import click
import csverve
import os


@click.group()
def cli():
    pass


@cli.command()
@click.option('--in_f', multiple=True, required=True, help='CSV file path, allows multiple paths.')
@click.option('--out_f', required=True, help='Path of resulting merged CSV.')
@click.option('--how', required=True, help='How to join CSVs.')
@click.option('--on', multiple=True, required=False, help='Column to join CSVs on, allowes multiple.')
@click.option('--write_header', is_flag=True, default=False, help='Writer header to resulting CSV.')
def merge(
    in_f,
    out_f,
    how,
    on,
    write_header,
):
    files = list(in_f)
    in_filenames = {}

    counter = 1
    for file in files:
        in_filenames[f"file_{counter}"] = file
        counter += 1

    csverve.merge_csv(
        in_filenames,
        out_f,
        how,
        ','.join(list(on)),
        write_header,
    )


@cli.command()
@click.option('--in_f', required=True, help='CSV file path. Expects YAML w/ the same path as CSV with .yaml extension.')
@click.option('--out_f', required=True, help='Path of resulting merged CSV.')
@click.option('--write_header', is_flag=True, default=False, help='Writer header to resulting CSV.')
def rewrite(
    in_f,
    out_f,
    write_header,
):
    assert os.path.exists(in_f)
    assert os.path.exists(f"{in_f}.yaml")

    csverve.rewrite_csv_file(
        in_f,
        out_f,
        write_header,
    )


@cli.command()
@click.option('--in_f', multiple=True, required=True, help='CSV file path, allows multiple paths.')
@click.option('--out_f', required=True, help='Path of resulting merged CSV.')
@click.option('--write_header', is_flag=True, default=False, help='Writer header to resulting CSV.')
def concat(
    in_f,
    out_f,
    write_header,
):
    files = list(in_f)
    in_filenames = {}

    counter = 1
    for file in files:
        in_filenames[f"file_{counter}"] = file
        counter += 1

    csverve.concatenate_csv(
        in_filenames,
        out_f,
        write_header,
    )


@cli.command()
@click.option('--in_f', multiple=True, required=True, help='CSV file path, allows multiple paths.')
@click.option('--out_f', required=True, help='Path of resulting merged CSV.')
@click.option('--col_name', required=True, help='Column name to be added.')
@click.option('--col_val', required=True, help='Column value to be added (one value for all).')
@click.option('--col_dype', required=True, help='Column pandas dtype.')
@click.option('--write_header', is_flag=True, default=False, help='Writer header to resulting CSV.')
def annotate(
    in_f,
    out_f,
    col_name,
    col_val,
    col_dtype,
    write_header,
):
    csvinput = csverve.CsverveInput(in_f)
    metrics_df = csvinput.read_csv()
    metrics_df[col_name] = col_val

    csv_dtypes = csvinput.dtypes
    csv_dtypes[col_name] = col_dtype

    output = csverve.CsverveOutput(out_f, csv_dtypes, header=write_header)
    output.write_df(metrics_df)


if __name__ == "__main__":
    cli()
