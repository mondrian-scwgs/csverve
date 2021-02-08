"""Console script for csverve."""
import sys
import click
import csverve


@click.group()
def cli():
    pass


@cli.command()
@click.option('--in_f', multiple=True, required=True, help='CSV file path, allows multiple paths.')
@click.option('--out_f', required=True, help='Path of resulting merged CSV.')
@click.option('--how', required=True, help='How to join CSVs.')
@click.option('--on', multiple=True, required=False, help='Column to join CSVs on, allowes multiple.')
@click.option('--write_header', is_flag=True, default=False, help='Writer header to resulting CSV.')
def merge_csv(
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


if __name__ == "__main__":
    cli()
