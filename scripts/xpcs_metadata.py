"""
Gather metadata for an hdf file into a json file.

Usage: python xpcs_metadata.py
"""
import os
import json
import numpy
import click
import h5py

GENERAL_METADATA = {
    "creators": [
        {
            "creatorName": "Suresh Narayanan"
        }
    ],
    "publicationYear": "2019",
    "publisher": "Argonne National Lab",
    "resourceType": {
        "resourceType": "Dataset",
        "resourceTypeGeneral": "Dataset"
    },
    "subjects": [
        {
            "subject": "beamline"
        }
    ],
}

# Keys that cause ingest into Globus Search to fail. This is likely due to
# another key of the same name being ingested previously, causing the types
# not to match (After first ingest, you cannot ingest a different type).
SPOILED_KEYS = ['measurement.instrument.source_begin.datetime']


@click.group(help='Tiny client for generating metadata out of a beamline hdf '
                  'file. Output can then be uploaded using: '
                  'pilot upload -j metadata mydataframe /')
def cli():
    pass


def gather_items(hdf5_dataframe):
    items = {}

    def gather_item(name, node):
        if isinstance(node, h5py.Dataset) and isinstance(node.value,
                                                         numpy.ndarray):
            val = node.value.tolist()[0]
            val = val[0] if len(val) == 1 else val
            # Skip if it isn't a relatively small datatype
            if not isinstance(val, list):
                items[name.replace('/', '.')] = val
            elif isinstance(val, list) and len(val) <= 3:
                items[name.replace('/', '.')] = val

    hdf5_dataframe.visititems(gather_item)
    return items


@click.command(help='Gather metadata into json file for upload')
@click.argument('dataframe',
                type=click.Path(exists=True, file_okay=True, dir_okay=False,
                                readable=True, resolve_path=True))
def gather(dataframe):
    hframe = h5py.File(dataframe, 'r')

    metafilename, _ = os.path.splitext(os.path.basename(dataframe))
    metafilename += '.json'
    metadata = GENERAL_METADATA.copy()
    metadata.update(gather_items(hframe))

    # Ingesting spoiled keys can cause explosive un-ingests
    for key in SPOILED_KEYS:
        if key in metadata.keys():
            print(f'Popping toxic key "{key}"')
            metadata.pop(key)

    with open(metafilename, 'w') as f:
        f.write(json.dumps(metadata, indent=4))
    print(f'Wrote {metafilename}')


cli.add_command(gather)

if __name__ == '__main__':
    cli()
