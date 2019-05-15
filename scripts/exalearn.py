
import csv
import re
import json
import datetime
import click
import os
import sys
import numpy
import pylab
import requests
import matplotlib
from urllib.parse import urlsplit, urlunsplit
import globus_sdk
from pilot.client import PilotClient
from pilot.search import scrape_metadata, gen_gmeta, GMETA_LIST
from pilot.validation import validate_json

pc = PilotClient()
pc.SEARCH_INDEX = 'be69a351-f893-4268-8647-70bcb06fcd00'
pc.SEARCH_INDEX_TEST = ''
pc.ENDPOINT = 'b8826a7a-676a-11e9-b7f5-0a37f382de32'
pc.BASE_DIR = ''
visible_to = []
output_file = 'exalearn_gmetas.json'
MASTER_MANIFEST = 'exalearn_master_manifest.json'


headers = {'Omega_m', 'Omega_b', 'Omega_L', 'H0', 'sigma_8',
           'nspec', 'w0', 'wa', 'redshift', 'data_fn', 'universe'}
hfloats = {'Omega_m', 'Omega_b', 'Omega_L', 'H0', 'sigma_8',
           'nspec', 'w0', 'wa', 'redshift'}
hints = {'universe_id'}

if sys.platform == 'darwin':
    matplotlib.use("macOSX")


def get_date_and_par(data_fn):
    info_string = data_fn.split('/')[0]
    # Matches "cosmoUniverse_2019_02_4parE"
    date_str, par = re.match('cosmoUniverse_(.+)par(\w)', info_string).groups()
    date = datetime.datetime.strptime(date_str, '%Y_%m_%d')
    return date, par


def get_universe_id(data_fn):
    # based on 'cosmoUniverse_2018_10_3parA/raw/15954636-627/out/
    #           ics_2018-10_a21297_dim512_full.npy'
    if 'parA' in data_fn or 'parB' in data_fn:
        return re.match('.*/raw/\d+-(\d+)/out/.*', data_fn).groups()[0]
    raise ValueError('Only accepts par a and par b')


def get_rows(filename, num_rows):
    """Fetch rows from a csv"""
    rows = []

    with open(filename) as fh:
        reader = csv.DictReader(fh)
        i = 0
        for row in reader:
            rows.append(row)
            i += 1
            if i == num_rows:
                break
    return rows


def coerce_type(content, keys, func):
    """Coerce a string to be a different type. Used with "hfloats" and "hints"
    above. """
    for key in keys:
        if content.get(key):
            content[key] = func(content[key])


def gen_image(filename):
    array_filename = filename
    image_filename = array_filename.rstrip('npy') + 'png'
    density = numpy.load(array_filename)

    # sum along a section along the x axis
    # add 1 to eliminate zero and take the log
    projection_x = numpy.log(density[:128].sum(0) + 1)

    figsize = (numpy.array(projection_x.shape) / 100.0)[::-1]
    fig = pylab.figure()
    pylab.axes([0, 0, 1, 1])  # Make the plot occupy the whole canvas
    pylab.axis('off')
    fig.set_size_inches(figsize)
    pylab.imshow(projection_x, origin='lower')
    pylab.savefig(image_filename, facecolor='black', edgecolor='black',
                  dpi=100)
    return image_filename


def parse_file(filename, num_rows):
    records = []
    for row in get_rows(filename, num_rows):
        if not set(row.keys()).issubset(headers):
            raise ValueError(f'Expected header set:\n{headers}\n'
                             f'Differs from {filename}:\n{set(row.keys())}')

        data_fname = row['data_fn']
        url = pc.get_globus_http_url(data_fname, '')
        data = scrape_metadata(filename, url)
        date, par = get_date_and_par(data_fname)
        data['cosmo'] = row
        if row.get('universe'):
            data['cosmo']['universe_id'] = row['universe'].split('-')[1]
            data['cosmo'].pop('universe')
        else:
            data['cosmo']['universe_id'] = get_universe_id(data_fname)
        data['cosmo']['par'] = par
        data['cosmo']['gmeta_dates'] = {'created': date.strftime('%Y-%m-%d')}

        coerce_type(data['cosmo'], hfloats, float)
        coerce_type(data['cosmo'], hints, int)

        par, uid, rs = (data['cosmo']['par'], data['cosmo']['universe_id'],
                        float(data['cosmo']['redshift']))
        identifier = f'par{par}-{uid}-{rs:.1f}'
        title = f'Par{par} - Universe {uid} - Redshift {rs:.1f}'
        data['cosmo']['redshift_str'] = str(data['cosmo']['redshift'])
        data['dc']['creators'] = [{'creatorName': 'Mathuriya, Amrita'},
                                  {'creatorName': 'Bard, Deborah'},
                                  {'creatorName': 'Mendygral, Peter'},
                                  {'creatorName': 'Meadows, Lawrence'},
                                  {'creatorName': 'Arnemann, James'},
                                  {'creatorName': 'Shao, Lei'},
                                  {'creatorName': 'He, Siyu'},
                                  {'creatorName': 'Karna, Tuomas'},
                                  {'creatorName': 'Moise, Daina'},
                                  {'creatorName': 'Pennycook, Simon J.'},
                                  {'creatorName': 'Maschoff, Kristyn'},
                                  {'creatorName': 'Sewall, Jason'},
                                  {'creatorName': 'Kumar, Nalini'},
                                  {'creatorName': 'Ho, Shirley'},
                                  {'creatorName': 'Ringenburg, Mike'},
                                  {'creatorName': 'Prabhat'},
                                  {'creatorName': 'Lee, Victor'}]
        data['dc']['formats'] = ['application/octet-stream']
        data['dc']['subjects'] = [{'subject': 'Cosmology'}]
        data['dc']['titles'] = [{'title': title, 'type': 'Other'}]
        data['dc']['dates'] = [{'date': date.isoformat() + 'Z',
                               'dateType': 'Created'}]
        data['dc']['publisher'] = 'Lawrence Berkeley National Laboratory'
        del data['files']
        del data['ncipilot']
        del data['field_metadata']

        validate_json('dc', data)
        # Make sure we aren't adding more fields
        subject = f'gsearch://{identifier}'
        records.append(gen_gmeta(subject, visible_to, data, validate=False))

    stripped_records = [r['ingest_data']['gmeta'][0] for r in records]
    gmetas = GMETA_LIST.copy()
    gmetas['ingest_data']['gmeta'] = stripped_records
    return stripped_records, gmetas


def remote_file_exists(gpath):
    path = urlsplit(gpath).path
    tauth = pc.get_authorizers()['transfer.api.globus.org']
    tc = globus_sdk.TransferClient(authorizer=tauth)
    resp = tc.operation_ls(pc.ENDPOINT, path=os.path.dirname(path))
    rfnames = [r['name'] for r in resp.data['DATA']]
    if os.path.basename(path) in rfnames:
        return True
    return False


@click.group()
def cli():
    pass


@cli.command(help='Print record metadata to console instead of ingesting')
@click.argument('filename', type=click.Path(exists=True, file_okay=True,
                dir_okay=False, readable=True, resolve_path=True))
@click.option('-n', default=10)
def dump(filename, n):
    stripped_records, _ = parse_file(filename, n)
    click.echo(json.dumps(stripped_records[:n], indent=2))


@cli.command(help='Download full datasets and create/upload images. Also '
                  'generates manifests')
@click.argument('filename', type=click.Path(exists=True, file_okay=True,
                dir_okay=False, readable=True, resolve_path=True))
@click.option('-n', default=10)
def images(filename, n):
    stripped_records, _ = parse_file(filename, n)
    fnames = [r['content']['cosmo']['data_fn'] for r in stripped_records]
    urls = [urlunsplit(
            ('https', '{}.e.globus.org'.format(pc.ENDPOINT),
             os.path.join('CosmoFlow', f), '', '')
            ) for f in fnames
            ]
    if not os.path.exists(MASTER_MANIFEST):
        man = {}
    else:
        with open(MASTER_MANIFEST) as fh:
            click.echo(f'Loaded {MASTER_MANIFEST}')
            man = json.load(fh)

    for url in urls:

        remote_image_path = url.rstrip('npy') + 'png'

        data_filename = os.path.basename(url)
        if url in man.keys():
            click.secho(f'{url} already completed, skipping...')
            continue

        click.echo(f'Downloading {url}')
        response = requests.get(url, headers=pc.http_headers, stream=True)
        response.raise_for_status()
        with open(data_filename, 'wb') as fh:
            for chunk in response.iter_content(chunk_size=2048):
                fh.write(chunk)

        local_image_path = gen_image(data_filename)
        man[url] = {
            'data_png': remote_image_path,
            'files': scrape_metadata(data_filename, url)['files']
        }
        with open(MASTER_MANIFEST, 'w+') as fh:
            fh.write(json.dumps(man))
        #os.unlink(data_filename)

        if remote_file_exists(remote_image_path):
            click.echo('Skipping upload, already exists.')
            continue

        assert remote_image_path.endswith('.png')
        with open(os.path.basename(remote_image_path), 'rb') as fh:
            requests.put(
                remote_image_path, headers=pc.http_headers,
                data=fh, allow_redirects=False)
        os.unlink(local_image_path)

    click.secho('Done!', fg='green')


@cli.command(help='Command will ask confirmation before ingesting.')
@click.argument('filename', type=click.Path(exists=True, file_okay=True,
                dir_okay=False, readable=True, resolve_path=True))
@click.option('-n', default=10000)
def ingest(filename, n):
    stripped_records, gmetas = parse_file(filename, n)
    subs = [r['subject'] for r in stripped_records]
    click.secho(f'Unique Subjects: {len(set(subs))}', fg='green')
    dsubs = '\n'.join(subs[:10])

    iname = pc.gsearch.get_index(pc.SEARCH_INDEX).data['display_name']
    click.secho(f'Subjects:\n{dsubs}', fg='green')
    click.secho(f'Ingest {len(stripped_records)} documents to index: ',
                nl=False, fg='yellow')
    click.secho(iname, nl=False, bg='yellow')
    if not click.confirm('?'):
        return
    pc.ingest_entry(gmetas)


cli.add_command(ingest)
cli.add_command(dump)
if __name__ == '__main__':
    cli()
