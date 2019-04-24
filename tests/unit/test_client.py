import os
from urllib.parse import urlparse
from pilot.client import PilotClient

def test_get_index():
    pc = PilotClient()
    assert pc.get_index() == pc.SEARCH_INDEX


def test_get_test_index():
    pc = PilotClient()
    index = pc.get_index(test=True)
    assert index == pc.SEARCH_INDEX_TEST


def test_get_path():
    pc = PilotClient()
    path = pc.get_path('dataframe.dat', 'my_folder')
    pieces = path.split('/')
    assert 'my_folder' in pieces
    assert 'dataframe.dat' in pieces
    assert pc.BASE_DIR in path
    assert pc.TESTING_DIR not in path


def test_get_test_path():
    pc = PilotClient()
    path = pc.get_path('dataframe.dat', 'my_folder',
                                        test=True)
    pieces = path.split('/')
    assert 'my_folder' in pieces
    assert 'dataframe.dat' in pieces
    assert pc.TESTING_DIR in path


def test_get_globus_http_url():
    pc = PilotClient()
    url = pc.get_globus_http_url('dataframe.dat', 'my_folder')
    purl = urlparse(url)
    assert purl.netloc == pc.ENDPOINT + '.e.globus.org'
    assert purl.scheme == 'https'
    assert 'my_folder' in purl.path
    assert pc.TESTING_DIR not in purl.path


def test_get_test_globus_http_url():
    pc = PilotClient()
    url = pc.get_globus_http_url('dataframe.dat', 'my_folder',
                                             test=True)
    purl = urlparse(url)
    assert pc.TESTING_DIR in purl.path


def test_get_globus_url():
    pc = PilotClient()
    url = pc.get_globus_url('dataframe.dat', 'my_folder')
    purl = urlparse(url)
    assert purl.netloc == pc.ENDPOINT
    assert purl.scheme == 'globus'
    assert 'my_folder' in purl.path
    assert pc.TESTING_DIR not in purl.path


def test_get_test_globus_url():
    pc = PilotClient()
    url = pc.get_globus_url('dataframe.dat', 'my_folder',
                                        test=True)
    purl = urlparse(url)
    assert pc.TESTING_DIR in purl.path


def test_get_subject_url():
    pc = PilotClient()
    args = ('dataframe.dat', 'my_folder', False)
    test_args = ('dataframe.dat', 'my_folder', True)
    assert pc.get_globus_url(*args) == \
           pc.get_subject_url(*args)
    assert pc.get_globus_url(*test_args) == \
           pc.get_subject_url(*test_args)


def test_upload_gcp(mock_transfer_client, simple_tsv, mock_config):
    pc = PilotClient()
    result = pc.upload_gcp(simple_tsv, 'bar', test=True)
    assert result.data['code'] == 'Accepted'

