import os
import json
import time
import globus_sdk
import urllib
import logging
from globus_sdk import AuthClient, SearchClient, TransferClient
from fair_research_login import NativeClient, LoadError, ScopesMismatch
from pilot import (
    profile, config, globus_clients, exc, logging_cfg, context, search,
    transfer_log, search_discovery, project as project_module,
)

logging_cfg.setup_logging()
log = logging.getLogger(__name__)


class PilotClient(NativeClient):
    r"""
    The Pilot Client is the core class of this library, providing a set of
    operations to authenticate users, track which project they're using, and
    handle core operations such as login, logout, upload, download,
    building URLs, and getting authenticated Globus Service clients.

    ** Methods **

    *  :py:meth:`.login`
    *  :py:meth:`.logout`
    *  :py:meth:`.is_logged_in`
    *  :py:meth:`.get_auth_client`
    *  :py:meth:`.get_search_client`
    *  :py:meth:`.get_transfer_client`
    *  :py:meth:`.get_http_client`
    *  :py:meth:`.get_group`
    *  :py:meth:`.get_endpoint`
    *  :py:meth:`.get_index`
    *  :py:meth:`.get_path`
    *  :py:meth:`.get_globus_http_url`
    *  :py:meth:`.get_globus_url`
    *  :py:meth:`.get_globus_app_url`
    *  :py:meth:`.get_portal_url`
    *  :py:meth:`.get_subject_url`
    *  :py:meth:`.ls`
    *  :py:meth:`.mkdir`
    *  :py:meth:`.get_search_entry`
    *  :py:meth:`.ingest_entry`
    *  :py:meth:`.delete_entry`
    *  :py:meth:`.upload`
    *  :py:meth:`.download`
    *  :py:meth:`.delete`
    ** Example **
    Login, upload, show various access methods, then delete.

    pc = PilotClient()

    # Setup the client (You can also do this with the CLI, both use the same
    #                   credentials)
    pc.login()
    pc.project.current = 'foo'

    # Create a directory, Upload a file to it
    pc.mkdir('bar')
    pc.upload('moo.txt', 'bar')

    # Get resources associated with the file
    pc.ls('bar')
    pc.get_path('bar/moo.txt')
    pc.get_portal_url('bar/moo.txt')
    pc.get_globus_app_url('bar/moo.txt')
    pc.get_globus_http_url('bar/moo.txt')
    pc.download('bar/moo.txt')

    # Delete the file
    pc.delete_entry('bar/moo.txt')
    pc.delete('bar/moo.txt')
    """
    DEFAULT_SCOPES = [
        'profile',
        'openid',
        'urn:globus:auth:scope:search.api.globus.org:all',
        'urn:globus:auth:scope:transfer.api.globus.org:all',
    ]
    GROUPS_SCOPE = 'urn:globus:auth:scope:nexus.api.globus.org:groups'

    def __init__(self):
        self.config = config.Config()
        self.context = context.Context(self)
        default_scopes = self.context.get_value('scopes')
        default_scopes = default_scopes or self.DEFAULT_SCOPES

        super().__init__(client_id=self.context.get_value('client_id'),
                         token_storage=self.config,
                         default_scopes=default_scopes,
                         app_name=self.context.get_value('app_name'))
        self.project = project_module.Project()
        self.profile = profile.Profile()

    def login(self, *args, **kwargs):
        r"""
        Do a Native App Auth Flow to get tokens for requested scopes, for
        general project usage
        **Parameters**
        ``no_local_server`` (*bool*)
          Disable spinning up a local server to automatically copy-paste the
          auth code. THIS IS REQUIRED if you are on a remote server, as this
          package isn't able to determine the domain of a remote service. When
          used locally with no_local_server=False, the domain is localhost with
          a randomly chosen open port number.
        ``no_browser`` (*string*)
          Do not automatically open the browser for the Globus Auth URL.
          Display the URL instead and let the user navigate to that location.
        ``requested_scopes`` (*list*)
          A list of scopes to request of Globus Auth during login.
          Example:
          ['openid', 'profile', 'email']
        ``refresh_tokens`` (*bool*)
          Ask for Globus Refresh Tokens to extend login time.
        ``force`` (*bool*)
          Force a login flow, even if loaded tokens are valid.
        """
        super().login(*args, **kwargs)
        auth_cli = self.get_auth_client()
        user_info = auth_cli.oauth2_userinfo()
        self.profile.save_user_info(user_info.data)

    def logout(self):
        """Clear tokens from configfile"""
        super().logout()
        self.config.clear()

    def is_logged_in(self):
        """Check if the user is logged in and tokens are active."""
        try:
            self.load_tokens(requested_scopes=self.context.get_value('scopes'))
            return True
        except LoadError:
            return False

    def get_auth_client(self):
        """
        Returns a live Globus Auth Client based on user login info.
        https://globus-sdk-python.readthedocs.io/en/stable/clients/auth/
        :return:
        """
        authorizer = self.get_authorizers()['auth.globus.org']
        return AuthClient(authorizer=authorizer)

    def get_search_client(self):
        """Returns a live Search Client based on user login info
        https://globus-sdk-python.readthedocs.io/en/stable/clients/search/
        """
        authorizer = self.get_authorizers()['search.api.globus.org']
        return SearchClient(authorizer=authorizer)

    def get_transfer_client(self):
        """
        Returns a live transfer client based on user info
        https://globus-sdk-python.readthedocs.io/en/stable/clients/transfer/
        """
        authorizer = self.get_authorizers()['transfer.api.globus.org']
        return TransferClient(authorizer=authorizer)

    def get_nexus_client(self):
        """
        Returns a nexus client, used for groups. Please don't use this directly
        """
        rs = {'requested_scopes': [self.GROUPS_SCOPE]}
        try:
            authorizer = self.get_authorizers_by_scope(**rs)[self.GROUPS_SCOPE]
            return globus_clients.NexusClient(authorizer=authorizer)
        except ScopesMismatch:
            return None

    def get_http_client(self, project):
        r"""
        Returns a general Globus HTTP Client for http uploads/downloads
        **Parameters**
        ``project`` (*bool*)
          The project to upload/download to/from
        """
        # Fetch the base url for the globus http endpoint
        url = urllib.parse.urlparse(self.get_globus_http_url(''))
        base_url = urllib.parse.urlunparse((url.scheme, url.netloc, '', '',
                                            '', ''))
        rs = self.project.get_info(project)['resource_server']
        auth = self.get_authorizers()[rs]
        return globus_clients.HTTPFileClient(authorizer=auth,
                                             base_url=base_url)

    def get_group(self, project=None):
        """
        Get the group for a given project.
        **Parameters**
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        """
        return self.project.get_info(project)['group'] or 'public'

    def get_endpoint(self, project=None):
        """
        Get the configured Globus Endpoint for the given project. Project
        defaults to current project if None.
        **Parameters**
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        """
        return self.project.get_info(project)['endpoint']

    def get_index(self, project=None):
        """
        Get the configured search index for the given project. Project defaults
        to current project if None.
        **Parameters**
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        """
        return self.project.get_info(project)['search_index']

    def get_path(self, path, project=None, relative=True):
        """
        Resolve the absolute Globus endpoint path given a relative project
        path. Raises PilotClientException if relative=False and path is not
        in the project's directory.
        **Parameters**
        ``path`` (*path string*)
          Path to a local resource on this project
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        ``relative`` (*bool*)
          If True, prepends the path to the project. If False,
          does not prepend path but ensures it's in the project's directory
        **Examples**
        >>> pc.get_path('foo/bar.txt')
        '/projects/myproject/foo/bar.txt'
        >>> pc.get_path('/projects/goo/moo.txt', project='goo', relative=False)
        '/projects/goo/moo.txt'
        """
        bdir = self.project.get_info(project)['base_path']
        path = path.lstrip('.')
        if relative is True:
            path = path.lstrip('/')
            if path:
                path = os.path.join(bdir, path.lstrip('/'))
            else:
                path = bdir
        else:
            if bdir not in path:
                log.warning(
                    'Absolute path {} not in project {} path {}'.format(
                        path, project or self.project.current, bdir)
                )
        log.debug('Path: {}'.format(path))
        return path

    def get_globus_http_url(self, path, project=None, relative=True):
        """
        Resolve an http url based on the path.
        **Parameters**
        ``path`` (*path string*)
          Path to a local resource on this project
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        ``relative`` (*bool*)
          If True, prepends the path to the project. If False,
          does not prepend path but ensures it's in the project's directory
        **Examples**
        >>> pc.get_globus_http_url('foo/bar.txt')
        'https://5590f6aa-a9f7-4cc3-b130-afc26e2ca0c0.e.globus.org/projects/
         myproject/foo/bar.txt'
        """
        host = '{}.e.globus.org'.format(self.get_endpoint(project))
        path = self.get_path(path, project, relative)
        parts = ['https', host, path, '', '', '']
        return urllib.parse.urlunparse(parts)

    def get_globus_url(self, path, project=None, relative=True):
        """
        Resolve a Globus URL for a resource on a Globus Endpoint
        **Parameters**
        ``path`` (*path string*)
          Path to a local resource on this project
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        ``relative`` (*bool*)
          If True, prepends the path to the project. If False,
          does not prepend path but ensures it's in the project's directory
        **Examples**
        >>> pc.get_globus_url('foo/bar.txt')
        'globus://5590f6aa-a9f7-4cc3-b130-afc26e2ca0c0/projects/myproject/
         foo/bar.txt'
        """
        path = self.get_path(path, project, relative)
        parts = ['globus', self.get_endpoint(project), path, '', '', '']
        return urllib.parse.urlunparse(parts)

    def get_globus_app_url(self, path, project=None, relative=True):
        """
        Resolve a URL to the Globus Webapp
        **Parameters**
        ``path`` (*path string*)
          Path to a local resource on this project
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        ``relative`` (*bool*)
          If True, prepends the path to the project. If False,
          does not prepend path but ensures it's in the project's directory
        **Examples**
        >>> pc.get_globus_app_url('foo/bar.txt')
        'https://app.globus.org/file-manager?
         origin_id=e55b4eab-6d04-11e5-ba46-22000b92c6ec&
         origin_path=%2FXPCSDATA%2Ftest%2Fnick-testing%2Ffoo%2Fbar.txt'
        """
        path = self.get_path(path, project, relative)
        params = {'origin_id': self.get_endpoint(project), 'origin_path': path}
        return urllib.parse.urlunparse([
            'https', 'app.globus.org', 'file-manager', '',
            urllib.parse.urlencode(params), ''
        ])

    def get_portal_url(self, path=None, project=None):
        """
        Resolve a URL to the Globus Webapp. Relies on the context setting:
          'projects_portal_url'
        Examples include:
          https://myportal.com/myprojects/{{project}}/{{subject}}/'
          https://example.com/{{project}}/foo/bar/{{subject}}/moo/
        **Parameters**
        ``path`` (*path string*)
          Path to a local resource on this project
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        **Examples**
        >>> pc.get_portal_url('foo/bar.txt')
        'https://myportal/foo/bar.txt'
        """
        portal_url = self.context.get_value('projects_portal_url')
        if not portal_url:
            return None
        project = project or self.project.current
        subject = ''
        if '{project}' not in portal_url or '{subject}' not in portal_url:
            log.warning('Invalid portal url "{}", the url string requires'
                        '"{{project}} and {{subject}}, for example:'
                        'https://myportal/myprojects/{{project}}/{{subject}}/'
                        .format(portal_url))
        if path:
            sub = self.get_subject_url(path, project=project)
            subject = urllib.parse.quote_plus(urllib.parse.quote_plus(sub))
        portal_url = portal_url.format(project=project, subject=subject)
        # If no subject was given, remove trailing slash
        if portal_url.endswith('//'):
            portal_url = portal_url[0:-1]
        return portal_url

    def get_subject_url(self, path, project=None, relative=True):
        """
        Resolve a subject URL for a given resource, as it should appear as a
        subject in Globus Search. Typically this is the same as a "Globus" URL
        and is equivalent to get_globus_url()
        """
        return self.get_globus_url(path, project, relative)

    def ls(self, path, project=None, relative=True, extended=False):
        """
        Perform a list on the remote endpoint for the given project, and list
        the contents of 'path'. Returns contents in a list.
        **Parameters**
        ``path`` (*path string*)
          Path to a local resource on this project
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        ``relative`` (*bool*)
          If True, prepends the path to the project. If False,
          does not prepend path but ensures it's in the project's directory
        ``extended`` (*bool*)
          Return a dict instead for each value, where the key is each item of
          content and the values are all file details for each item.
        **Examples**
        >>> pc.ls('bar')
        ['file1.png', 'foo', 'moo']
        """
        path = self.get_path(path, project, relative)
        endpoint = self.get_endpoint(project)
        r = self.get_transfer_client().operation_ls(endpoint, path=path)
        if extended:
            return {f['name']: f for f in r['DATA']}
        return [f['name'] for f in r['DATA']]

    def mkdir(self, path, project=None, relative=True):
        """
        Create a directory in the given project
        **Parameters**
        ``path`` (*path string*)
          Path to a local resource on this project
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        ``relative`` (*bool*)
          If True, prepends the path to the project. If False,
          does not prepend path but ensures it's in the project's directory
        **Examples**
        >>> pc.mkdir('myfolder')
        """
        rpath = self.get_path(path, project=project, relative=relative)
        tc = self.get_transfer_client()
        tc.operation_mkdir(self.get_endpoint(project), rpath)

    def search(self, project=None, index=None, custom_params=None):
        """
        Perform a search for records in a given project and index, optionally
        with custom_parameters. Returns the raw response from the Globus SDK
        SearchClient.post_search().
        **Parameters**
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        ``index`` (*bool*)
          If True, prepends the path to the project. If False,
          does not prepend path but ensures it's in the project's directory
        ``custom_params`` (*dict*)
          Allows setting custom parameters for modifying results returned.
          Standard params include the following:
          search_data = {
            'q': '*',
            'filters': {
                'field_name': 'project_metadata.project-slug',
                'type': 'match_all',
                'values': [project],
            },
            'limit': 100,
            'offset': 0,
          }
          Custom params will override these params (this may result in
          unexpected results from other projects if 'filters' is overrided).
        **Examples**
        Fetch all results in this project:
        >>> pc.list_entries()
        Fetch results in the 'foo' directory:
        >>> pc.list_entries('foo')
        """
        sc = self.get_search_client()
        project = project or self.project.current
        index = index or self.get_index(project=project)
        search_data = {
            'q': '*',
            'filters': {
                'field_name': 'project_metadata.project-slug',
                'type': 'match_all',
                'values': [project],
            },
            'limit': 100,
            'offset': 0,
        }
        search_data.update(custom_params or {})
        return sc.post_search(index, search_data).data

    def list_entries(self, path='', project=None, relative=True):
        """Search for files in the given project that match the given path.
        Returns a list of Globus Search GMetaEntries for any matches it finds.
        Paths for files in multi-file collections will return no results,
        use 'pc.get_search_entry' instead to find files in multi-file results.
        Paths matching specific entries will return only that entry.
        **Parameters**
        ``path`` (*path string*)
          Path to a local resource on this project. An empty path will return
          all entries in this project
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        ``relative`` (*bool*)
          If True, prepends the path to the project. If False,
          does not prepend path but ensures it's in the project's directory
        **Examples**
        Fetch all results in this project:
        >>> pc.list_entries('')
        Fetch results in the 'foo' directory:
        >>> pc.list_entries('foo')
        """
        project = project or self.project.current
        raw = self.search(project=project)
        log.info('Fetching entry list for project {} path {}'.format(project,
                                                                     path))
        path = self.get_path(path, project=project, relative=relative)
        return [ent for ent in raw['gmeta'] if path in ent.get('subject')]

    def get_search_entry(self, path, project=None, relative=True,
                         resolve_collections=True):
        """
        Get a search entry for a given resource
        **Parameters**
        ``path`` (*path string*)
          Path to a local resource on this project
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        ``relative`` (*bool*)
          If True, prepends the path to the project. If False,
          does not prepend path but ensures it's in the project's directory
        ``resolve_collections`` (*bool*)
          If the path given points to what might be a multi-file directory
          entry, attempt to resolve the entry.
        **Examples**
        >>> pc.get_search_entry('foo.txt')
          {'dc': {'creators': [{'creatorName': 'NOAA'}],
            'dates': [{'date': '2019-08-29T19:36:13.064917Z',
                    'dateType': 'Created'}],
             ...
          }
        """
        sc = self.get_search_client()
        project = project or self.project.current
        subject = self.get_subject_url(path, project, relative)
        try:
            entry = sc.get_subject(self.get_index(project), subject)
            return entry['content'][0]
        except globus_sdk.exc.SearchAPIError as sapie:
            if sapie.code == 'NotFound.Generic' and resolve_collections:
                return search_discovery.get_sub_in_collection(
                    subject, self.list_entries())

    def ingest_entry(self, gmeta_entry, index=None):
        """
        Ingest a complete gmeta_entry into search.
        **Parameters**
        ``gmeta_entry`` (*dict*)
          A dict matching a Globus Search gmeta entry. An example is here:
          https://globus-sdk-python.readthedocs.io/en/stable/clients/search/#globus_sdk.SearchClient.ingest  # noqa
          {'@version': '2016-11-09',
           'ingest_data': {'@version': '2016-11-09',
                             'gmeta': [{'@version': '2016-11-09',
                                        'content': {},
                                        'id': 'metadata',
                                        'subject': 'foo',
                                        'visible_to': ['public']}]},
           'ingest_type': 'GMetaList'}
        ``index`` (*string*)
          The index to ingest to. Defaults to configured current project index
        **Examples**
        >>> from pilot.search import scrape_metadata, gen_gmeta
        >>> metadata = scrape_metadata('moo.txt')
        >>> subject = pc.get_subject_url('bar/moo.txt')
        >>> group = pc.get_group()
        >>> gmeta = gen_gmeta(subject, [group], metadata)
        >>> pc.ingest_entry(gmeta)
        True
        """
        sc = self.get_search_client()
        index = index or self.get_index()
        result = sc.ingest(index, gmeta_entry)
        pending_states = ['PENDING', 'PROGRESS']
        log.info('Ingesting to {}'.format(index))
        task_status = sc.get_task(result['task_id'])['state']
        while task_status in pending_states:
            log.debug(f'Search task still {task_status}')
            time.sleep(.2)
            task_status = sc.get_task(result['task_id'])['state']
        task_info = sc.get_task(result['task_id'])
        if task_info['state'] != 'SUCCESS':
            log.debug(json.dumps(task_info.data, indent=4))
            log.warning('It is possible the ingest error occurred due to '
                        'types changing. For example: {"myval": 5} --> '
                        '{"myval": "5"} is not supported. Please verify new '
                        'metadata has not changed types.')
            msg = ('Globus Search Error: "{}". For more info, set '
                   '`export PILOT_LOGGING=DEBUG`'.format(task_info['message']))
            raise exc.PilotClientException(msg)
        return True

    def delete_entry(self, path, entry_id='metadata', full_subject=False):
        """
        Delete a search entry in Globus Search.
        **Parameters**
        ``path`` (*path string*)
          Path to a local resource on this project
        ``entry_id`` (*string*)
          Name of the entry to delete.
        ``full_subject`` (*bool*)
          Delete all entries in the subject. This is typically equivalent to
          normal deletes if only one entry exists for the given subject.
        **Examples**
        >>> pc.delete_entry('foo/bar.txt')
        >>> pc.delete_entry('moo.txt', entry_id='special_metadata')
        >>> pc.delete_entry('goo.json' full_subject=True)
        """
        index, subject = self.get_index(), self.get_subject_url(path)
        search_cli = self.get_search_client()
        if full_subject:
            return search_cli.delete_subject(index, subject)
        else:
            return search_cli.delete_entry(index, subject, entry_id=entry_id)

    def gather_metadata(self, dataframe, destination, previous_metadata=None,
                        custom_metadata=None, skip_analysis=False, project=None
                        ):
        """Gather metadata on a local file or directory. Returns a new dict
        which combines previous metadata and custom metadata. If skip_analysis
        is True, new analytics won't be attempted and old analytics will be
        carried over.
        **Parameters**
        ``dataframe`` (*path-to-file*)
          Path to a file on the local system
        ``destination`` (*path-string*)
          Path to upload on the remote endpoint, relative to the base path set
          by both the context and project.
        ``previous_metadata`` (*dict*)
          Previous metadata to be combined with the new metadata. This should
          be metadata collected with self.get_search_entry()
        ``custom_metadata`` (*dict*) Custom user provided metadata. Will be
          added to scraped metadata after collection.
        ``skip_analysis`` (*bool*) If true, will attempt to open the file and
          analyze the contents before uploading. This generates extra metadata
          which will be included in search.
        ``project`` (*string*)
          The project to use as the base path. Defaults to current project
        **Examples**
        """
        log.info('Gathering metadata on file {}'.format(dataframe))
        short_path = os.path.join(destination, os.path.basename(dataframe))
        url = self.get_globus_http_url(short_path, project=project)
        new_metadata = search.scrape_metadata(
            dataframe, url, self, skip_analysis=skip_analysis)
        return search.update_metadata(new_metadata, previous_metadata or {},
                                      custom_metadata or {})

    def register(self, dataframe, destination, metadata=None,
                 update=False, dry_run=False, skip_analysis=False):
        """
        Gather metadata on a local search record and register metadata in
        Globus Search. This method assumes either the dataframe already exists
        on a remote endpoint, or that the user will handle uploading the result
        manually. Returns the new metadata for the dataframe.
        This method follows the same behavior as the CLI, raising exceptions
        for several different edge cases, all of which derive from
        pilot.exc.PilotCodeException:
        * pilot.exc.DirectoryDoesNotExist - destination not found
        * pilot.exc.GlobusTransferError - unexpected transfer error
        * pilot.exc.RecordExists - Record exists and update flag was false
        * pilot.exc.NoChangesNeeded - Record was an exact match
        * pilot.exc.DryRun - Everything succeeded, but ingest was aborted

        **Parameters**
        ``dataframe`` (*path-to-file*)
          Path to a file on the local system
        ``destination`` (*path-string*)
          Path to upload on the remote endpoint, relative to the base path set
          by both the context and project.
        ``metadata`` (*dict*)
          A dictionary of metadata to include with this upload. Metadata gets
          registered into search to describe this dataframe
        ``update`` (*bool*) Update an existing dataframe. An exception will be
          raised if this is false and a dataframe exists to prevent an existing
          dataframe from being overwritten.
        ``skip_analysis`` (*bool*) If true, will attempt to open the file and
          analyze the contents before uploading. This generates extra metadata
          which will be included in search.
        ``project`` (*string*)
          The project to use as the base path. Defaults to current project
        **Examples**
        """

        try:
            self.ls(destination)
        except globus_sdk.exc.TransferAPIError as tapie:
            if tapie.code == 'ClientError.NotFound':
                raise exc.DirectoryDoesNotExist(fmt=[destination]) from None
            else:
                raise exc.GlobusTransferError(tapie.message) from None

        if self.get_search_entry(destination):
            raise exc.DestinationIsRecord(fmt=[destination])
        short_path = os.path.join(destination, os.path.basename(dataframe))
        prev_metadata = self.get_search_entry(short_path)
        if prev_metadata and not update and not dry_run:
            raise exc.RecordExists(prev_metadata, fmt=[short_path])
        new_metadata = self.gather_metadata(
            dataframe, destination, previous_metadata=prev_metadata,
            custom_metadata=metadata or {}, skip_analysis=skip_analysis
        )
        stats = search.gather_metadata_stats(new_metadata, prev_metadata or {})
        stats['ingest'] = {}
        if stats['metadata_modified'] is False:
            return stats
        gmeta = search.gen_gmeta(self.get_subject_url(short_path),
                                 [self.get_group()], new_metadata)
        if dry_run:
            return stats
        stats['ingest'] = self.ingest_entry(gmeta)
        return stats

    def upload(self, dataframe, destination, metadata=None, globus=True,
               update=False, dry_run=False, skip_analysis=False, project=None):
        """
        Register a dataframe in Globus Search then upload it to a relative
        project directory on the configured Globus endpoint.
        Raises all of the exceptions that `register` may raise, plus the
        following (All exceptions derive from pilot.exc.PilotCodeException):
        * pilot.exc.NoLocalEndpointSet - No Endpoint configured in profile
        * pilot.exc.NoDestinationProvided - Bad Destination
        **Parameters**
        ``dataframe`` (*path-to-file*)
          Path to a file on the local system
        ``destination`` (*path-string*)
          Path to upload on the remote endpoint, relative to the base path set
          by both the context and project.
        ``metadata`` (*dict*)
          A dictionary of metadata to include with this upload. Metadata gets
          registered into search to describe this dataframe
        ``globus`` (*bool*) Use globus to upload the dataframe. Requires that
          a local endpoint is set in the current user's profile. If false, an
          http put to the configured Globus endpoint will be done instead.
        ``update`` (*bool*) Update an existing dataframe. An exception will be
          raised if this is false and a dataframe exists to prevent an existing
          dataframe from being overwritten.
        ``skip_analysis`` (*bool*) If true, will attempt to open the file and
          analyze the contents before uploading. This generates extra metadata
          which will be included in search.
        ``project`` (*string*)
          The project to use as the base path. Defaults to current project
        **Examples**
        # With context `base_path` set to '/projects/'
        # With project `base_path` set to 'my-project'
        # Uploads to /projects/my-project/foo.txt
        upload('foo.txt', '/')
        # Uploads to /projects/my-project/my/subdir/foo.txt
        upload('local/dir/foo.txt', 'my/subdir/')
        # Raise an exception before uploading to projects/your-project/foo.txt
        upload('foo.txt',
               '/',
               metadata={'my-meta': 'fooiness'},
               globus=True,
               update=True,
               dry_run=True,
               skip_analysis=True,
               project='your-project'
               )
        """
        if globus and not self.profile.load_option('local_endpoint'):
            raise exc.NoLocalEndpointSet()
        if not destination:
            raise exc.NoDestinationProvided(fmt=[self.ls('')])

        stats = self.register(
            dataframe, destination, metadata=metadata, update=update,
            dry_run=dry_run, skip_analysis=skip_analysis
        )
        stats['protocol'] = 'globus' if globus else 'http'
        stats['upload'] = {}
        up = self.upload_globus if globus else self.upload_http
        if not dry_run and stats['files_modified'] is True:
            log.debug('Uploading using {}'.format(up))
            stats['upload'] = up(dataframe, destination, project=project)
        return stats

    def upload_http(self, dataframe, destination, project=None):
        """Upload to the configured HTTP endpoint for this context/project.
        Executes a simple upload without any metadata or checking the
        destination for existing files. Overwrites any existing dataframe.
        The project must have a configured http endpoint on petrel
        """
        return_values = []
        for local_path, remote_path in search.get_subdir_paths(dataframe):
            path = self.get_path(os.path.join(destination, remote_path))
            rv = self.get_http_client(project).put(path, filename=local_path)
            return_values.append(rv)
        return return_values

    def upload_globus(self, dataframe, destination, project=None,
                      globus_args=None):
        """Upload a dataframe to a project using a Globus Transfer. A local
        endpoint must be configured.
        ** parameters **
        ``dataframe`` (*path-to-file*)
          Path to a file on the local system
        ``destination`` (*path-string*)
          Path to upload on the remote endpoint, relative to the base path set
          by both the context and project.
        ``project`` (*string*)
          The project to use as the base path. Defaults to current project
        ``globus_args`` (*dict*)
          Other arguments to pass to Globus Transfer. Overwrites any defaults.
          See ``transfer_file`` for more info.
          https://globus-sdk-python.readthedocs.io/en/stable/clients/transfer/#globus_sdk.TransferClient.submit_transfer  # noqa
        """
        log.info('Uploading (Globus) {} to {}'.format(dataframe, destination))
        paths = []
        for file_path, remote_short_path in search.get_subdir_paths(dataframe):
            rel_dest = os.path.join(destination, remote_short_path)
            paths.append((file_path, self.get_path(rel_dest, project=project)))
        result = self.transfer_files(
            self.profile.load_option('local_endpoint'),
            self.get_endpoint(),
            paths,
            **(globus_args or {})
        )
        tl = transfer_log.TransferLog()
        dest = os.path.join(destination, os.path.basename(dataframe))
        tl.add_log(result, dest)
        return result

    def transfer_file(self, src_ep, dest_ep, src_path, dest_path,
                      globus_args=None):
        """Low level utility for transferring a single file using Globus.
        Does not account for context/project basepaths.
        ** parameters **
        ``src_ep`` (*uuid*)
          Source Globus Endpoint
        ``destination`` (*uuid*)
          Destination Globus Endpoint
        ``src_path`` (*path-string*)
          Source path to file to-be-transferred
        ``dest_path`` (*path-string*)
          Destination path to transfer file.
        ``globus_args`` (*dict*)
          Globus Transfer options. Defaults include:
          {
            'label': 'MyApp Transfer',
            'notify_on_succeeded': False,
            'sync_level': 'checksum',
            'encrypt_data': True,
          }
          See more options at:
          https://globus-sdk-python.readthedocs.io/en/stable/clients/transfer/#globus_sdk.TransferClient.submit_transfer  # noqa
        """
        return self.transfer_files(src_ep, dest_ep, [(src_path, dest_path)],
                                   globus_args=globus_args)

    def transfer_files(self, src_ep, dest_ep, paths, globus_args=None):
        """Low level utility for transferring a multiple files using Globus.
        Does not account for context/project basepaths. All files are expected
        to originate from the same src_ep, to be transferred to the same
        destination.
        ** parameters **
        ``src_ep`` (*uuid*)
          Source Globus Endpoint
        ``destination`` (*uuid*)
          Destination Globus Endpoint
        ``paths`` (*list of two item tuples*)
          A list of items to transfer. Each item must be a tuple with two
          entries, the first the path to the source file, and the second
          the path of the destination. For example:
          [('/users/foo/bar.txt', '~/bar.txt'), ('a.json', '~/a.json')]
        ``globus_args`` (*dict*)
          Globus Transfer options. Defaults include:
          {
            'label': 'MyApp Transfer',
            'notify_on_succeeded': False,
            'sync_level': 'checksum',
            'encrypt_data': True,
          }
          See more options at:
          https://globus-sdk-python.readthedocs.io/en/stable/clients/transfer/#globus_sdk.TransferClient.submit_transfer  # noqa
        """
        tc = self.get_transfer_client()
        g_defaults = {
            'label': '{} Transfer'.format(self.context.get_value('app_name')),
            'notify_on_succeeded': False,
            'sync_level': 'checksum',
            'encrypt_data': True,
        }
        g_defaults.update(globus_args or {})
        tdata = globus_sdk.TransferData(tc, src_ep, dest_ep, **g_defaults)
        for src_path, dest_path in paths:
            log.debug('Transferring {} to {}'.format(src_path, dest_path))
            tdata.add_item(src_path, dest_path)
        transfer_result = tc.submit_transfer(tdata)
        log.debug('Submitted Transfer')
        return transfer_result

    def download(self, path, project=None, relative=True, globus=False):
        downloader = self.download_globus if globus else self.download_http
        return downloader(path, project=project, relative=relative)

    def download_parts(self, path, project=None, relative=True, range=None):
        """Download a file in parts over HTTP and yield the number of bytes
        written for each part. Yields a generator for each part."""
        fname = os.path.basename(path)
        url = self.get_path(path, project=project, relative=relative)
        http_client = self.get_http_client(project=project)
        log.debug(f'Fetching item {url}')
        response = http_client.get(url, range=range)
        with open(fname, 'wb') as fh:
            for part in response.iter_content:
                yield fh.write(part)
        log.debug('Fetch Successful')
        return 0

    def download_http(self, path, project=None, relative=True, range=None):
        """
        Download a file to the local system using HTTPS to the local filesystem
        using the 'path' basename. Returns the total number of bytes written
        to the filesystem.
        **Parameters**
        ``path`` (*path string*)
          Path to a local resource on this project
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        ``relative`` (*bool*)
          If True, prepends the path to the project. If False,
          does not prepend path but ensures it's in the project's directory
        ``range`` (*string*)
          Specify a byte range to fetch from the file. Range must be in the
          format: "0-10,15-20" where each number is a byte.
        **Examples**
        >>> pc.download('foo.txt')
        >>> pc.download('foo.txt', project='bar', range='0-100')
        >>> pc.download('bar/moo.txt', range='0-100,150-200')
        """
        return sum(self.download_parts(path, project, relative, range))

    def download_globus(self, path, globus_args=None):
        result = self.transfer_file(
            self.profile.load_option('local_endpoint'),
            self.get_endpoint(),
            path,
            os.path.basename(path),
            **globus_args)
        return result

    def delete(self, path, project=None, relative=True, recursive=False):
        """
        Delete a file on the remote endpoint for the given project.
        to the filesystem.
        **Parameters**
        ``path`` (*path string*)
          Path to a local resource on this project
        ``project`` (*string*)
          The project to fetch info for. Defaults to current project
        ``relative`` (*bool*)
          If True, prepends the path to the project. If False,
          does not prepend path but ensures it's in the project's directory
        ``recursive`` (*bool*)
          If the 'path' refers to a folder, delete everything inside it
        **Examples**
        >>> pc.delete('foo.txt')
        >>> pc.delete('bar', recursive=True)
        """
        tc = self.get_transfer_client()
        endpoint = self.get_endpoint(project)
        full_path = self.get_path(path, project=project, relative=relative)
        app_name = self.context.get_value('app_name')
        ddata = globus_sdk.DeleteData(
            tc, endpoint, recursive=recursive, notify_on_succeeded=False,
            label='File Deletion with {}'.format(app_name))
        ddata.add_item(full_path)
        delete_result = tc.submit_delete(ddata)
        log.debug(delete_result)
