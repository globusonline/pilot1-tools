from pilot.client import PilotClient
from pilot.search import gen_gmeta

PROJECT = 'tomogan'
pc = PilotClient()
pc.project.current = PROJECT

tomogan_entries = pc.list_entries('', relative=False)

for entry in tomogan_entries:
    entry['content'][0]['dc']['creators'] = [
            {'creatorName': 'Zhengchun Liu'},
            {'creatorName': 'Tekin Bicer'},
            {'creatorName': 'Raj Kettimuthu'},
            {'creatorName': 'Ian Foster'},
        ]
    entry['content'][0]['dc']['publisher'] = 'Argonne'

    is_simulation = entry['content'][0]['project_metadata']['sample'] == 'simulation'
    exalearn_group = '6a60cf30-676a-11e9-8b34-0e4a32f5e3b8'
    visible_to = [exalearn_group] if is_simulation else ['public']

    gmeta = gen_gmeta(entry['subject'], visible_to, entry['content'][0])
    pc.ingest_entry(gmeta)
