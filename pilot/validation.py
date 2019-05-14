import os
import json
import jsonschema

BASE_DIR = os.path.dirname(__file__)
BASE_SCHEMA_DIR = os.path.join(BASE_DIR, 'schemas')


def get_schemas():
    schemas = {}
    files = [f for f in os.listdir(BASE_SCHEMA_DIR)
             if os.path.splitext(f)[1] == '.json']

    for f in files:
        fname = os.path.join(BASE_SCHEMA_DIR, f)
        with open(fname) as fh:
            sname, _ = os.path.splitext(f)
            try:
                schemas[sname] = json.load(fh)
            except json.decoder.JSONDecodeError:
                raise Exception('Error loading {}'.format(fname))
    return schemas


def validate_dataset(dataset):
    validate_json('dataset', dataset)


def validate_user_provided_metadata(metadata):
    validate_json('user_provided_metadata', metadata)


def validate_json(name, json):
    schema = get_schemas()[name]
    resolver = jsonschema.RefResolver(
        base_uri="file://{}/{}".format(BASE_SCHEMA_DIR, name),
        referrer=name
    )
    # setting the resolver for datacite causes problems. It seems to resolve
    # fine when no custom resolver is used.
    if name == 'dc':
        resolver = None
    jsonschema.validate(schema=schema, resolver=resolver, instance=json)
