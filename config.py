import datetime
import os.path
from os.path import join
import json
from types import SimpleNamespace


def dump_dict(data, *, dumpfile=None, append=False, key='', ordered=True, indent=4, print_stdout=True):
    """
    Write a dict to stdout and/or a file.

    args:
        data            dict to write

    keyword-only args:
        dumpfile        if set, write to this file
        append          append to dumpfile, otherwise overwrite (default:
                        False)
        key             string to print as first line (default: "", no
                            printing is performed)
        ordered         order dict keys (default: True)
        indent          number of spaces for indentation (default: 4)
        print_stdout    print to stdout (default: True)
    """

    if type(data) is SimpleNamespace:
        data = data.__dict__

    s_stdout = json.dumps(data, indent=indent, sort_keys=ordered)

    if len(key) > 0:
        s_stdout = key + ":\n" + s_stdout
        s_file = json.dumps((key, data), indent=indent, sort_keys=ordered)
    else:
        s_file = s_stdout

    mode = 'a' if append else 'w'

    if dumpfile is not None:
        with open(dumpfile, mode) as f:
            f.write(s_file)
            f.write('\n')

    if print_stdout:
        print(s_stdout)


def write(name, *, config_c, config_v, variant, recall_cfg, dirname='cfg', overwrite=False):
    fn = join(dirname, name+'.json')

    if not overwrite and os.path.exists(fn):
        raise IOError('config exists, cannot overwrite: '+fn)

    now = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    meta = {'name': name, 'written_at': now}

    data = dict(config_c=config_c, config_v=config_v, variant=variant, recall_cfg=recall_cfg, __meta__=meta)

    dump_dict(data, dumpfile=fn, print_stdout=False)


def load(name, dirname='cfg'):
    fn = join(dirname, name+'.json')

    if not os.path.exists(fn):
        raise IOError('cannot find config: '+fn)

    with open(fn, 'r') as f:
        data = json.load(f)

    assert set(data.keys()) == {'config_c', 'config_v', 'variant', 'recall_cfg', '__meta__'}

    return data['config_c'], data['config_v'], data['variant'], data['recall_cfg']
