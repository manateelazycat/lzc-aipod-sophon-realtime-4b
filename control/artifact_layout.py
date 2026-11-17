"""Preserve public snapshot paths over Host 1.x's flat artifact transport."""
import hashlib
import re


def transport_name(path):
    if (not re.fullmatch(r'[A-Za-z0-9_./-]{1,1024}', path)
            or any(part in ('', '.', '..') for part in path.split('/'))):
        raise ValueError('Invalid model artifact path')
    return path if '/' not in path else 'nested-' + hashlib.sha256(path.encode()).hexdigest()


def model_mounts(root, entry):
    files = entry['files']
    names = [transport_name(item['path']) for item in files]
    if len(set(names)) != len(names):
        raise ValueError('Model transport names collide')
    if all('/' not in item['path'] for item in files):
        return [root + ':/model-root:ro']
    role = entry['role']
    return [f"{root}/{role}/{name}:/model-root/{role}/{item['path']}:ro"
            for item, name in zip(files, names)]
