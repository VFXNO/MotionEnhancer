"""Read-only checkpoint fingerprint worker; no PyTorch, deserialization or GPU work."""
import argparse
import hashlib
import json
import os
import stat


def _identity(info):
    # Windows Python versions can report different ctime semantics through stat
    # and fstat. File identity, size and modification time agree across both;
    # SHA256 is independently checked again by the actual training process.
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def file_identity(path):
    """Cheap change detection only; the training process still verifies SHA256."""
    return _identity(os.stat(path))


def fingerprint(path):
    """Stream the selected file, rejecting concurrent edits or path replacement."""
    path = os.path.abspath(os.fspath(path))
    info = os.stat(path)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Choose an existing checkpoint file, not a folder or device')
    before = _identity(info)
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        if _identity(os.fstat(handle.fileno())) != before:
            raise ValueError('Checkpoint changed before SHA256 calculation; try again when saving has finished')
        while block := handle.read(1024 * 1024):
            digest.update(block)
        if _identity(os.fstat(handle.fileno())) != before:
            raise ValueError('Checkpoint changed during SHA256 calculation; try again when saving has finished')
    if file_identity(path) != before:
        raise ValueError('Checkpoint changed during SHA256 calculation; select it again')
    return {'path': path, 'sha256': digest.hexdigest(), 'identity': before}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', required=True)
    parser.add_argument('--request', required=True)
    args = parser.parse_args(argv)
    try:
        event = {'event': 'checkpoint_hash', 'request': args.request, **fingerprint(args.path)}
        code = 0
    except (OSError, ValueError) as error:
        event = {'event': 'checkpoint_hash_error', 'request': args.request, 'message': str(error)}
        code = 1
    print(json.dumps(event), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
