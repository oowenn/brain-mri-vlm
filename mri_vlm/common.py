"""Small helpers shared by every stage: repository paths, hashing, CSV reading and exact comparison."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREPARED = REPO / 'data/prepared-v3-conservative-patients'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def assert_matches(actual, expected, path='metrics'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f'{path}: keys differ')
        for key in expected:
            assert_matches(actual[key], expected[key], f'{path}/{key}')
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f'{path}: {actual} != {expected}')
    elif actual != expected:
        raise ValueError(f'{path}: values differ')


def passed_tests(log_path):
    """Unit-test count from a run's tests.txt, or None for runs launched without a test step.

    Archived runs all include tests.txt. The CPU tests are kept out of the published
    repository, so launchers no longer run them; a log that is present must still show a pass.
    """
    if not log_path.exists():
        return None
    log = log_path.read_text()
    count = re.search(r'Ran (\d+) tests', log)
    if not count or not log.rstrip().endswith('OK'):
        raise ValueError(f'Unit tests did not pass: {log_path}')
    return int(count.group(1))
