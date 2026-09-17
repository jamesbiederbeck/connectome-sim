"""Fetch a dataset's raw upstream files listed in doom/datasets.json into connectome_data/."""
import argparse,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def download(dataset='malecns_v1'):
    registry=json.loads((ROOT/'doom/datasets.json').read_text())
    files=registry['datasets'][dataset]['files']
    target=ROOT/'connectome_data'/dataset
    target.mkdir(parents=True,exist_ok=True)
    lock_path=ROOT/'data-provenance'/dataset/'source.lock.json'
    lock=json.loads(lock_path.read_text()) if lock_path.exists() else {}
    for name,url in files.items():
        dest=target/name
        print(f'Fetching {name}...')
        urllib.request.urlretrieve(url,dest)
        if name in lock:
            info=lock[name]
            assert dest.stat().st_size==info['bytes'],f'{name}: size mismatch'
            assert digest(dest)==info['sha256'],f'{name}: sha256 mismatch'
            print(f'{name} verified against source.lock.json')
        else:
            print(f'{name}: no source.lock.json entry to verify against')
    return target

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='malecns_v1')
    download(parser.parse_args().dataset)
