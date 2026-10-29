"""Download checksum-verified public LoED recordings (CC BY 4.0).

The default is the publisher's six-day sample, not a performance-selected sample.
No downloaded Python or notebook code is executed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/public/loed'
API = 'https://zenodo.org/api/records/4121430'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    metadata_path = OUT/'zenodo_metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    else:
        with urllib.request.urlopen(API, timeout=30) as response:
            metadata = json.load(response)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    chosen = 'LoED_LoRaWAN_at_edge_dataset'+('' if args.full else '-SAMPLE')+'.zip'
    records = []
    for entry in sorted(metadata['files'], key=lambda e: e['key'] != chosen):
        if entry['key'] not in [chosen, 'README.md', 'Laksh_Bhatia_LoED_LoRaWAN_at_the_edge.pdf']:
            continue
        path = OUT/entry['key']
        expected = entry['checksum'].split(':', 1)[1]
        if not path.exists():
            part = path.with_suffix(path.suffix+'.part')
            urls = [f'https://zenodo.org/records/4121430/files/{entry["key"]}?download=1',
                    entry['links']['self']]
            errors = []
            for url in urls:
                print('Fetching', url, flush=True)
                request = urllib.request.Request(url, headers={'User-Agent': 'RF-event-association-research/1.0'})
                try:
                    with urllib.request.urlopen(request, timeout=25) as response, part.open('wb') as stream:
                        total = 0
                        while chunk := response.read(1024*1024):
                            total += len(chunk)
                            if total > entry['size']:
                                raise ValueError('Download larger than publisher metadata')
                            stream.write(chunk)
                    break
                except Exception as exc:
                    errors.append(str(exc))
                    print(type(exc).__name__, str(exc), flush=True)
            else:
                if entry['key'] == chosen:
                    raise RuntimeError('Both official download routes failed: '+repr(errors))
                records.append(dict(file=entry['key'], status='OPTIONAL_DOWNLOAD_FAILED', errors=errors))
                continue
            if total != entry['size']:
                raise ValueError('Incomplete download')
            if hashlib.md5(part.read_bytes()).hexdigest() != expected:
                raise ValueError('Publisher MD5 mismatch')
            part.replace(path)
        payload = path.read_bytes()
        assert len(payload) == entry['size']
        assert hashlib.md5(payload).hexdigest() == expected
        records.append(dict(file=entry['key'], bytes=len(payload), md5=expected,
                            sha256=hashlib.sha256(payload).hexdigest(), url=entry['links']['self']))
        print('Verified', entry['key'], len(payload), flush=True)
    archive = OUT/chosen
    with zipfile.ZipFile(archive) as source:
        members = [dict(name=x.filename, bytes=x.file_size) for x in source.infolist()]
    report = dict(dataset_doi='10.5281/zenodo.4121430', license='CC-BY-4.0',
                  source=API, selection='publisher six-day sample' if not args.full else 'full archive',
                  status='VERIFIED_DOWNLOAD', files=records, archive_members=members)
    (OUT/'download_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(members, indent=2))


if __name__ == '__main__':
    main()
