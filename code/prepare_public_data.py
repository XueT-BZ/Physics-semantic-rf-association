"""Fetch and inventory the public Sorbonne multi-sniffer WiFi archive.

Standard library only. This is a provenance/availability tool, not a benchmark.
It never emits association accuracy from dataset descriptions. A previously
downloaded original ZIP can be supplied with --archive PATH.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOI = '10.57745/HAOPHF'
FILE_DOI = '10.57745/EVKTZB'
URL = ('https://entrepot.recherche.data.gouv.fr/api/access/datafile/'
       ':persistentId?persistentId=doi:10.57745/EVKTZB')
EXPECTED_MD5 = '3ce2868b97eb1a8750233e67fb3cfbe3'
NAME = '220211012-SU-Outdoors-Campus.zip'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Original ZIP downloaded from the publisher')
    parser.add_argument('--timeout', type=int, default=25)
    args = parser.parse_args()
    out = ROOT/'data/public/sorbonne_wifi'
    out.mkdir(parents=True, exist_ok=True)
    audit = ROOT/'audit/revision_20261020/public_data_status.json'
    audit.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'dataset_doi': DOI, 'file_doi': FILE_DOI,
        'landing_page': 'https://doi.org/'+DOI, 'download_url': URL,
        'license_per_publisher': 'Etalab Open License 2.0',
        'expected_archive_md5': EXPECTED_MD5,
        'metadata_checked_date': '2026-10-20',
        'publisher_description': 'Single source; ten sniffers; multiple distances; raw and synchronized packet traces.',
        'archive_downloaded_and_verified': False,
        'rf_association_experiment_executed': False,
        'reported_performance_metrics': None,
    }
    path = args.archive if args.archive is not None else out/NAME
    try:
        if not path.exists():
            if args.archive is not None:
                raise FileNotFoundError(path)
            req = urllib.request.Request(URL, headers={'User-Agent': 'RF-association-reproducibility/1.0'})
            # Bound both download and archive expansion; do not disable TLS checks.
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                blob = response.read(20_000_001)
            if len(blob) > 20_000_000:
                raise ValueError('Archive exceeds the 20 MB download limit')
        else:
            blob = path.read_bytes()
        digest = hashlib.md5(blob).hexdigest()
        if digest != EXPECTED_MD5:
            raise ValueError(f'Publisher checksum mismatch: {digest}')
        # Keep the original bytes only after validating the publisher checksum.
        if not path.exists():
            path.write_bytes(blob)
        report.update(archive_downloaded_and_verified=True,
                      archive_sha256=hashlib.sha256(blob).hexdigest(), archive_bytes=len(blob))
        records = []
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            if sum(i.file_size for i in archive.infolist()) > 100_000_000:
                raise ValueError('Expanded archive exceeds the 100 MB inventory limit')
            for item in archive.infolist():
                parts = PurePosixPath(item.filename.replace('\\', '/'))
                if parts.is_absolute() or '..' in parts.parts or ':' in item.filename:
                    raise ValueError(f'Unsafe archive member: {item.filename}')
                row = {'member': item.filename, 'bytes': item.file_size}
                if item.filename.lower().endswith('.csv'):
                    text = archive.read(item).decode('utf-8-sig')
                    sample = text[:8192]
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
                    except csv.Error:
                        dialect = csv.excel
                    reader = csv.reader(io.StringIO(text), dialect)
                    row['header'] = next(reader, [])
                    row['data_rows'] = sum(bool(r) for r in reader)
                records.append(row)
        (out/'archive_inventory.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        report.update(status='ARCHIVE_VERIFIED_AWAITING_BENCHMARK_VALIDATION',
                      archive_members=len(records),
                      next_step='Inspect actual schemas, repeated sequence IDs/retries and upstream synchronization before forming event truth or train/test splits.')
    except Exception as exc:
        report.update(status='UNAVAILABLE_NO_REAL_DATA_RESULTS',
                      error_type=type(exc).__name__, error=str(exc),
                      next_step='Retry the publisher download or provide its original ZIP with --archive; validate packet identities before evaluating association.')
        audit.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    audit.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
