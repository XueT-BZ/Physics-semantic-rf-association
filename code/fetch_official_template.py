#!/usr/bin/env python3
"""Optional, user-run publisher-template download. Requires internet access.

  python code/fetch_official_template.py

Copies the official legacy pdfTeX Definitions/ directory from MDPI's repository.
Existing files with the same name are not overwritten. The official template was
retrieved and its manuscript build verified on 2026-10-20. A bundled mdpi.cls
therefore needs no additional download.
"""
from pathlib import Path, PurePosixPath
from io import BytesIO
from urllib.request import Request, urlopen
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
URL='https://codeload.github.com/MDPI-AG/luatex-template/zip/refs/heads/main'
PREFIX='luatex-template-main/legacy-pdftex-version/Definitions/'

def main():
    request=Request(URL,headers={'User-Agent':'RF-Manuscript-Template-Installer/1.0'})
    with urlopen(request,timeout=60) as response:
        payload=response.read(50_000_001)
    if len(payload)>50_000_000:
        raise RuntimeError('Archive exceeds the expected size limit.')
    archive=zipfile.ZipFile(BytesIO(payload))
    selected=[]
    for entry in archive.infolist():
        if not entry.filename.startswith(PREFIX) or entry.is_dir():continue
        relative=PurePosixPath(entry.filename[len(PREFIX):])
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unexpected archive path.')
        destination=ROOT/'Definitions'/Path(*relative.parts)
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite {destination}; move the existing file aside first.')
        selected.append((entry,destination))
    if not any(dst.name=='mdpi.cls' for _,dst in selected):
        raise RuntimeError('The publisher repository layout changed; inspect it manually.')
    for entry,destination in selected:
        destination.parent.mkdir(parents=True,exist_ok=True)
        with archive.open(entry) as source,destination.open('wb') as output:
            shutil.copyfileobj(source,output)
    print('Installed official Definitions/. main.tex will now prefer mdpi.cls.')
    print('Recompile and inspect all pages after changing the template version.')
if __name__=='__main__':main()
