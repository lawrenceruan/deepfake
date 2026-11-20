"""Extract a new verified raw copy without changing the existing Kaggle archive."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from deepfake.preprocess import fingerprint
from deepfake.prepare_full_segments import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--expected',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();archive=a.archive.resolve()
    expected=json.loads(a.expected.read_text())
    if archive.is_relative_to(out) or out.is_relative_to(archive.parent):
        raise ValueError('New raw copy must be separate from the existing raw/archive folder')
    config=dict(archive=str(archive),expected_inventory=fingerprint(a.expected))
    if out.exists():
        if json.loads((out/'extraction_config.json').read_text())!=config:
            raise ValueError('Existing output belongs to a different extraction')
    else:
        out.mkdir(parents=True)
        write_json(out/'extraction_config.json',config)
    before=(archive.stat().st_size,archive.stat().st_mtime_ns)
    with zipfile.ZipFile(archive) as z:
        for i,(rel,signature) in enumerate(expected.items(),1):
            target=out/rel
            if not target.resolve().is_relative_to(out) or len(Path(rel).parts)!=2:
                raise ValueError('Unsafe member path')
            if target.exists():
                if fingerprint(target)!=signature:
                    raise ValueError(f'Existing raw file does not match: {rel}')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_suffix(target.suffix+'.partial')
                with z.open('FaceForensics++_C23/'+rel) as source,temporary.open('wb') as dest:
                    shutil.copyfileobj(source,dest)
                if fingerprint(temporary)!=signature:
                    raise ValueError(f'Archive content checksum mismatch: {rel}')
                temporary.replace(target)
            if i%250==0:print(json.dumps(dict(extracted_verified=i,total=len(expected))),flush=True)
    assert before==(archive.stat().st_size,archive.stat().st_mtime_ns)
    write_json(out/'extraction_complete.json',dict(files=len(expected),all_sha256_verified=True,
        archive_unchanged=True,archive=str(archive)))


if __name__=='__main__':main()
