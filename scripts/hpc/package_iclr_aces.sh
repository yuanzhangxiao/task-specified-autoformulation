#!/usr/bin/env bash
# Build a transferable pinned checkout, the public inputs and the offline vocabulary.
set -euo pipefail
cd "$(dirname "$0")/../.."
repo=$PWD
commit=$(git rev-parse HEAD)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
case "${1:-default}" in
    default) name=iclr-nonbaseline-aces ;;
    full-120b) name="iclr-anchors-full-120b-${commit:0:7}" ;;
    ordered-120b) name="iclr-ordered-full-120b-${commit:0:7}" ;;
    *) echo 'Use package_iclr_aces.sh [full-120b|ordered-120b]' >&2; exit 2 ;;
esac
release="$stage/$name"
code="$release/code"
mkdir -p "$code" "$release/inputs" "$code/assets/harmony"
git archive "$commit" src scripts configs schemas pyproject.toml docs/PROJECT_CONTEXT.md \
    docs/ICLR_2027_ACES_EXECUTION.md docs/ICLR_2027_NONBASELINE_RUNBOOK.md docs/ICLR_2027_CONTROLLED_SEARCH.md \
    docs/ICLR_2027_REFERENCE_AND_INTERVENTIONS.md docs/ICLR_2027_STRESS_STUDIES.md |
    tar -xf - -C "$code"
test -f "$code/scripts/iclr_aces_pilot.py"
printf '%s\n' "$commit" >"$code/SOURCE_COMMIT"
AF_VOCAB_DIR=${AF_VOCAB_DIR:-$repo/output/insulin-revision-delivery-20261105/bundle-smoke/phase-c-insulin-revision-v1/assets/harmony}
python3 "$code/scripts/harmony_vocab.py" verify --directory "$AF_VOCAB_DIR" >/dev/null
cp "$AF_VOCAB_DIR/o200k_base.tiktoken" "$code/assets/harmony/"
if [[ $name == iclr-nonbaseline-aces ]]; then
python3 - "$repo/output" "$release/inputs" <<'PY'
import hashlib
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile
original, dest = map(Path, sys.argv[1:])
public = original / 'iclr-nonbaseline-public.zip'
analyst = original / 'iclr-nonbaseline-evidence.zip'
if hashlib.sha256(public.read_bytes()).hexdigest() != '0ca53ebd526679cbce964ec9868c1e059f745512211a829000d39ca0b277eef6':
    raise ValueError('public discovery release differs')
if hashlib.sha256(analyst.read_bytes()).hexdigest() != '7e1a64fe6ef8e9c93f3521f8e513edc2aac14f2b8144bcb081f2d53dca315663':
    raise ValueError('source of corrected public CSTR files differs')
shutil.copyfile(public, dest / public.name)
with ZipFile(analyst) as z:
    for name in ('manifest.json','train.csv','validation.csv','proposer_prompt.txt'):
        z.extract('corrected-easy/'+name, dest)
    for lane, folder in (('anchors','anchor-search-plan'), ('stress','stress-search-plan')):
        (dest / (lane + '-design.json')).write_bytes(z.read('plans/' + folder + '/plan.json'))
PY
fi
if [[ ${1:-default} == ordered-120b ]]; then
python3 - "$repo" "$code" "$release/inputs/public" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path
repo, code, dest = map(Path, sys.argv[1:])
roster = json.loads((code / 'configs/iclr_full_120b_ordered_v1.json').read_text())
for case in roster['cases']:
    if case['reference'] != 'public':
        continue
    root = repo / 'artifacts/phase-c-development-final-v2/public' / case['cell']
    for name, expected in case['public_files'].items():
        source = root / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'public file changed: {case["cell"]}/{name}')
        target = dest / case['cell'] / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
PY
fi
python3 - "$release" "$stage/delivery.tar.gz" <<'PY'
import hashlib
import sys
import tarfile
from pathlib import Path
root = Path(sys.argv[1])
lines = [hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(root).as_posix()
         for p in sorted(root.rglob('*')) if p.is_file()]
(root / 'SHA256SUMS').write_text('\n'.join(lines)+'\n')
# Package ordinary file data and Unix modes without the build host's extended attributes.
with tarfile.open(sys.argv[2], "w:gz", format=tarfile.GNU_FORMAT) as archive:
    archive.add(root, arcname=root.name)
PY
mv "$stage/delivery.tar.gz" "$repo/output/$name.tar.gz"
printf 'Source %s\n%s\n' "$commit" "$repo/output/$name.tar.gz"
