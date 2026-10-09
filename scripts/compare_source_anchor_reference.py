"""Secondary comparison with the previously trained, non-reset pixel control.

No new inference, sample selection, or metric definition: match the exact saved
conditions and reapply the same screen. This is not the paired anchor ablation.
"""

import argparse
import shutil
from pathlib import Path

from campaign_protocol import read, sha, write
from diagnose_learnability import verify
from evaluate_procedural_structure import render, summarize


def compare(root, step):
    verify(root)
    plan = read(root / 'plan.json')
    parent = Path(plan['parent'])
    verify(parent)
    assert plan['data'] == read(parent / 'plan.json')['data']
    assert sha(root / 'inputs.npz') == sha(parent / 'inputs.npz')
    out = root / 'warm-reference'
    out.mkdir(exist_ok=True)
    write(out / 'plan.json', dict(plan, comparison_methods=['pixel', 'source'],
        secondary_reference=True, paired_initialization=False))
    for split in ('train', 'probe'):
        assert sha(root / f'{split}.json') == sha(parent / f'{split}.json')
        shutil.copyfile(root / f'{split}.json', out / f'{split}.json')
    bindings = {}
    for category in ('evaluation', 'broad'):
        path = f'{category}/step-{step}/metrics.json'
        control = [r for r in read(parent / path)['rows'] if r['method'] == 'pixel']
        candidate = [r for r in read(root / path)['rows'] if r['method'] == 'source']
        fields = ('split','case','draw') if category == 'evaluation' else (
            'split','partial','strength','kind','variant','generation_key','family','pair')
        assert len(control) == len(candidate) == (64 if category == 'evaluation' else 456)
        for a,b in zip(control,candidate,strict=True):
            assert all(a.get(k) == b.get(k) for k in fields)
        write(out / path, dict(step=step, rows=control+candidate, adoption=False))
        bindings[str(parent / path)] = sha(parent / path)
        bindings[str(root / path)] = sha(root / path)
    for split in ('train','probe'):
        for method,origin in [('pixel',parent),('source',root)]:
            path=f'broad/step-{step}/{method}-{split}.npz'
            target=out/path
            if not target.exists():shutil.copyfile(origin/path,target)
            assert sha(target)==sha(origin/path)
            bindings[str(origin/path)]=sha(origin/path)
    summarize(out,step)
    render(out,step)
    write(out / f'binding-{step}.json',dict(files=bindings,script_sha256=sha(Path(__file__)),
        conditions_identical=True,comparison='Source anchor versus non-reset warm pixel control',
        paired_initialization=False,adoption=False))
    summary=read(out / f'broad/step-{step}/summary.json')
    print(dict(step=step,secondary_failures=len(summary['failures']),halo=summary['halo_checks']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--step',type=int,choices=[1000,4000],required=True)
    args=parser.parse_args()
    compare(args.root.resolve(),args.step)
