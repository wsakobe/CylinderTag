"""Compare detector output availability on unannotated canonical sample videos.

Does not equate a nonempty detection with a correct recognition. Inputs remain
unchanged; all methods receive the same full dictionary and frame settings.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('videos', nargs='+', type=Path)
    p.add_argument('--dictionary', type=Path, default=Path('CTag_2f12c.marker'))
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--bugfix', type=Path, required=True)
    p.add_argument('--improved', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('out/sample_benchmark'))
    p.add_argument('--summarize-existing', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    result = {'metric': 'Detection availability, not ground-truth recognition accuracy',
              'settings': {'scale': 1, 'threshold_window': 5, 'refine': True,
                           'refinement_radius': 5, 'roi': None, 'tracking': False,
                           'dictionary_transform': None}, 'videos': []}
    for video in a.videos:
        item = {'path': str(video), 'sha256': sha256(video),
                'dictionary_sha256': sha256(a.dictionary),
                'ground_truth': None, 'runs': []}
        sequences = []
        for label, exe in [('original', a.original), ('bugfix', a.bugfix), ('robust', a.improved)]:
            prefix = a.output / (video.stem + '_' + label)
            command = [str(exe.resolve()), str(video.resolve()), str(a.dictionary.resolve()),
                       str(prefix.resolve()), '1', '5', '1']
            if not a.summarize_existing:
                with prefix.with_suffix('.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = [json.loads(x) for x in prefix.with_suffix('.jsonl').read_text().splitlines()]
            assert rows and [x['frame'] for x in rows] == list(range(len(rows)))
            ids = [{m['id'] for m in x['markers']} for x in rows]
            sequences.append(ids)
            detected = sum(bool(x) for x in ids)
            run = {'label': label, 'executable_sha256': sha256(exe),
                   'reproduction_command': command, 'outputs_reused': a.summarize_existing,
                   'frames': len(rows), 'detected_frames': detected,
                   'detection_percent': 100 * detected / len(rows),
                   'ids_by_frame_count': dict(sorted(Counter(i for x in ids for i in x).items())),
                   'missing_frames': [i for i, x in enumerate(ids) if not x]}
            if not a.summarize_existing:
                run['executed_command'] = command
            item['runs'].append(run)
        assert len({len(x) for x in sequences}) == 1, 'Different frame counts'
        item['robust_misses_with_original_detection'] = [i for i, (old, new) in
            enumerate(zip(sequences[0], sequences[2])) if old and not new]
        result['videos'].append(item)
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (a.output / 'summary.csv').open('w') as stream:
        w = csv.writer(stream)
        w.writerow(['video', 'method', 'frames', 'detected_frames', 'detection_percent', 'id_frame_counts'])
        for item in result['videos']:
            for run in item['runs']:
                w.writerow([item['path'], run['label'], run['frames'], run['detected_frames'],
                            run['detection_percent'], json.dumps(run['ids_by_frame_count'])])
                print(item['path'], run['label'], f"{run['detected_frames']}/{run['frames']}",
                      f"{run['detection_percent']:.2f}%", run['ids_by_frame_count'])


if __name__ == '__main__':
    main()
