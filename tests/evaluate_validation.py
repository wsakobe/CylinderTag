"""Evaluate every labeled synthetic frame; print counts without hiding misses."""
import json
from pathlib import Path
import sys

labels = json.loads(Path(sys.argv[1]).read_text())
rows = [json.loads(line) for line in Path(sys.argv[2]).read_text().splitlines()]
assert len(labels) == len(rows), 'Frame count mismatch'
result = dict(positive_frames=0, correct_id_frames=0, wrong_id_frames=0,
              negative_frames=0, negative_detection_frames=0, mirror_flag_errors=0)
for index, (label, row) in enumerate(zip(labels, rows)):
    assert label['frame'] == row['frame'] == index
    ids = {marker['id'] for marker in row['markers']}
    if label['id'] is None:
        result['negative_frames'] += 1
        result['negative_detection_frames'] += bool(ids)
    else:
        result['positive_frames'] += 1
        result['correct_id_frames'] += label['id'] in ids
        result['wrong_id_frames'] += bool(ids-{label['id']})
        result['mirror_flag_errors'] += any(marker['id']==label['id'] and
            'mirrored' in marker and marker['mirrored']!=label['mirrored'] for marker in row['markers'])
print(json.dumps(result, indent=2))
