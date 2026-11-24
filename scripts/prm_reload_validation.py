"""CPU-only, fail-closed validation of complete PRM prediction records.

The default comparison is absolute tolerance 1e-5, relative tolerance zero.
Both the saved and independently reloaded records must pass the same checks.
"""
import math


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    if not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return value


def _task_ids(ids):
    ids = list(ids)
    if not ids or any(type(i) is not int for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Expected task IDs must be nonempty, integer and unique')
    return set(ids)


def _tolerances(atol, rtol):
    if _number(atol, 'atol') < 0 or _number(rtol, 'rtol') < 0:
        raise ValueError('Comparison tolerances must be nonnegative')


def _close(actual, reference, atol, rtol):
    # Explicit reference-relative tolerance; all operands have been validated.
    return abs(actual - reference) <= atol + rtol * abs(reference)


def validate_predictions(records, expected_task_ids, *, atol=1e-5, rtol=0.0):
    """Validate coverage, positions, finite scores and total=sum(step_scores)."""
    _tolerances(atol, rtol)
    expected = _task_ids(expected_task_ids)
    if not isinstance(records, list) or not records:
        raise ValueError('Prediction records must be a nonempty list')
    indexed = {}
    for row in records:
        if not isinstance(row, dict) or type(row.get('task_id')) is not int:
            raise ValueError('Each prediction must have an integer task_id')
        task = row['task_id']
        if task in indexed:
            raise ValueError(f'Duplicate prediction task_id: {task}')
        positions, scores = row.get('turn_end_indices'), row.get('step_scores')
        if (not isinstance(positions, list) or not positions
                or any(type(i) is not int or i < 0 for i in positions)
                or any(a >= b for a, b in zip(positions, positions[1:]))):
            raise ValueError(f'Invalid or empty scoring positions: {task}')
        if not isinstance(scores, list) or len(scores) != len(positions):
            raise ValueError(f'Score/position count mismatch: {task}')
        for score in scores:
            _number(score, f'step score for task {task}')
        total = _number(row.get('total'), f'total for task {task}')
        try:
            summed = math.fsum(scores)
        except OverflowError as error:
            raise ValueError(f'Nonfinite score sum: {task}') from error
        if not math.isfinite(summed) or not _close(total, summed, atol, rtol):
            raise ValueError(f'Total differs from step-score sum: {task}')
        indexed[task] = row
    if set(indexed) != expected:
        raise ValueError('Missing or unexpected prediction task IDs')
    return indexed


def compare_predictions(current, reference, expected_task_ids, *, atol=1e-5, rtol=0.0):
    """Compare every aligned step and total; never discard duplicate records."""
    expected_task_ids = list(expected_task_ids)
    current = validate_predictions(current, expected_task_ids, atol=atol, rtol=rtol)
    reference = validate_predictions(reference, expected_task_ids, atol=atol, rtol=rtol)
    max_step = max_total = 0.0
    steps = 0
    for task in expected_task_ids:
        row, old = current[task], reference[task]
        if row['turn_end_indices'] != old['turn_end_indices']:
            raise ValueError(f'Independent reload scoring positions differ: {task}')
        # Equal positions imply equal, nonempty vector lengths, checked above.
        for actual, saved in zip(row['step_scores'], old['step_scores']):
            delta = abs(actual - saved)
            if not math.isfinite(delta) or not _close(actual, saved, atol, rtol):
                raise ValueError(f'Independent reload step score differs: {task}')
            max_step = max(max_step, delta)
            steps += 1
        delta = abs(row['total'] - old['total'])
        if not math.isfinite(delta) or not _close(row['total'], old['total'], atol, rtol):
            raise ValueError(f'Independent reload total differs: {task}')
        max_total = max(max_total, delta)
    return {'atol': atol, 'rtol': rtol, 'tasks_compared': len(current),
            'steps_compared': steps, 'max_step_difference': max_step,
            'max_total_difference': max_total}
