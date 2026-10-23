"""IO wrappers for plot-ready results and the retained paper styles."""
from pathlib import Path
import pickle

import numpy as np
from PIL import Image

from ..analyses.bounds import _curve
from ..outputs import artifact_path, begin, finish, identity, read_json, stage_directory, write_json


def matches(record, selection):
    for path, expected in selection.items():
        value = record
        for key in path.split('.'):
            if not isinstance(value, dict) or key not in value:
                return False
            value = value[key]
        if value != expected:
            return False
    return True


def bound_data(records, panels):
    """Aggregate only the explicitly selected experiment results."""
    curves, summaries = {}, {}
    for panel in panels:
        selected = [record for record in records if record['name'] == 'errors'
                    and matches(record, panel.get('where', {}))]
        # An analysis can be referenced by several experiment configurations.
        selected = {record['id']: record for record in selected}.values()
        rows = [read_json(artifact_path(record['directory']) / 'metrics.json') for record in selected]
        curves[panel['label']], summaries[panel['label']] = _curve(rows, panel['dimension'])
    return curves, summaries


def layer_data(records, selection):
    selected = [record for record in records if record['name'] == 'geometry'
                and matches(record, selection)]
    results = {}
    selected_ids = {}
    for record in selected:
        name = record['representation']
        if name in results:
            if selected_ids[name] == record['id']:
                continue
            raise ValueError('Select one model trial and geometry configuration per layer panel.')
        with (Path(record['directory']) / 'measures.pkl').open('rb') as stream:
            results[name] = pickle.load(stream)
            selected_ids[name] = record['id']
    if not results:
        raise ValueError(f'No geometry results match {selection}')
    return results


def render(records, request, output_dir):
    """Named aggregation/rendering selection, with no callback dispatch."""
    from .bounds import create_repetitions_shaded_plot
    from .layers import create_layer_plots
    root = Path(output_dir)
    config = dict(request=request, results=sorted({r['id'] for r in records}))
    plot_id = identity(config)
    directory = stage_directory(root / 'plots', plot_id, request['filename'])
    if not begin(directory, config):
        return directory
    match request['name']:
        case 'bounds':
            curves, summaries = bound_data(records, request['panels'])
            aggregate_dir = stage_directory(root / 'analysis-outputs', plot_id,
                                            f"aggregate__{request['filename']}__bounds")
            if begin(aggregate_dir, config):
                np.savez(aggregate_dir / 'curves.npz', **curves)
                write_json(aggregate_dir / 'regression.json', summaries)
                finish(aggregate_dir, ['curves.npz', 'regression.json'])
            create_repetitions_shaded_plot(curves, directory, name=request['filename'],
                                            log_scale=request.get('log_scale', False))
            suffix = 'log' if request.get('log_scale', False) else 'absolute'
            for extension in ('png', 'pdf'):
                (directory / f"{request['filename']}_shaded_{suffix}.{extension}").rename(
                    directory / f"{request['filename']}.{extension}")
        case 'layers':
            rows = []
            for panel in request['panels']:
                measures = layer_data(records, panel['where'])
                label = panel['label']
                create_layer_plots(measures, dataset_type=label, output_dir=directory,
                                   aggregate=panel.get('aggregate', False),
                                   legend_title=panel.get('legend_title'))
                rows.append(directory / f'layers_{label}.png')
            images = []
            for path in rows:
                with Image.open(path) as image:
                    images.append(image.convert('RGB'))
            width = max(image.width for image in images)
            images = [image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
                      for image in images]
            figure = Image.new('RGB', (width, sum(image.height for image in images)), 'white')
            offset = 0
            for image in images:
                figure.paste(image, (0, offset))
                offset += image.height
            figure.save(directory / f"{request['filename']}.png", dpi=(300, 300))
            figure.save(directory / f"{request['filename']}.pdf", resolution=300)
        case _:
            raise ValueError(f"Unknown plot: {request['name']}")
    finish(directory, [f"{request['filename']}.{extension}" for extension in ('png', 'pdf')])
    return directory
