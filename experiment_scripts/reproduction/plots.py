"""Render Figures 3–6 using the preserved submission styles."""
import argparse
from pathlib import Path

from .bounds import FILENAMES
from .common import run_experiment


def plot_bounds(output, bounds_dir, figure):
    bounds_dir = Path(bounds_dir)

    def action(directory):
        from ._bound_style import create_repetitions_shaded_plot
        stem = FILENAMES[figure]
        create_repetitions_shaded_plot(bounds_dir / (stem + '.npz'), directory,
                                       log_scale=False)
        files = []
        for suffix in ('png', 'pdf'):
            source = directory / f'{stem}_shaded_absolute.{suffix}'
            target = directory / f'figure_{figure}.{suffix}'
            source.rename(target)
            files.append(target)
        return files

    return run_experiment(output, action, requires=[bounds_dir])


def plot_layers(output, dsprites_analysis, coil_analysis):
    dsprites_analysis, coil_analysis = Path(dsprites_analysis), Path(coil_analysis)

    def action(directory):
        from PIL import Image
        from . import _layer_style
        _layer_style.MEASURES_DIRS.update(
            dsprites=str(dsprites_analysis / 'measures'), coil=str(coil_analysis / 'measures'))
        _layer_style.create_layer_plots('dsprites', str(directory), aggregate=False,
                                        legend_title='Object class')
        _layer_style.create_layer_plots('coil', str(directory), aggregate=True)
        # Figure 6 contains both rows. Preserve each original row as PNG/PDF too.
        with Image.open(directory / 'layers_dsprites.png') as ds, Image.open(directory / 'layers_coil.png') as coil:
            top, bottom = ds.convert('RGB'), coil.convert('RGB')
            width = max(top.width, bottom.width)
            rows = [image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
                    for image in (top, bottom)]
            figure = Image.new('RGB', (width, sum(image.height for image in rows)), 'white')
            figure.paste(rows[0], (0, 0))
            figure.paste(rows[1], (0, rows[0].height))
            figure.save(directory / 'figure_6.png', dpi=(300, 300))
            figure.save(directory / 'figure_6.pdf', resolution=300)
        return [directory / 'figure_6.png', directory / 'figure_6.pdf',
                directory / 'layers_dsprites.pdf', directory / 'layers_coil.pdf']

    return run_experiment(output, action, requires=[dsprites_analysis, coil_analysis])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', type=int, choices=[3, 4, 5, 6], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bounds-dir', type=Path)
    parser.add_argument('--dsprites-analysis', type=Path)
    parser.add_argument('--coil-analysis', type=Path)
    args = parser.parse_args()
    if args.figure == 6:
        if args.dsprites_analysis is None or args.coil_analysis is None:
            parser.error('Figure 6 needs --dsprites-analysis and --coil-analysis')
        plot_layers(args.output, args.dsprites_analysis, args.coil_analysis)
    else:
        if args.bounds_dir is None:
            parser.error('Bound figures need --bounds-dir')
        plot_bounds(args.output, args.bounds_dir, args.figure)


if __name__ == '__main__':
    main()
