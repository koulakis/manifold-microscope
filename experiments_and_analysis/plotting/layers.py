"""Paper layer styles, receiving explicitly selected in-memory analysis results."""

import os

import numpy as np
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from microscope.computations_grid.data_analysis.data_analysis import (
    clip_results_multiclass,
)


def format_two_decimals(y, _):
    """Format y-axis values with at most 2 decimal places (or scientific notation for extremes).

    This ensures labels show at most 2 numbers behind the decimal point.
    """
    if y == 0:
        return '0'

    sign = '-' if y < 0 else ''
    ay = abs(y)

    # Use scientific notation for very small or very large values
    if ay < 1e-3 or ay >= 1e4:
        return sign + f'{ay:.2e}'

    # Otherwise use at most 2 decimal places, strip trailing zeros
    s = f'{ay:.2f}'.rstrip('0').rstrip('.')
    if s == '':
        s = '0'
    return sign + s


def format_three_decimals(y, _):
    if y == 0:
        return '0'

    sign = '-' if y < 0 else ''
    ay = abs(y)

    if ay < 1e-4 or ay >= 1e4:
        return sign + f'{ay:.3e}'

    s = f'{ay:.3f}'.rstrip('0').rstrip('.')
    if s == '':
        s = '0'
    return sign + s


class CustomLogFormatter(LogFormatterMathtext):
    """Custom log formatter with scientific notation for large values."""

    def __call__(self, x, _=None):
        """Format log scale values with proper scientific notation."""
        # Log scale should only have positive values
        if x <= 0:
            return '0'

        # Use scientific notation for values outside [-99, 99] range
        if x >= 99:
            return f'{x:.1e}'

        # For values within [0, 99], use at most 2 decimal places
        if x < 0.01:
            return f'{x:.2e}'
        elif x < 1:
            formatted = f'{x:.3f}'.rstrip('0').rstrip('.')
            return formatted if formatted != '' else '0'
        else:
            formatted = f'{x:.2f}'
            if formatted.endswith('.00'):
                return f'{int(x)}'
            return formatted.rstrip('0').rstrip('.')


def assign_colors_by_group(measure_names):
    """Assign colors to measures, using consistent color scheme."""
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    color_map = {}
    for i, measure_name in enumerate(measure_names):
        color_map[measure_name] = colors[i % len(colors)]

    return color_map


def determine_scale(values, threshold_small=1e-3, threshold_large=1e3):
    """Determine if log scale should be used based on value ranges."""
    if len(values) == 0:
        return False

    # Get all non-zero values
    non_zero_values = [v for v in values if v > 0]
    if not non_zero_values:
        return False

    min_val = min(non_zero_values)
    max_val = max(non_zero_values)

    # Use log scale if values span very large range or include very small/large values
    return (min_val < threshold_small or max_val > threshold_large or
            (max_val / min_val) > 1000)


# Faint background tints for the encoder / decoder halves of the network.
# Kept near-white so they never clash with the saturated class line colors.
ENCODER_TINT = '#ECF2F9'
DECODER_TINT = '#FBF1EA'


def shade_encoder_decoder(ax, layer_names, label=True, fontsize=None):
    """Tint the encoder vs decoder x-ranges and mark the bottleneck.

    Encoder spans input..mu, decoder spans mu..output. The latent layer
    ('mu') sits on the seam. Adds 'Encoder'/'Decoder' labels when requested.
    """
    boundary = layer_names.index('mu') if 'mu' in layer_names else (len(layer_names) - 1) / 2
    xmin, xmax = ax.get_xlim()

    ax.axvspan(xmin, boundary, color=ENCODER_TINT, zorder=0)
    ax.axvspan(boundary, xmax, color=DECODER_TINT, zorder=0)
    ax.axvline(boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

    if label:
        trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
        label_fontsize = fontsize if fontsize is not None else 16
        ax.text((xmin + boundary) / 2, 0.97, 'Encoder', transform=trans,
                ha='center', va='top', fontsize=label_fontsize, fontstyle='italic', color='dimgray')
        ax.text((boundary + xmax) / 2, 0.97, 'Decoder', transform=trans,
                ha='center', va='top', fontsize=label_fontsize, fontstyle='italic', color='dimgray')


def prettify_layer_name(name):
    """Short x-tick labels so rotated ticks don't collide with the legend."""
    if name.startswith('encoder_relu_'):
        return f"enc {name.split('_')[-1]}"
    if name.startswith('decoder_relu_'):
        return f"dec {name.split('_')[-1]}"
    return name


measure_names_fns = {
        "Volume": lambda a: a.total_volumes,
        # "Normalized volume": lambda a: a.normalized_total_volumes,
        "Curvature": lambda a: a.normalized_total_curvatures,
        "Curvature negative": lambda a: a.normalized_total_curvatures_negative,
        "Curvature positive": lambda a: a.normalized_total_curvatures_positive,
        "Reach": lambda a: a.normalized_min_reaches,
        # "Average reach": lambda a: a.normalized_total_reaches,
        # "Total distances": lambda a: a.normalized_total_distances,
        "Average distances": lambda a: a.normalized_average_distances
    }


def create_layer_plots(layer_results, dataset_type='dsprites', output_dir='.', aggregate=False,
                       output_suffix='', legend_title=None, suptitle=None,
                       sample_classes=None, omit_positive_negative_curvature=False,
                       fontsize=None, title_fontsize=None, text_fontsize=None,
                       plot_height_percentage=None, legend_y=None):
    """Create layer analysis plots for the specified dataset.

    aggregate=False: one line per class (full class-wise view, e.g. appendix).
    aggregate=True:  a single median line + interquartile band over all classes
                     (simplified summary, e.g. COIL-20 in the main paper).
    sample_classes:  if set to an int N, keep only N evenly-spaced classes
                     (deterministic) before plotting. Ignored when aggregate.
    legend_y:       y-position of the shared legend in figure coordinates.
                    More negative values move it farther down. None preserves
                    the original value, -0.11.
    """
    print(f"Processing {dataset_type} dataset (aggregate={aggregate})...")

    effective_title_fontsize = title_fontsize if title_fontsize is not None else fontsize
    effective_text_fontsize = text_fontsize if text_fontsize is not None else fontsize

    aggregates = [clip_results_multiclass(measures).measure_aggregates
                  for measures in layer_results.values()]
    layer_names = list(layer_results)
    if not aggregates:
        print(f"No data found for {dataset_type} dataset")
        return

    # Get actual class names and sprite combinations from the data
    actual_class_names = list(aggregates[0].total_volumes.keys())
    sprites_combinations = list(aggregates[0].normalized_total_distances.keys())

    # Optionally keep only N evenly-spaced classes (deterministic) for a lighter view.
    if sample_classes and sample_classes < len(actual_class_names):
        idx = np.linspace(0, len(actual_class_names) - 1, sample_classes)
        idx = sorted(set(int(round(i)) for i in idx))
        kept = {actual_class_names[i] for i in idx}
        actual_class_names = [c for c in actual_class_names if c in kept]
        sprites_combinations = [p for p in sprites_combinations
                                if p[0] in kept and p[1] in kept]
        print(f"Sampled {len(actual_class_names)} classes: {actual_class_names}")

    print(f"Found classes in data: {actual_class_names}")
    print(f"Found sprite combinations: {sprites_combinations[:3]}...")  # Show first few

    # Assign colors to measures
    current_measure_names_fns = dict(measure_names_fns)
    if omit_positive_negative_curvature:
        current_measure_names_fns.pop("Curvature negative", None)
        current_measure_names_fns.pop("Curvature positive", None)

    measure_names = list(current_measure_names_fns.keys())
    color_map = assign_colors_by_group(measure_names)

    # Create the plot
    cols = len(current_measure_names_fns)
    base_fig_height = 8
    if plot_height_percentage is None:
        fig_height = base_fig_height
    else:
        if plot_height_percentage <= 0:
            raise ValueError("plot_height_percentage must be positive when set.")
        fig_height = base_fig_height * plot_height_percentage / 100.0

    fig, axes = plt.subplots(1, cols, figsize=(7*cols, fig_height))
    if cols == 1:
        axes = [axes]

    # Store all handles and labels for shared legend
    all_handles = []
    all_labels = []

    def axis_coord_to_data_y(ax, y_axis_coord):
        """Convert a y coordinate in axes fraction (0..1) to data coordinate using transforms."""
        disp = ax.transAxes.transform([(0, y_axis_coord)])
        data = ax.transData.inverted().transform(disp)
        return data[0][1]

    for i, (measure_name, measure_fn) in enumerate(current_measure_names_fns.items()):
        # Determine which classes to use (sprite combinations for distances, actual class names for others)
        classes_to_plot = sprites_combinations if "distances" in measure_name.lower() else actual_class_names

        if aggregate:
            # Collapse all classes into one mean curve +/- one standard deviation.
            stacked = np.array([[measure_fn(a)[sprite] for a in aggregates]
                                for sprite in classes_to_plot])  # (n_classes, n_layers)
            mean = np.mean(stacked, axis=0)
            std = np.std(stacked, axis=0)
            lower, upper = mean - std, mean + std
            # Scale on the band extent so the summary fills the panel.
            all_values = list(lower) + list(upper) + list(mean)
            use_log = determine_scale(all_values)

            x = range(len(layer_names))
            axes[i].fill_between(x, lower, upper, color='tab:blue', alpha=0.25, zorder=2)
            axes[i].plot(x, mean, color='tab:blue',
                         marker='o', linewidth=3, markersize=8, zorder=3)

            if i == 0:
                all_handles.append(Line2D([0], [0], color='tab:blue', lw=3,
                                          marker='o', markersize=8))
                all_labels.append(f'Mean over {len(classes_to_plot)} classes')
                all_handles.append(Patch(facecolor='tab:blue', alpha=0.25))
                all_labels.append(r'$\pm\,1$ std. dev. ($\sigma$)')
        else:
            # Collect ALL values (including negative) to ensure proper scaling
            all_values = []
            for sprite in classes_to_plot:
                values = [measure_fn(a)[sprite] for a in aggregates]
                all_values.extend(values)  # Include ALL values, not just positive ones

            use_log = determine_scale(all_values)

            # Plot each class/sprite combination as its own line
            for j, sprite in enumerate(classes_to_plot):
                values = [measure_fn(a)[sprite] for a in aggregates]

                # Use consistent colors across subplots for same classes
                color_key = str(sprite)
                if color_key not in color_map:
                    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple',
                             'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
                    color_map[color_key] = colors[j % len(colors)]

                line = axes[i].plot(range(len(layer_names)), values,
                                   color=color_map[color_key],
                                   marker='o', linewidth=3, markersize=8,
                                   label=str(sprite) if i == 0 else "")

                # Collect handles and labels from first subplot only
                if i == 0:
                    all_handles.extend(line)
                    all_labels.append(str(sprite))

        # Configure axes
        axes[i].set_title(measure_name, fontsize=effective_title_fontsize if effective_title_fontsize is not None else 28,
                          # fontweight='bold',
                          pad=15)

        if i == 0:
            axes[i].set_ylabel('Measure Value', fontsize=effective_text_fontsize if effective_text_fontsize is not None else 24, labelpad=8)

        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', which='major', labelsize=effective_text_fontsize if effective_text_fontsize is not None else 20)
        axes[i].set_xticks(range(len(layer_names)))
        axes[i].set_xticklabels([prettify_layer_name(n) for n in layer_names],
                                rotation=45, ha='right')

        # Set x-axis limits to ensure all layer names are visible
        axes[i].set_xlim(-0.5, len(layer_names) - 0.5)

        # Set scale and formatter with proper limits to ensure all values are visible
        if use_log:
            # For log-like scale, set symlog so both positive/negative values can be shown
            y_min = min(all_values)
            y_max = max(all_values)
            axes[i].set_yscale('symlog')
            # Add some padding while respecting sign
            pad_low = y_min * 0.5 if y_min < 0 else y_min * 0.5
            pad_high = y_max * 1.5
            # If y_min is negative and small in magnitude, ensure lower limit is less than upper
            axes[i].set_ylim(pad_low, pad_high)
            # Use our two-decimal formatter
            axes[i].yaxis.set_major_formatter(FuncFormatter(format_two_decimals))

        else:
            axes[i].yaxis.set_major_formatter(FuncFormatter(format_two_decimals))
            # For linear scale, set explicit limits to ensure all values are visible
            if all_values:
                y_min = min(all_values)
                y_max = max(all_values)
                if y_min == y_max:
                    # Handle case where all values are identical
                    if y_min == 0:
                        axes[i].set_ylim(-0.1, 0.1)
                    else:
                        padding = abs(y_min) * 0.1
                        axes[i].set_ylim(y_min - padding, y_min + padding)
                else:
                    # Add 10% padding to ensure values don't touch plot edges
                    y_range = y_max - y_min
                    padding = y_range * 0.1
                    axes[i].set_ylim(y_min - padding, y_max + padding)

        # Force a draw so transforms are correct (needed for converting axis coords to data coords)
        fig.canvas.draw()

        # Set exactly 5 equidistant visual ticks (in axis fraction space) and map back to data coords
        n_ticks = 5
        axis_positions = np.linspace(0.0, 1.0, n_ticks)
        tick_values = [axis_coord_to_data_y(axes[i], a) for a in axis_positions]
        axes[i].set_yticks(tick_values)
        # Ensure labels respect our 2-decimal formatter
        if measure_name == "Reach":
            axes[i].yaxis.set_major_formatter(FuncFormatter(format_three_decimals))
        else:
            axes[i].yaxis.set_major_formatter(FuncFormatter(format_two_decimals))

        # Tint encoder vs decoder halves and label the bottleneck.
        shade_encoder_decoder(axes[i], layer_names, fontsize=effective_text_fontsize)

    # Explain the encoder/decoder shading once in the shared legend.
    all_handles.append(Patch(facecolor=ENCODER_TINT))
    all_labels.append('Encoder (input → latent)')
    all_handles.append(Patch(facecolor=DECODER_TINT))
    all_labels.append('Decoder (latent → output)')

    # Add shared legend below the plots, aligned horizontally with the plot area
    if all_handles:
        # Calculate optimal number of columns to make legend span the plot width
        max_cols = len(all_labels)
        # Try to fit legend in 2-3 rows maximum for better horizontal alignment
        if max_cols <= 6:
            optimal_cols = max_cols  # Single row if 6 or fewer items
        elif max_cols <= 12:
            optimal_cols = (max_cols + 1) // 2  # Two rows
        else:
            optimal_cols = (max_cols + 2) // 3  # Three rows

        effective_legend_y = -0.11 if legend_y is None else legend_y
        legend = fig.legend(all_handles, all_labels, loc='lower center',
                           ncol=optimal_cols,
                           bbox_to_anchor=(0.5, effective_legend_y),  # Just below the rotated x-axis labels
                           fontsize=effective_text_fontsize if effective_text_fontsize is not None else 18, markerscale=1.8,
                           columnspacing=0.8,  # Reduced space between columns
                           handletextpad=0.4,  # Reduced space between marker and text
                           title=legend_title,
                           frameon=True)       # Keep legend frame
        if legend_title:
            legend.get_title().set_fontsize(effective_text_fontsize if effective_text_fontsize is not None else 18)

    # Adjust layout with proper spacing for horizontally aligned legend
    plt.tight_layout()
    if all_handles:
        # Calculate bottom margin based on number of legend rows
        num_rows = (len(all_labels) + optimal_cols - 1) // optimal_cols
        bottom_margin = 0.14 + (num_rows - 1) * 0.03  # Room for rotated ticks + legend
        plt.subplots_adjust(bottom=bottom_margin)

    if suptitle:
        # Reserve headroom so the suptitle clears the per-panel titles.
        plt.subplots_adjust(top=0.84)
        fig.suptitle(suptitle, fontsize=effective_title_fontsize if effective_title_fontsize is not None else 26, fontweight='bold', y=0.97)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"layers_{dataset_type}{output_suffix}.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {output_path}")

    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF: {pdf_path}")

    plt.close()


