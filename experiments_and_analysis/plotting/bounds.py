"""Required paper bound renderer, copied from submission revision 9f519c9.

Only the shared style helpers and mean/one-sigma renderer are retained.
Experiment selection and completion handling live in render.py.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext


def format_title_text(title_text):
    return (
        title_text
        .replace("_", " ")
        .replace("dim", "dimension")
        .title()
        .replace("Mmls", "MMLS")
        .replace("Beta Vae", r"$\beta$-VAE")
    )


# noinspection PyUnusedLocal
def format_abs_values(y, p):
    """Format y-axis values to show readable absolute values, handling log scale properly."""
    if y <= 0:
        return '0'

    # Handle very small values
    if y < 0.001:
        return f'{y:.1e}'
    # For values between 0.001 and 1, show 3 decimal places
    elif y < 1:
        return f'{y:.3f}'.rstrip('0').rstrip('.')
    # For values between 1 and 10, show 1 decimal place if needed
    elif y < 10:
        formatted = f'{y:.1f}'
        if formatted.endswith('.0'):
            return formatted[:-2]  # Remove .0 for whole numbers
        return formatted
    # For values >= 10, show as integers
    else:
        return f'{int(round(y))}'


class CustomLogFormatter(LogFormatterMathtext):
    """Custom log formatter that shows absolute values instead of scientific notation."""

    def __call__(self, x, pos=None):
        """Format log scale values as absolute numbers."""
        if x <= 0:
            return '0'
        elif x < 0.001:
            return f'{x:.1e}'
        elif x < 1:
            return f'{x:.3f}'.rstrip('0').rstrip('.')
        elif x < 10:
            formatted = f'{x:.1f}'
            if formatted.endswith('.0'):
                return formatted[:-2]
            return formatted
        else:
            return f'{int(round(x))}'


def format_label(label):
    """Format bound labels to readable format."""
    if 'upper' in label.lower():
        return f"Upper Bound - {label.replace('upper', '').replace('_', ' ').strip().title()}"
    elif 'lower' in label.lower():
        return f"Lower Bound - {label.replace('lower', '').replace('_', ' ').strip().title()}"
    else:
        return label.replace('_', ' ').title()


def assign_colors_by_group(bound_types):
    """
    Assign colors to bound types, grouping by common substrings.
    Keys with shared substrings (like 'genovese') get the same color.
    Returns both color_map and group_info for legend handling.
    """
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    color_map = {}
    groups = {}
    color_index = 0

    # Group bounds by common substrings
    for bound_type in bound_types:
        bound_lower = bound_type.lower()
        group_found = False

        # Check if this bound belongs to an existing group
        for group_key in groups:
            if group_key in bound_lower:
                groups[group_key].append(bound_type)
                group_found = True
                break

        # If no group found, check if it should start a new group
        if not group_found:
            # Look for potential group identifiers (common substrings)
            potential_groups = ['genovese', 'fefferman', 'empirical', 'upper', 'lower']
            group_key = None

            for potential in potential_groups:
                if potential in bound_lower:
                    group_key = potential
                    break

            if group_key and group_key not in groups:
                groups[group_key] = [bound_type]
            elif not group_key:
                # Create individual group for bounds without common identifiers
                groups[bound_type] = [bound_type]

    # Assign colors to groups and create group info for legends
    group_info = {}
    for group_key, group_bounds in groups.items():
        color = colors[color_index % len(colors)]

        # Create combined label for groups with multiple bounds
        if len(group_bounds) > 1:
            # Check if it's upper/lower bounds of same author
            has_upper = any('upper' in bound.lower() for bound in group_bounds)
            has_lower = any('lower' in bound.lower() for bound in group_bounds)

            if has_upper and has_lower:
                # Extract author name (e.g., "genovese" from "upper_genovese")
                author = group_key.title()
                combined_label = f"Upper/lower Bound {author}"
                group_info[group_key] = {
                    'label': combined_label,
                    'color': color,
                    'bounds': group_bounds
                }
            else:
                # Just use first bound's formatted label
                group_info[group_key] = {
                    'label': format_label(group_bounds[0]),
                    'color': color,
                    'bounds': group_bounds
                }
        else:
            group_info[group_key] = {
                'label': format_label(group_bounds[0]),
                'color': color,
                'bounds': group_bounds
            }

        for bound in group_bounds:
            color_map[bound] = color
        color_index += 1

    return color_map, group_info


# noinspection PyUnusedLocal
def get_legend_entries(bound_types, color_map, group_info):
    """
    Get unique legend entries, combining bounds with same colors.
    """
    legend_entries = []
    processed_groups = set()

    for bound_type in bound_types:
        # Find which group this bound belongs to
        group_key = None
        for gkey, ginfo in group_info.items():
            if bound_type in ginfo['bounds']:
                group_key = gkey
                break

        if group_key and group_key not in processed_groups:
            legend_entries.append({
                'label': group_info[group_key]['label'],
                'color': group_info[group_key]['color']
            })
            processed_groups.add(group_key)

    return legend_entries


def get_grouped_legend_label(bound_type, group_info, emitted_groups):
    """Return one legend label per grouped bound family."""
    for group_key, group in group_info.items():
        if bound_type in group['bounds']:
            if group_key in emitted_groups:
                return ""
            emitted_groups.add(group_key)
            return group['label']

    return format_label(bound_type)


def compute_figsize(n_cols, width_per_col=9, height=9, min_width=10):
    """Compute a `figsize` tuple that scales naturally with the number of columns.

    width_per_col: horizontal space allocated per subplot column
    height: figure height in inches
    min_width: minimum total figure width in inches
    """
    fig_width = max(min_width, width_per_col * n_cols)
    return (fig_width, height)


def resolve_plot_fontsizes(fontsize=None, title_fontsize=None, text_fontsize=None):
    """Resolve optional global/title/text font-size controls.

    Defaults reproduce the original hard-coded values:
    titles=28, axis labels=24, ticks=20, legends=22.
    If fontsize is set, it controls all text unless title_fontsize or
    text_fontsize are set explicitly.
    """
    if fontsize is None:
        title_size = 28
        label_size = 24
        tick_size = 20
        legend_size = 22
    else:
        title_size = fontsize
        label_size = fontsize
        tick_size = fontsize
        legend_size = fontsize

    if title_fontsize is not None:
        title_size = title_fontsize
    if text_fontsize is not None:
        label_size = text_fontsize
        tick_size = text_fontsize
        legend_size = text_fontsize

    return title_size, label_size, tick_size, legend_size


def resolve_legend_y(legend_y=None):
    return -0.05 if legend_y is None else legend_y


def create_repetitions_shaded_plot(
        data,
        output_dir,
        log_scale=False,
        first_only=False,
        no_axes=False,
        full_version=False,
        fontsize=None,
        title_fontsize=None,
        text_fontsize=None,
        legend_y=None,
        name="bounds"
):
    """
    Create empirical curves, shading ±sigma only with at least three repetitions.

    Args:
        data: Mapping of panel names to plot-ready bound dictionaries
        output_dir: Directory to save the plots
        log_scale: Whether to use logarithmic scale for y-axis
        first_only: Whether to plot only the first key with legend inside figure
        no_axes: Whether to disable axes and axis labels
        full_version: If true, don't sub-sample curve points.
    """
    title_size, label_size, tick_size, legend_size = resolve_plot_fontsizes(
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        text_fontsize=text_fontsize,
    )
    shared_legend_y = resolve_legend_y(legend_y)

    # Load the data

    # Get the keys dynamically from the numpy object
    keys = list(data.keys())
    if first_only:
        keys = keys[:1]  # Only use first key
    n_cols = len(keys)

    # Check if any key has empirical_all_repetitions
    has_repetitions = False
    for key in keys:
        dim_data = data[key]
        if 'empirical_all_repetitions' in dim_data:
            has_repetitions = True
            break

    if not has_repetitions:
        print("No empirical_all_repetitions data found, skipping shaded plot generation.")
        return

    # Extract bound types from first dimension to determine legend
    sample_data = data[keys[0]]
    bound_types = [
        key for key in sample_data.keys()
        if key != 'x' and key != 'x_perc' and key != 'empirical_all_repetitions'
    ]

    # Assign colors based on groups
    color_map, group_info = assign_colors_by_group(bound_types)
    emitted_legend_groups = set()

    # Create the multicolumn plot (larger figsize)
    figsize = compute_figsize(n_cols, width_per_col=10, height=9, min_width=10)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    # Plot each dimension
    for i, key in enumerate(keys):
        dim_data = data[key]

        if 'empirical_all_repetitions' not in dim_data:
            continue

        x = dim_data['x']
        x_relative = (x / x[-1]) * 100

        # Determine subsampling indices for bounds if needed
        if len(x) > 20 and not full_version:
            step = len(x) // 15  # Keep about 15 points
            indices = np.arange(0, len(x), step)
            if indices[-1] != len(x) - 1:
                indices = np.append(indices, len(x) - 1)  # Always include last point
            use_subsampling = True
        else:
            indices = np.arange(len(x))
            use_subsampling = False

        # Plot other bounds first (dotted lines) with subsampling
        for j, bound_type in enumerate(bound_types):
            if bound_type in dim_data and bound_type != 'empirical':
                y = dim_data[bound_type]

                # Skip if y contains non-numeric data
                try:
                    y = np.array(y, dtype=float)
                except (ValueError, TypeError):
                    continue

                # Apply subsampling if needed
                if use_subsampling:
                    x_bound = x[indices]
                    x_relative_bound = (x_bound / x_bound[-1]) * 100
                    y_bound = y[indices]
                else:
                    x_relative_bound = x_relative
                    y_bound = y

                formatted_label = (
                    get_grouped_legend_label(bound_type, group_info, emitted_legend_groups)
                    if i == 0 else ""
                )
                axes[i].plot(
                    x_relative_bound,
                    y_bound,
                    color=color_map[bound_type],
                    marker='o',
                    linewidth=3,
                    markersize=8,
                    linestyle=':',
                    label=formatted_label
                )

        # Extract repetitions data and compute statistics
        repetitions_data = dim_data['empirical_all_repetitions']

        means = []
        stds = []
        counts = []
        for j in range(len(x)):
            if hasattr(repetitions_data, 'iloc'):
                values = repetitions_data.iloc[j]
            else:
                values = repetitions_data[j]

            if isinstance(values, (list, np.ndarray)):
                values = np.array(values)
                means.append(np.mean(values))
                stds.append(np.std(values))
                counts.append(len(values))
            else:
                means.append(float(values))
                stds.append(0.0)
                counts.append(1)

        means = np.array(means)
        stds = np.array(stds)

        # Use the same subsampling as bounds for consistency
        if use_subsampling:
            x_sub = x[indices]
            x_relative_sub = (x_sub / x_sub[-1]) * 100
            means_sub = means[indices]
            stds_sub = stds[indices]
        else:
            x_relative_sub = x_relative
            means_sub = means
            stds_sub = stds

        shade = np.array(counts)[indices] >= 3
        # Shade only intervals whose two endpoints have enough repetitions.
        has_band = np.any(shade[:-1] & shade[1:])
        empirical_label = ('Empirical (Mean ± σ)' if shade.all() else
                           'Empirical (Mean; ± σ where n ≥ 3)' if has_band else
                           'Empirical (Mean)' if max(counts) > 1 else 'Empirical')
        label_text = empirical_label if i == 0 else ""
        axes[i].plot(
            x_relative_sub,
            means_sub,
            color='tab:blue',
            marker='o',
            linewidth=4,
            markersize=8,
            linestyle='-',
            label=label_text
        )

        if has_band:
            axes[i].fill_between(
                x_relative_sub,
                means_sub - stds_sub,
                means_sub + stds_sub,
                where=shade,
                color='tab:blue',
                alpha=0.3
            )

        # Configure axes based on no_axes flag
        if no_axes:
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='both', which='major', labelbottom=False, labelleft=False)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            axes[i].set_title('')
        else:
            title_text = format_title_text(key)

            axes[i].set_title(title_text, fontsize=title_size,
                              # fontweight='bold',
                              pad=15)
            axes[i].set_xlabel('Relative Sample Size (%)', fontsize=label_size, labelpad=8)
            if i == 0:
                axes[i].set_ylabel('Hausdorff Distance', fontsize=label_size, labelpad=8)

            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='both', which='major', labelsize=tick_size)
            axes[i].tick_params(axis='x', rotation=45)
            # Format x-axis to show integer values only
            axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda xx, p: f'{int(xx)}'))

        if log_scale:
            axes[i].set_yscale('log')
            # Use custom log formatter for log scale
            axes[i].yaxis.set_major_formatter(CustomLogFormatter())
        else:
            # Use regular formatter for linear scale
            axes[i].yaxis.set_major_formatter(FuncFormatter(format_abs_values))

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    if first_only:
        axes[0].legend(handles, labels, loc='upper right', fontsize=legend_size, markerscale=2.5)
    else:
        fig.legend(
            handles,
            labels,
            loc='lower center',
            ncol=len(labels),
            bbox_to_anchor=(0.5, shared_legend_y),
            fontsize=legend_size,
            markerscale=2.5
        )

    # Adjust layout
    plt.tight_layout()
    if not first_only:
        plt.subplots_adjust(bottom=0.25)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    input_filename = name
    scale_suffix = "_log" if log_scale else "_absolute"
    first_suffix = "_first" if first_only else ""
    axes_suffix = "_no_axes" if no_axes else ""
    output_filename = f"{input_filename}_shaded{scale_suffix}{first_suffix}{axes_suffix}.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved shaded plot: {output_path}")

    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF: {pdf_path}")

    plt.close()


