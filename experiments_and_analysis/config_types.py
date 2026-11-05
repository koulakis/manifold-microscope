"""Dictionary schemas for experiment options and exported records."""

from typing import TypedDict


class DatasetOptions(TypedDict, total=False):
    """Dataset options before defaults and source metadata are resolved."""

    name: str
    dimension: int
    path: str
    sample_size: int
    ground_truth_size: int
    seed: int
    reference_seed: int
    range_sizes: list[float]
    cyclic_dimensions: list[int]
    patch_sizes: list[int]
    class_names: list[str]
    feature_ndim: int


class SplitOptions(TypedDict, total=False):
    """Optional split settings supplied before defaults are applied."""

    method: str
    weighted: bool
    train_fraction: float
    test_fraction: float
    max_test_size: int
    seed: int


class ModelOptions(TypedDict, total=False):
    """Model selection and scalar parameters validated by the model dataclasses."""

    name: str | None
    parameters: dict[str, bool | float | str]


class TrainingOptions(TypedDict, total=False):
    """Training overrides supplied before defaults are applied."""

    enabled: bool
    seed: int
    trial: int


class InferenceOptions(TypedDict, total=False):
    """Inference overrides supplied before defaults are applied."""

    enabled: bool
    domain: str
    representations: str | list[str]
    batch_size: int
    evaluation_size: int
    seed: int
    decoder_sampling: bool
    persist: bool
    buffer: str


class GeometryOptions(TypedDict, total=False):
    """Optional keyword arguments forwarded to the geometry computation."""

    normalize_for_volume: bool
    normalize_curvatures: bool
    reach_subsample: int
    reach_batch_size: int


class AnalysisOptions(GeometryOptions, total=False):
    """Analysis selection and optional geometry settings."""

    name: str
    domain: str
    representations: str | list[str]


class ExperimentOverrides(TypedDict, total=False):
    """Optional top-level overrides for an assembled experiment."""

    training: TrainingOptions
    inference: InferenceOptions
    output_layout: str
    device: str
    num_workers: int


class ExperimentOptions(ExperimentOverrides):
    """Assembled input sections, whose nested options may still omit defaults."""

    dataset: DatasetOptions
    split: SplitOptions
    model: ModelOptions
    analyses: list[str | AnalysisOptions]


class Repetition(TypedDict):
    """One sweep repetition and its random seed."""

    index: int
    seed: int


class SweepOverrides(TypedDict, total=False):
    """Optional plots requested after a sweep."""

    outputs: list[dict]


class SweepOptions(SweepOverrides):
    """An assembled experiment and its independently varied sweep axes."""

    base: ExperimentOptions
    grid: dict[str, list[int] | list[Repetition]]


class ExportDataset(TypedDict):
    """Dataset field used by the bundle summary export."""

    sample_size: int


class ExportTraining(TypedDict):
    """Training fields used by the bundle summary export."""

    seed: int
    trial: int


class ExportConfiguration(TypedDict):
    """Required fields of a saved configuration read by the bundle exporter."""

    dataset: ExportDataset
    training: ExportTraining


class ExportResult(TypedDict, total=False):
    """Optional completion status read from a saved experiment result."""

    status: str
