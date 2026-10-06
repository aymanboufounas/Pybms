# Generated static re-exports for IDE completion.
from .data import (
    load_data as load_data,
    from_url as from_url,
    read_csv as read_csv,
    read_json as read_json,
    read_excel as read_excel,
    read_parquet as read_parquet,
    save_data as save_data,
    sample_data as sample_data,
    random_data as random_data,
    classification_data as classification_data,
    regression_data as regression_data,
    cluster_data as cluster_data,
    split_data as split_data,
    shuffle_data as shuffle_data,
    batch_data as batch_data,
    describe_data as describe_data,
    preview_data as preview_data,
    select_columns as select_columns,
    join_data as join_data,
    concat_data as concat_data,
)
from .cleaning import (
    clean_data as clean_data,
    clean_names as clean_names,
    fill_missing as fill_missing,
    drop_missing as drop_missing,
    missing_report as missing_report,
    drop_duplicates as drop_duplicates,
    coerce_numbers as coerce_numbers,
    coerce_dates as coerce_dates,
    trim_strings as trim_strings,
    encode_categories as encode_categories,
    scale_data as scale_data,
    normalize_data as normalize_data,
    clip_outliers as clip_outliers,
    drop_outliers as drop_outliers,
    outlier_report as outlier_report,
    drop_constant_columns as drop_constant_columns,
    drop_sparse_columns as drop_sparse_columns,
    replace_values as replace_values,
    balance_data as balance_data,
    validate_data as validate_data,
)
from .arrays import (
    array as array,
    zeros as zeros,
    ones as ones,
    random_array as random_array,
    reshape as reshape,
    flatten as flatten,
    one_hot_encode as one_hot_encode,
    safe_divide as safe_divide,
    moving_average as moving_average,
    standardize as standardize,
    minmax as minmax,
    correlation as correlation,
    distance as distance,
    top_k as top_k,
    array_stats as array_stats,
)
from .plotting import (
    plot as plot,
    line_plot as line_plot,
    scatter_plot as scatter_plot,
    bar_plot as bar_plot,
    hist_plot as hist_plot,
    box_plot as box_plot,
    heatmap as heatmap,
    pie_plot as pie_plot,
    pair_plot as pair_plot,
    missing_plot as missing_plot,
    class_plot as class_plot,
    loss_plot as loss_plot,
    confusion_plot as confusion_plot,
    residual_plot as residual_plot,
    feature_plot as feature_plot,
)
from .training import (
    infer_task as infer_task,
    train as train,
    auto_train as auto_train,
    compare_models as compare_models,
    tune_model as tune_model,
    predict as predict,
    evaluate as evaluate,
    cross_validate as cross_validate,
    feature_importance as feature_importance,
    save_model as save_model,
    load_model as load_model,
    recommend_params as recommend_params,
    build_network as build_network,
    fit_network as fit_network,
    explain_model as explain_model,
)
from .optimization import (
    gradient_descent as gradient_descent,
    numerical_gradient as numerical_gradient,
    check_gradient as check_gradient,
    learning_rate_schedule as learning_rate_schedule,
    clip_gradients as clip_gradients,
    initialize_weights as initialize_weights,
    sigmoid as sigmoid,
    relu as relu,
    softmax as softmax,
    loss_value as loss_value,
)
from .syntax import (
    check_syntax as check_syntax,
    fix_syntax as fix_syntax,
    suggest_function as suggest_function,
    function_help as function_help,
    list_functions as list_functions,
)
from .model import Model as Model, Sequential as Sequential, History as History, easy as easy
from .layers import Dense as Dense
from .optimizers import SGD as SGD, Adam as Adam
from .device import device as device
from .data import DataSplit as DataSplit
from .training import TrainingResult as TrainingResult
from .optimization import OptimizeResult as OptimizeResult
from .syntax import SyntaxResult as SyntaxResult
from .exceptions import (
    PyBMSError as PyBMSError,
    ConfigurationError as ConfigurationError,
    DataError as DataError,
    ShapeError as ShapeError,
    InvalidLayerError as InvalidLayerError,
    UnknownActivationError as UnknownActivationError,
    UnknownLossError as UnknownLossError,
    UnknownOptimizerError as UnknownOptimizerError,
    ModelNotBuiltError as ModelNotBuiltError,
    ModelNotCompiledError as ModelNotCompiledError,
    TrainingError as TrainingError,
    NotSupportedError as NotSupportedError,
)

__version__: str
__all__: list[str]

def about() -> dict: ...
