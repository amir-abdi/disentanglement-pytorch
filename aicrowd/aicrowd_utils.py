import os
import numpy as np
import gin
import logging

from aicrowd import utils_pytorch
from disentanglement_lib.config.unsupervised_study_v1 import sweep as unsupervised_study_v1
import gin.tf
from disentanglement_lib.data.ground_truth import named_data


def is_on_aicrowd_server():
    on_aicrowd_server = os.getenv('AICROWD_IS_GRADING', False)
    on_aicrowd_server = True if on_aicrowd_server != False else on_aicrowd_server
    return on_aicrowd_server


def get_gin_config(config_files, metric_name):
    for gin_eval_config in config_files:
        metric_name_of_config = gin_eval_config.split("/")[-1].replace(".gin", "")
        if metric_name == metric_name_of_config:
            return gin_eval_config
    return None


@gin.configurable("evaluation")
def _evaluate(representation_function, dataset, evaluation_fn, random_seed, name):
    del name
    results = evaluation_fn(
        dataset,
        representation_function,
        random_state=np.random.RandomState(123),
    )
    return results


def evaluate_disentanglement_metric(model, metric_name='mig', dataset_name='mpi3d_toy'):
    from disentanglement_lib.evaluation.metrics import beta_vae  # pylint: disable=unused-import
    from disentanglement_lib.evaluation.metrics import dci  # pylint: disable=unused-import
    from disentanglement_lib.evaluation.metrics import downstream_task  # pylint: disable=unused-import
    from disentanglement_lib.evaluation.metrics import factor_vae  # pylint: disable=unused-import
    from disentanglement_lib.evaluation.metrics import irs  # pylint: disable=unused-import
    from disentanglement_lib.evaluation.metrics import mig  # pylint: disable=unused-import
    from disentanglement_lib.evaluation.metrics import modularity_explicitness  # pylint: disable=unused-import
    from disentanglement_lib.evaluation.metrics import reduced_downstream_task  # pylint: disable=unused-import
    from disentanglement_lib.evaluation.metrics import sap_score  # pylint: disable=unused-import
    from disentanglement_lib.evaluation.metrics import unsupervised_metrics  # pylint: disable=unused-import

    expected_evaluation_metrics = [
        'dci',
        'factor_vae_metric',
        'sap_score',
        'mig',
        'irs'
    ]
    if metric_name not in expected_evaluation_metrics:
        logging.warning('metric {} not among available metrics: {}'.format(metric_name, expected_evaluation_metrics))
        return 0

    _study = unsupervised_study_v1.UnsupervisedStudyV1()
    evaluation_configs = sorted(_study.get_eval_config_files())
    evaluation_configs.append(os.path.join(os.getenv("PWD", ""), "extra_metrics_configs/irs.gin"))
    # eval_bindings = [
    #     "evaluation.random_seed = {}".format(0),
    #     "evaluation.name = '{}'".format(metric_name)
    # ]

    # Get the correct config file and load it
    my_config = get_gin_config(evaluation_configs, metric_name)
    if my_config is None:
        logging.warning('metric {} not among available configs: {}'.format(metric_name, evaluation_configs))
        return 0
    gin.parse_config_file(my_config)

    model_path = os.path.join(model.ckpt_dir, 'temporary_evaluate.pt')
    utils_pytorch.export_model(utils_pytorch.RepresentationExtractor(model.model.encoder, 'mean'),
                               input_shape=(1, model.num_channels, model.image_size, model.image_size),
                               path=model_path)
    model = utils_pytorch.import_model(path=model_path)
    representation_function = utils_pytorch.make_representor(model)

    dataset = named_data.get_named_ground_truth_data(dataset_name)

    results = _evaluate(representation_function, dataset)
    gin.clear_config()
    return results
