"""
Borrowed from https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit/blob/master/local_evaluation.py
"""
# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
# from disentanglement_lib.evaluation import evaluate
import disentanglement_lib

try:
    # Monkey patch in our own evaluate, which supports pytorch *and* tensorflow.
    from aicrowd import evaluate

    disentanglement_lib.evaluation.evaluate = evaluate
    MONKEY = True
except ImportError:
    # No pytorch, no problem.
    print('******* exception caught')
    MONKEY = False
print('Monkey', MONKEY)

from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
from disentanglement_lib.visualize import visualize_model
from disentanglement_lib.config.unsupervised_study_v1 import sweep as unsupervised_study_v1
import tensorflow as tf
import gin.tf
import json
import numpy as np

##############################################################################
# 0. Settings
# By default, we save all the results in subdirectories of the following path.
##############################################################################
base_path = os.getenv("AICROWD_OUTPUT_PATH", "./scratch/shared")
experiment_name = os.getenv("AICROWD_EVALUATION_NAME", "experiment_name")
DATASET_NAME = "auto"
overwrite = True
experiment_output_path = os.path.join(base_path, experiment_name)
ROOT = os.getenv("NDC_ROOT", ".")

# Print the configuration for reference
if not MONKEY:
    print("Evaluating Experiment '{experiment_name}' from {base_path}.")
else:
    from aicrowd import utils_pytorch

    exp_config = utils_pytorch.get_config()
    print("Evaluating Experiment '{exp_config.experiment_name}' "
          "from {exp_config.base_path} on dataset {exp_config.dataset_name}")


# ----- Helpers -----


def get_full_path(filename):
    return os.path.join(ROOT, filename)


##############################################################################
# Gather Evaluation Configs | Compute Metrics
##############################################################################
_study = unsupervised_study_v1.UnsupervisedStudyV1()
evaluation_configs = sorted(_study.get_eval_config_files())
# Add IRS
evaluation_configs.append(get_full_path("extra_metrics_configs/irs.gin"))

# Compute individual metrics
expected_evaluation_metrics = [
    'dci',
    'factor_vae_metric',
    'sap_score',
    'mig',
    'irs'
]

for gin_eval_config in evaluation_configs:
    metric_name = gin_eval_config.split("/")[-1].replace(".gin", "")
    if metric_name not in expected_evaluation_metrics:
        # Ignore unneeded evaluation configs
        continue
    print("Evaluating Metric : {}".format(metric_name))
    result_path = os.path.join(
        experiment_output_path,
        "metrics",
        metric_name
    )
    representation_path = os.path.join(
        experiment_output_path,
        "representation"
    )
    eval_bindings = [
        "evaluation.random_seed = {}".format(0),
        "evaluation.name = '{}'".format(metric_name)
    ]
    evaluate.evaluate_with_gin(
        representation_path,
        result_path,
        overwrite,
        [gin_eval_config],
        eval_bindings
    )

# Gather evaluation results
evaluation_result_template = "{}/metrics/{}/results/aggregate/evaluation.json"
final_scores = {}
sum_scores = 0
for _metric_name in expected_evaluation_metrics:
    evaluation_json_path = evaluation_result_template.format(
        experiment_output_path,
        _metric_name
    )
    evaluation_results = json.loads(
        open(evaluation_json_path, "r").read()
    )
    if _metric_name == "factor_vae_metric":
        _score = evaluation_results["evaluation_results.eval_accuracy"]
        final_scores["factor_vae_metric"] = _score
    elif _metric_name == "dci":
        _score = evaluation_results["evaluation_results.disentanglement"]
        final_scores["dci"] = _score
    elif _metric_name == "mig":
        _score = evaluation_results["evaluation_results.discrete_mig"]
        final_scores["mig"] = _score
    elif _metric_name == "sap_score":
        _score = evaluation_results["evaluation_results.SAP_score"]
        final_scores["sap_score"] = _score
    elif _metric_name == "irs":
        _score = evaluation_results["evaluation_results.IRS"]
        final_scores["irs"] = _score
    else:
        raise Exception("Unknown metric name : {}".format(_metric_name))
    sum_scores += _score

print("Final Scores : ", final_scores)
print('sum_scores : ', sum_scores)

##############################################################################
# (Optional) Generate Visualizations
##############################################################################
# model_directory = os.path.join(experiment_output_path, "model")
# visualize_model.visualize(model_directory, "viz_output/")
