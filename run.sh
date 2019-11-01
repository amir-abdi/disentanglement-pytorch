#!/bin/bash

# Root is where this file is.
export NDC_ROOT="$( cd "$(dirname "$0")" ; pwd -P )"
echo NDC_ROOT=$NDC_ROOT

# Source the training environment (see the env variables defined therein) if we are not evaluating
if [ ! -n "${AICROWD_IS_GRADING+set}" ]; then
  echo "AICROWD_IS_GRADING is not set and you're running locally."
  echo "This script (run.sh) is designed for runs on the AIcrowd server."
  echo "Please run your favourite script from the scripts/ directory"
  exit 1
else
  # We're on the AICrowd evaluator. Add root to python path
  export PYTHONPATH=${PYTHONPATH}:${NDC_ROOT}
fi

bash ${NDC_ROOT}/scripts/aicrowd_challenge.sh
