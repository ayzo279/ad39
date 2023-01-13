#!/usr/bin/env bash
# File       : m2-install.sh
# Description: Milestone 2 driver script to import ad39_package into a virtual environment

python3 -m venv test_env
source test_env/bin/activate

python -m pip install numpy
git clone git@code.harvard.edu:CS107/team39.git

# Run tests upon cloning (optional)
cd team39/tests
./run_tests.sh pytest
cd ..

cd src
python3
