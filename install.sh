#!/bin/bash
systemName=$(uname -a)

echo =======================================================================================
echo Install Solver for Climate Project
echo =======================================================================================

echo Starting installation process...
if [[ $systemName == *"Darwin"* ]]; then
  export MACOSX_DEPLOYMENT_TARGET=12.0
fi
echo ===============================================================================
echo Step 1: Install numba and pybind11
pip3 install pybind11
pip3 install plotly

echo ===============================================================================
echo Step 2: Install model solution core
pip3 install ./src/cppcore
echo ===============================================================================
echo Step 3: Install C kernels for petsc4py
pip3 install ./src/linearsystemcore

