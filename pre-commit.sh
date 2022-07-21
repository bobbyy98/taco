#!/bin/bash

# This script fails when any of its commands fail.
set -e

# clang-format
find src/ -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i
