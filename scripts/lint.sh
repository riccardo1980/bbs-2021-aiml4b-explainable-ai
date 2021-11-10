#!/usr/bin/env bash
set -e

SRCS="myproject"
TEST_SRCS=
TOOLS="notebooks"

[ -d $SRCS ] || (echo "Run this script from project root"; exit 1)

set -x

black $SRCS $TEST_SRCS $TOOLS
isort $SRCS $TEST_SRCS $TOOLS
mypy $SRCS $TEST_SRCS $TOOLS
flake8 $SRCS $TEST_SRCS $TOOLS