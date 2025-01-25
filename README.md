# Pykaleidoscope

<p align="center">
  <img alt="Logo" src="doc/kal256.png" />
</p>

## What is this?

LLVM comes with a [great tutorial](http://llvm.org/docs/tutorial/) that
builds a compiler for a simple language called Kaleidoscope. The
compiler parses Kaleidoscope into an AST, from which LLVM code is then
generated using the LLVM IR building APIs. Once we have LLVM IR, it can
be JITed to generate machine code and run it. In other words, convert
your language into LLVM IR and leave the rest to LLVM itself (including
world-class optimizations).

The tutorial is presented in several \"chapters\" that start with a
simple lexer and build up the language step by step.

This repository contains a chapter-by-chapter translation of the LLVM
tutorial into Python, using the
[llvmlite](https://github.com/numba/llvmlite) package that exposes LLVM
to Python.

This repository is fairly complete - the whole Kaleidoscope language is
implemented. The only thing missing is Chapter 9 - Adding Debug
Information, because `llvmlite` does not yet support convenient emission
of debug info.

Note: the majority of the Python code in this repository was written
in 2015, so it may not reflect the very latest modern practices. However,
it was designed to be simple and readable, and has been tested to work
well on the latest Python versions as of 2025.

## How to use this code

Go through the [LLVM tutorial](http://llvm.org/docs/tutorial/). The
files in this repository are named after tutorial chapters and roughly
correspond to the C++ code presented in the tutorial. In each source
file, the `__main__` section of code in the bottom is a small sample of
usage, and there are also unit tests that check a variety of cases.

To run the code for a specific chapter, I recommend using `uv`, for example:

    uv run chapter6.py

## Testing

Running unit tests for a single chapter or all of them:

    uv run pytest chapter6.py
    uv run pytest *.py

## Setting up llvmlite

This repository was updated in early 2025 to use
[uv](https://github.com/astral-sh/uv). `llvmlite` now has binary wheels on PyPI
that include the right LLVM shared objects; therefore, setting it up is much
easier. When you clone this project, all you should need is to have `uv`
installed, and then use `uv run ...` as mentioned above.

Check out the git history of this repo for old instructions of setting up
`llvmlite` and `LLVM` manually (back from when Pykaleidoscope was originally
written, in 2015).
