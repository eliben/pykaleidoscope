Pykaleidoscope
==============

What is this?
-------------

LLVM comes with a `great tutorial <http://llvm.org/docs/tutorial/>`_ that builds
a compiler for a simple language called Kaleidoscope. The compiler parses
Kaleidoscope into an AST, from which LLVM code is then generated using the LLVM
IR building APIs. Once we have LLVM IR, it can be JITed to generate machine code
and run it. In other words, convert your language into LLVM IR and leave the
rest to LLVM itself (including world-class optimizations).

The tutorial is presented in several "chapters" that start with a simple lexer
and build up the language step by step.

This repository contains a chapter-by-chapter translation of the LLVM tutorial
into Python, using the `llvmlite <https://github.com/numba/llvmlite>`_ package
that exposes LLVM to Python.

This repository is fairly complete - the whole Kaleidoscope language is
implemented. The only thing missing is Chapter 9 - Adding Debug Information,
because ``llvmlite`` does not yet support convenient emission of debug info.

How to use this code
--------------------

Go through the `LLVM tutorial <http://llvm.org/docs/tutorial/>`_. The files in
this repository are named after tutorial chapters and roughly correspond to the
C++ code presented in the tutorial. In each source file, the ``__main__``
section of code in the bottom is a small sample of usage, and there are also
unit tests that check a variety of cases.

Testing
-------

Some of the files have unit test classes in them. To run all unit tests::

    $ python3.4 -m unittest discover -p "*.py"

Version of LLVM, Python and llvmlite
------------------------------------

Last tested with Python 3.4, LLVM 3.7 and top-of-tree ``llvmlite``.

Setting up llvmlite
-------------------

Back in January 2015 I wrote a `blog post about setting up llvmlite
<http://eli.thegreenplace.net/2015/building-and-using-llvmlite-a-basic-example>`_,
but as often happens in LLVM-land, things have changed and it may no longer
work.

The easiest way to use llvmlite right now is to download a binary release. If
you can do that, save yourself the trouble and go do that. No need to read any
further :-)

If you insist to build llvmlite on your own, you'll need LLVM. The easiest way
to get LLVM is to grab a binary release from http://llvm.org/releases/. Be sure
to grab a release that llvmlite works with (llvmlite has a correspondence of
versions with the LLVMs supported).

When building llvmlite you'll have to pass in some flags to the ``Makefile``
that gets invoked by the Python setup process:

.. sourcecode:: shell

  $ CXX_FLTO_FLAGS= LD_FLTO_FLAGS= \
    CXX=<path/to/clang++> LLVM_CONFIG=<path/to/llvm-config> \
    python3.4 setup.py build

Where ``path/to`` points to the binaries within the ``bin`` directory of the
untarred LLVM binary release. The reasons for this complication are:

1. Recent versions of Clang are built with themselves (bootstrapped), and
   ``llvm-config`` may have some compiler flags gcc doesn't support, so
   compiling with gcc won't work. We therefore use the same compiler that
   LLVM/Clang was built with to build llvmlite.
2. Clang binary builds don't support LTO, and llvmlite's ``Makefile`` passes
   ``-flto`` when compiling. The ``*_FLTO_FLAGS`` settings are made to avoid
   that.

Note that these directions work at the time of writing (last updated: Mar 15,
2016) and may change with new versions of LLVM and/or llvmlite. I'll try to keep
up but feel free to open issues if anything needs to be done differently.
