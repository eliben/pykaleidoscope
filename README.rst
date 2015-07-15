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
that exposes LLVM to Python. For tips on setting up ``llvmlite`` with LLVM, `see
this blog post
<http://eli.thegreenplace.net/2015/building-and-using-llvmlite-a-basic-example/>`_.

This repository is fairly complete - the whole Kaleidoscope language is
implemented. The only thing missing is Chapter 8 - Debug Information, because
``llvlite`` does not yet support convenient emission of debug info.

How to use this code
--------------------

Go through the `LLVM tutorial <http://llvm.org/docs/tutorial/>`_. The files in
this repository are named after tutorial chapters and roughly correspond to the
C++ code presented in the tutorial. In each source file, the ``__main__``
section of code in the bottom is a small sample of usage, and there are also
unit tests that check a variety of cases.

Version of LLVM, Python and llvmlite
------------------------------------

Last tested with Python 3.4, LLVM 3.6 and top-of-tree ``llvmlite``.

Testing
-------

Some of the files have unit test classes in them. To run all unit tests::

    $ python3.4 -m unittest discover -p "*.py"

The most interesting program written in Kaleidoscope is the Mandelbrot set generator in chapter 6:

.. sourcecode::

    $ python3.4 chapter6.py
    ****************************************************************
    ****************************************************************
    ****************************************++++++******************
    ************************************+++++...++++++**************
    *********************************++++++++.. ...+++++************
    *******************************++++++++++..   ..+++++***********
    ******************************++++++++++.     ..++++++**********
    ****************************+++++++++....      ..++++++*********
    **************************++++++++.......      .....++++********
    *************************++++++++.   .            ... .++*******
    ***********************++++++++...                     ++*******
    *********************+++++++++....                    .+++******
    ******************+++..+++++....                      ..+++*****
    **************++++++. ..........                        +++*****
    ***********++++++++..        ..                         .++*****
    *********++++++++++...                                 .++++****
    ********++++++++++..                                   .++++****
    *******++++++.....                                    ..++++****
    *******+........                                     ...++++****
    *******+... ....                                     ...++++****
    *******+++++......                                    ..++++****
    *******++++++++++...                                   .++++****
    *********++++++++++...                                  ++++****
    **********+++++++++..        ..                        ..++*****
    *************++++++.. ..........                        +++*****
    ******************+++...+++.....                      ..+++*****
    *********************+++++++++....                    ..++******
    ***********************++++++++...                     +++******
    *************************+++++++..   .            ... .++*******
    **************************++++++++.......      ......+++********
    ****************************+++++++++....      ..++++++*********
    *****************************++++++++++..     ..++++++**********
    *******************************++++++++++..  ...+++++***********
    *********************************++++++++.. ...+++++************
    ***********************************++++++....+++++**************
    ***************************************++++++++*****************
    ****************************************************************
    ****************************************************************

