Automation and Make
===================

Question: what are the problems with this?

    cc -o scheduler main.o schedule.o optimise.o io.o utils.o -I./include -L./lib -lm -lblas -llapack -lrng

Answer:

* Type a lot.
* Remember syntax, flags, inputs, libraries, dependencies.
* Ensure .o files have been created.

Automate - "write once, run many":

* Reduce retyping.
* Document syntax, flags, inputs, libraries, dependencies.
* Recreate files - binaries, output data, graphs - only when needed.
* Input files => process => output files.
* Source code => compiler => library or executable.
* Configuration and data files => analysis => data files.
* Data files => visualisation => images.

Make:

* Widely-used, fast, free, well-documented, build tool.
* Stuart Feldman, Bell Labs 1977 summer intern to Vice President of Computer Science at IBM Research and Google ACM Software System Award, 2003.

Others:

* Apache ANT, Maven, Python doit.
* Platform independent build tools e.g CMake, Autoconf/Automake generate platform-dependent build scripts e.g. Make, Visual Studio project files etc.

Data processing pipeline
------------------------

Computational fluid dynamics (CFD) example:

* Study of the mechanics of fluid flow, liquids and gases in motion.
* Determine the flow pattern of a fluid in a cavity.
* Cavity is a square box with an inlet on one side and an outlet on another - [image](./Cavity.png).

Determine the flow pattern:

    cat configs/32x32.cfg
    python cfd.py configs/32x32.cfg 32x32.dat
    head 32x32.dat

Plot the flow pattern:

    python plot_flow.py 32x32.dat 32x32.jpg

Makefile
--------

    python cfd.py configs/32x32.txt 32x32.dat
    head 32x32.dat

    touch configs/32x32.cfg         # Update time-stamp - mock update
    ls -l configs/32x32.cfg 32x32.dat

Output `32x32.dat` is now older than input `configs/32x32.cfg`, so needs update.

Question: we could write a shell script but what might be the problems?

Answer: if many sources to compile or data to analyse, don't want to update everything, just those that need updated.

Create `Makefile`:

    # Determine flow pattern
    32x32.dat : configs/32x32.cfg
	    python cfd.py configs/32x32.cfg 32x32.dat

Add Makefile format information as comments:

    # Make comments
    # target: dependency1 dependency1 dependency2 ...
    # TAB rule1
    # TAB rule2
    # TAB rule3
    # TAB ...

* Target - 'thing' to be built.
* Dependencies - other 'things' that 'thing' depends upon.
* Actions - commands to run to build the target, or update it.
* Actions indented using TAB, not 8 spaces. Legacy of 70's origins.

<p/>

    make              # Use default Makefile
    make -f Makefile  # Use named makefile
    make

Question: why did nothing happen?

Answer: the target is now up-to-date and newer than its dependency. Make uses a file's 'last modification time'.

    64x64.dat : configs/64x64.cfg
        python cfd.py configs/64x64.cfg 64x64.dat

`touch` updates a file's time-stamp which makes it look as if it's been modified.

    touch configs/64x64.cfg
    make

Nothing happens as the first, default, rule in the makefile, is used.

    make 64x64.dat

Phony targets:

    .PHONY : all
    all : 32x32.dat 64x64.dat

`all` is not a file or directory but depends on files and directories, so can trigger their rebuilding.

A dependency in one rule can be a target in another.

    make all
    touch configs/32x32.cfg configs/64x64.cfg
    make all

Order of rebuilding dependencies is arbitrary.

Dependencies must make up a directed acyclic graph.

Exercise 1 - add a rule 
-----------------------

See [exercises](MakeExercises.md).

Solution:

    96x96.dat : configs/96x96.cfg
        python cfd.py configs/96x96.cfg 96x96.dat

    all : 32x32.dat 64x64.dat 96x96.dat

Patterns
--------

    analysis.tar.gz : 32x32.dat 64x64.dat 96x96.dat
        tar -czf analysis.tar.gz 32x32.dat 64x64.dat 96x96.dat

<p/>

    make analysis.tar.gz

Makefiles are code. Repeated code creates maintainability issues. 

    tar -czf $@ 32x32.dat 64x64.dat 96x96.dat

<p/>

    # Make's special macros and notation:
    # $@ Target of the current rule.

<p/>

    tar -czf $@ $^

<p/>

    # $^ All dependencies of the current rule.

Wild-cards can be used in file names:

    analysis.tar.gz : *.dat

<p/>

    make analysis.tar.gz
    touch *.dat
    make analysis.tar.gz

    rm *.dat
    make analysis.tar.gz

Question: any guesses as to why this is?

Answer: there are no files that match `*.dat` so the name `*.dat` is used as-is.

    make all

Dependencies on data and code
-----------------------------

Output data depends on both input data and programs that create it:

    32x32.data : configs/32x32.cfg cfd.py
    ...
    64x64.dat : configs/64x64.cfg cfd.py
    ...
    96x96.dat : configs/96x96.cfg cfd.py
    ...

<p/>

    touch cfd.py
    make all

`.cfg` files are input files and have no dependencies. To make these depend on `cfd.py` would introduce a 'false dependency'.

Pattern rules
-------------

Question: Makefile still has repeated content. Where?

Answer: the rules for each .dat file.

    %.dat : configs/%.cfg cfd.py

    # % - Make pattern.

Exercise 2 - simplify a rule 
----------------------------

See [exercises](MakeExercises.md).

You will need another special macro:

    # $< First dependency of the current rule.

Solution: 

    %.dat : configs/%.cfg cfd.py
	    python cfd.py $< $@

Macros
------

    analysis.tar.gz : *.dat cfd.py
	tar -czf $@ $^

Question: there's still duplication in our makefile, where?

Answer: the program name. Suppose the name of our program changes?

    CFD_SRC=cfd.py

Question: is there an alternative to this?

Answer: we might change our programming language or the way in which our command is invoked:

    CFD_SRC=cfd.py
    CFD_EXE=python $(CFD_SRC)
    
Exercise 3 - use a macro
------------------------

See [exercises](MakeExercises.md).

Solution:

    # CFD analysis
    CFD_SRC=cfd.py
    CFD_EXE=python $(CFD_SRC)

    %.dat : configs/%.cfg $(CFD_SRC)
        $(CFD_EXE) $< $@

    analysis.tar.gz : *.dat $(CFD_SRC)
        tar -czf $@ $^

Keep macros at the top of a Makefile so they are easy to find, or move to `config.mk`:

    # CFD analysis
    CFD_SRC=cfd.py
    CFD_EXE=python $(CFD_SRC)

<p/>

    include config.mk

Good programming practice:

* Separate code from data.
* No need to edit code which reduces risk of introducing a bug.
* Code that is configurable is more modular, flexible and reusable.

What make will do
-----------------

    touch configs/*.cfg
    make -n analysis.tar.gz  # Display commands make will run

Exercise 4 - add another processing stage
-----------------------------------------

See [exercises](MakeExercises.md).

Solution:

Makefile, `Makefile`:

    # Plot flow.
    %.jpg : %.dat $(PLOT_SRC)
        $(PLOT_EXE) $< $@

    all : 32x32.jpg 64x64.jpg 96x96.jpg

    analysis.tar.gz : *.dat *.jpg $(CFD_SRC) $(PLOT_SRC)
        tar -czf $@ $^

    clean : 
        rm -f analysis.tar.gz
        rm -f *.dat
        rm -f *.jpg

Configuration file, `config.mk`:

    # Plot flow
    PLOT_SRC=plot_flow.py
    PLOT_EXE=python $(PLOT_SRC)

shell and patsubst
------------------

Avoid hard-coding file names:

    CFG_FILES=$(shell find configs -type f -name '*.cfg')
    DAT_FILES=$(patsubst configs/%.cfg, %.dat, $(CFG_FILES))
    JPG_FILES=$(patsubst configs/%.cfg, %.jpg, $(CFG_FILES))

    .PHONY : dats
    dats : $(DAT_FILES)

    .PHONY : jpgs
    jpgs : $(JPG_FILES)

Parallel jobs
-------------

Make can run on multiple cores if available:

    make -j 4 analysis.tar.gz

Conclusion
----------

See [the purpose of Make](MakePurpose.png).

Build scripts:

* Automate repetitive tasks.
* Reduce errors.
* Document how software is built, data is created, graphs are done, papers formed.
* Document dependencies between code, scripts, tools, inputs, configurations, outputs.
* Are code so use meaningful variable names, comments, and separate configuration from computation.
* Should be kept under revision control.
