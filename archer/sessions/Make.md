Automation and Make
===================

Question: what are the problems with this?

    cc -o scheduler main.o schedule.o optimise.o io.o utils.o -I./include -L./lib -lm -lblas -llapack -lrng

Answer:

* Need to type a lot.
* Need to remember syntax.
* Need to remember arguments.
* Need to remember dependencies.
* Need to ensure .o files have been created.

Solution:

* Record all the tricky software options.
* Capture software-based processes and dependencies.
* Regenerate files (binaries, output data) only when needed.
* Automate - "write once, run many".

Make:

* Widely-used build tool.
* Stuart Feldman, Bell Labs, 1977. From summer intern to Vice President of Computer Science at IBM Research and Google ACM Software System Award, 2003.
* Fast, free, well-documented.

Other build tools:

* Apache ANT
* Maven
* Python doit
* Platform independent build tools e.g CMake, Autoconf/Automake:
 * Automatically discover a machine's configuration.
 * Generate platform-dependent Makefiles or Visual Studio project files.
 * Harder to debug.

Automated build:

 * Input files => process => output files.
 * Source code => compiler => library or executable.
 * Configuration and data files => processor => data files.

See [Not just for compiling code](MakeUses.jpg).

Example:

* Text files.
* Python script to read text files output words and their frequencies.
* It doesn't really matter which programs we are using, could be anything.

A simple data processing pipeline
---------------------------------

Count the words in a text file:

    $ python wordcount.py books/war.txt war.dat
    $ head war.dat
    $ python wordcount.py books/abyss.txt abyss.dat
    $ head abyss.dat

Count the words in a text file over a certain length:

    $ python wordcount.py books/war.txt war.dat 12
    $ head war.dat

Plot the first 10 word counts:

    $ python plotcount.py war.dat show
    $ python plotcount.py abyss.dat show

Plot the first N word counts in a data file:

    $ python plotcount.py war.dat show N
    $ python plotcount.py abyss.dat show N

Plot the word counts in a data file and save:

    $ python plotcount.py war.dat war.jpg
    $ python plotcount.py war.dat war.jpg 5

A first makefile
----------------

Type command manually:

    $ python wordcount.py books/war.txt war.dat
    $ head war.dat

Shell script. But:

    $ touch books/war.txt
    $ ls -l books/war.txt war.dat

`war.dat` is now older than `books/war.txt` - 'out-of-date' - so needs to be updated.

If many source files to compile or data files to analyse, don't want to reanalyse them all just because one has changed.

Write a makefile, `Makefile`:

    # Calculate word frequencies.
    war.dat : books/war.txt
	    python wordcount.py books/war.txt war.dat

Makefile format: 

* `#` - comment.
* Target - 'thing' to be built.
* Dependencies - other 'things' that 'thing' depends upon.
* Actions - commands to run to build the target, or update it.
 * Actions intented using TAB not 8 spaces.
 * Legacy of make's 1970's origins.

Run:

    $ make

`-f` can name a specific makefile. If omitted, then a default of `Makefile` is assumed.

Make uses 'last modification time' to determine if dependencies are newer than targets.

    $ make

Question: why did nothing happen?

Answer: the target is now up-to-date and newer than its dependency.

Add a rule:

    jekyll.dat : books/jekyll.txt
        python wordcount.py books/jekyll.txt jekyll.dat

`touch` updates a file's time-stamp which makes it look as if it's been modified.

    $ touch books/jekyll.txt
    $ make

Nothing happens to `jekyll.dat` as the first rule in the makefile, the default rule, is used.

    $ make jekyll.dat

Introduce a phony target:

    .PHONY : all
    all : war.dat jekyll.dat

`all` is not a 'thing' - a file or directory - but depends on 'things' that are, and so can be used to trigger their rebuilding.

    $ make all
    $ touch books/war.txt books/jekyll.txt
    $ make all

Order of rebuilding dependencies is arbitrary.

A dependency in one rule e.g. `war.dat`, can be a target in another.

Dependencies between files must make up a directed acyclic graph.

Exercise 1 - add a rule 
-----------------------

See [exercises](MakeExercises.md).

Solution:

    bridge.dat : books/bridge.txt
        python wordcount.py books/bridge.txt bridge.dat

    all : war.dat jekyll.dat bridge.dat

Patterns
--------

Add:

    analysis.tar.gz : war.dat jekyll.dat bridge.dat
        tar -czf analysis.tar.gz war.dat jekyll.dat bridge.dat

Run:

    $ make analysis.tar.gz

Duplication and repeated code creates maintainability issues. Makefiles are a type of code.

Rewrite action:

    tar -czf $@ war.dat jekyll.dat bridge.dat

`$@` means 'the target of the current rule'. It is an 'automatic variable'.

Rewrite action:

    tar -czf $@ $^

`$^` means 'the dependencies of this rule'.

Bash wild-card can be used in file names. Replace dependencies with:

    analysis.tar.gz : *.dat

    $ make analysis.tar.gz
    $ touch *.dat
    $ make analysis.tar.gz

But watch what happens:

    $ rm *.dat
    $ make analysis.tar.gz

Question: any guesses as to why this is?

Answer: there are no files that match `*.dat` so the name `*.dat` is used as-is.

Create `.data` files in a more manual way:

    $ make war.dat jekyll.dat bridge.dat

Dependencies on data and code
-----------------------------

Output data is not just dependent upon input data but also programs that create it. `.dat` files are dependent upon `wordcount.py`.

    war.data : books/war.txt wordcount.py
    ...
    jekyll.dat : books/jekyll.txt wordcount.py
    ...
    bridge.dat : books/bridge.txt wordcount.py
     ...

`.txt` files are input files and have no dependencies. To make these depend on `python.py` would introduce a 'false dependency'.

    $ touch wordcount.py
    $ make all

Pattern rules
-------------

Makefile still has duplicated and repeated content. Where?

Replace `.dat` targets and dependencies with a single target and dependency:

    %.dat : books/%.txt wordcount.py

`%` is a Make wild-card and this rule is termed a 'pattern rule'.

Exercise 2 - simplify a rule 
----------------------------

See [exercises](MakeExercises.md).

You will need an automatic variable `$<` which means 'use the first dependency only'.

Solution: 

    %.dat : books/%.txt wordcount.py
	    python wordcount.py $< $@

Macros
------

Add the program to the archive:

    analysis.tar.gz : *.dat wordcount.py
	tar -czf $@ $^

Question: there's still duplication in our makefile, where?

Answer: the program name. Suppose the name of our program changes?

Use a 'macro', a Make variable:

    PROCESSOR=wordcount.py

Exercise 3 - use a macro
------------------------

See [exercises](MakeExercises.md).

Solution:

    PROCESSOR=wordcount.py

    # Calculate word frequencies.
    %.dat : books/%.txt $(PROCESSOR)
        python $(PROCESSOR) $< $@

    analysis.tar.gz : *.dat $(PROCESSOR)
        tar -czf $@ $^

Keep macros at the top of a Makefile so they are easy to find. Or put in another file.

Move the macros to `config.mk`:

    # Word frequency calculations.
    PROCESSOR=wordcount.py

Read the macros into the makefile:

    include config.mk

A separate configuration allows for one makefile with many configurations:

* No need to edit the makefile which reduces risk of introducing a bug.
* Separates program (makefile) from its data.

Programs that use configuration files, rather than configuration values hard-coded throughout the code, are more modular, flexible and usable. This applies not just to makefiles but any program.

What make will do
-----------------

If unsure of what make would do:

    $ touch books/*.txt
    $ make -n analysis.tar.gz

Displays commands that make would run.

Exercise 4 - add another processing stage
-----------------------------------------

See [exercises](MakeExercises.md).

Solution:

Makefile, `Makefile`:

    # Calculate images
    %.jpg : %.dat $(PLOTTER)
        python $(PLOTTER) $< $@

    analysis.tar.gz : *.dat *.jpg $(PROCESSOR)
        tar -czf $@ $^

    clean : 
        rm -f analysis.tar.gz
        rm -f *.dat
        rm -f *.jpg

Configuration file, `config.mk`:

    # Image output calculations
    PLOTTER=plotcount.py

To recreate all `.dat` and `.jpg` files:

    TXT_FILES=$(shell find books -type f -name '*.txt')
    DAT_FILES=$(patsubst books/%.txt, %.dat, $(TXT_FILES))
    JPG_FILES=$(patsubst books/%.txt, %.jpg, $(TXT_FILES))

    .PHONY : dats
    dats : $(DAT_FILES)

    .PHONY : jpgs
    jpgs : $(JPG_FILES)

Conclusion
----------

See [the purpose of Make](MakePurpose.png).

Build tools:

* Automate repetitive tasks.
* Reduce errors.
* Free up time to do research.

Built scripts document:

* How your software is built.
* How your data is created.
* How your reports are formed.
* Dependencies between your analysis programs, input and configuration data, and output data.

Build scripts are programs:

* Use meaningful variable names.
* Provide useful comments.
* Separate configuration from computation.
* Keep under revision control.
* Treat them with same respect you give any program.
