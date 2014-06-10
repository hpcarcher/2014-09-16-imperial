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
* Platform independent build tools e.g CMake, Autoconf/Automake:
 * Automatically discover a machine's configuration.
 * Generate platform-dependent Makefiles or Visual Studio project files.
 * Harder to debug.

Automated build:

 * Input files => process => output files.
 * Source code => compiler => library or executable.
 * Configuration and data files => processor => data files.

See [Not just for compiling code](notjustforcode.jpg).

Example:

* Protein database files.
* `awk` script to extract author, atom symbol and coordinates and count total number of atoms.
* Output files with this information.
* It doesn't really matter which programs we are using, could be anything.

See [Data dependencies](datadependencies.jpg).

A first makefile
----------------

Edit `cubane.pdb` and change the author's name.

    ls -l *.pdb *.data

`cubane.pdb.data` is now older than `cubane.pdb` - 'out-of-date' - so needs to be updated.

Write a makefile, `pdbprocess.mk`:

    # PDB atom summarizer.
    cubane.pdb.data : cubane.pdb
        awk -f program.awk cubane.pdb > cubane.pdb.data

Makefile format: 

* `#` - comment.
* Target - 'thing' to be built.
* Dependencies - other 'things' that 'thing' depends upon.
* Actions - commands to run to build the target, or update it.
 * Actions intented using TAB not 8 spaces.
 * Legacy of make's 1970's origins.

Run, using `-f` to specify the makefile. If omitted then a default of `Makefile` is assumed.

    make -f pdbprocess.mk

Make uses 'last modification time' to determine if dependencies are newer than targets.

    make -f pdbprocess.mk

Question: why did nothing happen?

Answer: the target is now up-to-date and newer than its dependency.

Add a rule:

    ethane.pdb.data : ethane.pdb
        awk -f program.awk ethane.pdb > ethane.pdb.data

`touch` updates a file's time-stamp which makes it look as if it's been modified.

    touch ethane.pdb
    make -f pdbprocess.mk

Nothing happens to `ethane.pdb` as the first rule in the makefile, the default rule, is used.

    make -f pdbprocess.mk ethane.pdb.data

Introduce a phony target:

    all : cubane.pdb.data ethane.pdb.data

`all` is not a 'thing' - a file or directory - but depends on 'things' that are, and so can be used to trigger their rebuilding.

    make -f pdbprocess.mk all
    touch cubane.pdb ethane.pdb
    make -f pdbprocess.mk all

See [all phony target](allphonytarget.jpg).

Order of rebuilding dependencies is arbitrary.

A dependency in one rule e.g. `cubane.pdb.data`, can be a target in another.

Dependencies between files must make up a directed acyclic graph.

Exercise 1 - add a rule 
-----------------------

See AutomationMakeExercises.md.

Solution:

    methane.pdb.data : methane.pdb
        awk -f program.awk methane.pdb > methane.pdb.data
    all : cubane.pdb.data ethane.pdb.data methane.pdb.data

Patterns
--------

Replace `all` target with:

    PDBAnalysis.tar.gz : cubane.pdb.data ethane.pdb.data methane.pdb.data
        tar -czf PDBAnalysis.tar.gz cubane.pdb.data ethane.pdb.data methane.pdb.data

Run:

    make -f pdbprocess.mk PDBAnalysis.tar.gz

See [processing dependencies](processingdependencies.jpg).

Duplication and repeated code creates maintainability issues. Makefiles are a type of code.

Rewrite action:

    tar -czf $@ cubane.pdb.data ethane.pdb.data methane.pdb.data

`$@` means 'the target of the current rule'. It is an 'automatic variable'.

Rewrite action:

    tar -czf $@ $^

`$^` means 'the dependencies of this rule'.

Bash wild-card can be used in file names. Replace dependencies with:

    PDBAnalysis.tar.gz : *.pdb.data

But watch what happens:

    rm *.pdb.data
    make -f pdbprocess.mk PDBAnalysis.tar.gz

Question: any guesses as to why this is?

Answer: there are no files that match `*.pdb.data` so the name `*.pdb.data` is used as-is.

Create `.data` files in a more manual way:

    make -f pdbprocess.mk cubane.pdb.data
    make -f pdbprocess.mk methane.pdb.data
    make -f pdbprocess.mk ethane.pdb.data

Dependencies on data and code
-----------------------------

Output data is not just dependent upon input data but also programs that create it. `.pdb.data` files are dependent upon `program.awk`.

    cubane.pdb.data : cubane.pdb program.awk
    ...
    ethane.pdb.data : ethane.pdb program.awk
    ...
    methane.pdb.data : methane.pdb program.awk
     ...

`.pdb` files are input files and have no dependencies. To make these depend on `program.awk` would introduce a 'false dependency'.

    touch program.awk
    make -f pdbprocess.mk

Pattern rules
-------------

Makefile still has duplicated and repeated content. Where?

Replace `.pdb.data` targets and dependencies with a single target and dependency:

    %.pdb.data : %.pdb program.awk

`%` is a Make wild-card and this rule is termed a 'pattern rule'.

Exercise 2 - simplify a rule 
----------------------------

See AutomationMakeExercises.md.

Solution: 

    %.pdb.data : %.pdb program.awk
        awk -f program.awk $^ > $@

Question: what did you notice when running your makefile?

Answer: `program.awk` is included in the processing because `$^` matches all dependencies.

More on patterns
----------------

Change pattern rule action to:

        awk -f program.awk $< > $@

`$<` means 'use the first dependency only'.

    touch program.awk
    make -f pdbprocess.mk

Exercise 3 - gzip the files
---------------------------

See AutomationMakeExercises.md.

Solution:

    PDBAnalysis.tar.gz : *.pdb.data.gz
        tar -czf $@ $^

    %.pdb.data.gz : %.pdb.data
        gzip -c $< > $@

Question: why do the `.gz` files need to be manually created first?

Answer: if `*.pdb.data.gz` doesn't match any existing files, then it's left as-is and it percolates down and the result is a single `*.pdb.data.gz` file.

What make will do
-----------------

If unsure of what make would do:

    make -n

Displays commands that make would run.

Macros
------

Add the program to the archive:

    PDBAnalysis.tar.gz : *.pdb.data program.awk

Question: there's still duplication in our makefile, where?

Answer: the program name is cited all over the place? Suppose the name of our program changes?

Use a 'macro', a Make variable:

    PROCESSOR=program.awk

Replace `program.awk` in each rule with `$(PROCESSOR)`.

`awk` is a program too. Another user might only have `gawk`:

    AWKPROG=awk 

Replace `awk` in each rule with `$(AWKPROG)`.

Keep macros at the top of a Makefile so they are easy to find. Or put in another file.

Move the macros to `config.mk`:

    # PDB atom summarizer configuration.
    PROCESSOR=program.awk
    AWKPROG=awk

Read the macros into the makefile:

    include config.mk

A separate configuration allows for one makefile with many configurations:

* No need to edit the makefile which reduces risk of introducing a bug.
* Separates program (makefile) from its data.

Programs that use configuration files, rather than configuration values hard-coded throughout the code, are more modular, flexible and usable. This applies not just to makefiles but any program.

Exercise 4 - create a macro
---------------------------

See AutomationMakeExercises.md.

Solution:

`config.mk`:

    TARFILE=PDBAnalysis

`pdbprocess.mk`:

    $(TARFILE).tar.gz : *.pdb.data.gz $(PROCESSOR)
        tar -czf $@ $^

Completed makefile and configuration file
-----------------------------------------

Makefile, `pdbprocess.mk`:

    # PDB atom summarizer.
    include config.mk

    $(TARFILE).tar.gz : *.pdb.data.gz $(PROCESSOR)
        tar -czf $@ $^

    %.pdb.data.gz : %.pdb.data
        gzip -c $< > $@

    %.pdb.data : %.pdb $(PROCESSOR)
        $(AWKPROG) -f $(PROCESSOR) $< > $@

Configuration file, `config.mk`:

    # PDB atom summarizer configuration.

    PROCESSOR=program.awk
    AWKPROG=awk
    TARFILE=PDBAnalysis

Conclusion
----------

* Automates repetitive tasks
* Reduces errors
* Frees up time to do research
* Documents:
 * How your software is built
 * How your data is created
 * How your reports are formed
 * Dependencies between your analysis programs, input and configuration data, and output data
* Build files are programs
 * Use meaningful variable names
 * Provide useful comments
 * Separate configuration from computation
 * Keep under revision control
 * Treat them with same respect you give any program
