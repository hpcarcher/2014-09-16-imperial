
Bash Shell
==========

A number of shell flavours e.g. ksh, csh so some specifics may differ though the fundamental concepts are the same.

Navigation
----------

    $ pwd    # Absolute path to current directory
    $ cd /   # Root directory
    $ cd ~   # Home directory - there's no place like ~
    $ ls -a  # Hidden files
    $ ls .   # Current directory
    $ ls ..  # Parent directory

Auto-completion (tab completion)
---------------

    $ gre    # Press TAB to see auto-completion options
    $ ls ma  # Press TAB to see auto-completion options for the file

Command history
---------------

Move back and forward through commands executed:

* `CTRL-P` or up arrow - move back to older command.
* `CTRL-N` or down arrow - move forward to newer command.

Edit a command:

* `CTRL-B` or left arrow - move left in line.
* `CTRL-F` or right arrow - move right in line.
* `CTRL-A` - jump to start of line.
* `CTRL-E` - jump to end of line.

Avoid having to retype commands.

Save time.

Input and output redirection
----------------------------

    $ ls books/*.txt > txt_files.txt  # > redirects output (AKA standard output)
    $ cat txt_files.txt
    $ wc books/*.txt > words.txt
    $ cat words.txt
    $ cat > myscript.txt  # Echo standard input and redirect
    Blah
    CTRL-D

    $ ls *.cfg > output.txt
    $ cat output.txt

Question: Why is this empty?

Answer: outputs and errors happen on two different streams.

    $ ls *.cfg 2> output.txt  # 2 is standard error
    $ ls books/*.txt 1> output.txt  # 1 is standard output
    $ ls *.cfg *.txt *.png > output.txt 2>&1

    $ ./interactive.sh
    $ cat config.properties
    $ ./interactive.sh < config.properties  # < redirects input (AKA standard input)
    $ ./interactive.sh < config.properties > out.txt 2>&1

Backticks
---------

Everything inside backticks is executed before the current command. The output is used within the current command:

    $ FILES=`ls books/*.txt`  # FILES is a shell variable.
    $ echo $FILES
    $ HOST=`hostname`
    $ echo HOST
    $ WHEREIWAS=`pwd`
    $ cd /
    $ cd $WHEREIWAS

Power of the pipe
-----------------

Count text files:

    $ find . -name '*.txt' > files.tmp
    $ wc -l files.tmp

`find` outputs a list of files, `wc` inputs a list of files. Skip the need for the temporary file:

    $ find . -name '*.txt' | wc -l  # | is a pipe
    $ echo "Number of .txt files:" ; find . -name '*.txt' | wc -l # ; runs each command separately

Question: what does this do?

    $ ls | grep s | wc -l

Answer: count the number of files with `s` in their name.

Pipe demonstrates principles of good programming practice:

* Power of modular components with well-defined interfaces.
* High cohesion - degree to which elements of a component belong together.
* Low coupling - degree to which a component depends on other components.
* "little pieces loosely joined".
* Bolt together to create powerful computational and data processing workflows.
* `history` + `grep` = function to search for a command.
* Applies to C functions and libraries, FORTRAN sub-routines and modules, Java packages, classes and methods, Python functions and classes etc.

Command history revisited
-------------------------

    $ history
    $ !NNNN  # Rerun Nth command in history
    $ history | grep 'wget'
    $ CTRL-R
    Type letter(s). CTRL-R to go
    (reverse-i-search)`;
    $ fc -l N     # Display command 10 onwards
    $ fc -l M N   # Display commands 10 to 20
    $ fc -l ssh   # Display commands from last 'ssh' command
    $ history -c  # Clear history e.g. you accidently type your password

Avoid having to up-arrow through 100s of commands.

Save time.

`source` versus `sh`
--------------------

    $ cat variables.sh
    $ ./variables.sh
    $ echo $EXAMPLE_DIR
    $ sh variables.sh
    $ echo $EXAMPLE_DIR

These spawn a new shell, run the commands and shut down the new shell. This can be problematic if setting variables in the current shell.

    $ source variables.sh
    $ echo $EXAMPLE_DIR

Packaging
---------

    $ mkdir tmp
    $ cd tmp
    $ cp ../books/*.txt .
    $ tar -cvzf books.tar.gz *txt  # TAR Create Verbose, TAR File, gZip
    $ rm *.txt
    $ tar -xvf books.tar.gz  # eXtract

Put content in a directory then zip or tar up that single directory so it doesn't overwite someone else's files when unpacked.

Put the version number or a date in the bundle name. If someone asks for advice, you'll know exactly what version they have.

    $ cd ..
    $ tar -cvzf books.tar.gz books
    $ mkdir unpack-nice
    $ cd unpack-nice
    $ mv ../books.tar.gz .
    $ tar -xvf books.tar.gz  # eXtract

    $ ls -l books.tar.gz
    $ md5sum books.tar.gz  # MD5 checksum

For files online, file size and MD5 sum allow others to check the files have not been tampered with.

Jobs
----

    $ count.sh > count1.out &  # Start a job in background
    $ count.sh > count2.out &
    $ count.sh > count3.out &
    $ cat count1.out
    $ cat count1.out
    $ cat count1.out
    $ jobs -l  # Current jobs + is current, - is previous
    $ ps  # Processes across all shells
    $ fg 2  # Bring job to foreground
    $ CTRL-Z  # Suspend job - not on GitBash :-(
    $ jobs -l
    $ bg 2  # Restart job in background 
    $ jobs -l
    $ fg 1 
    $ CTRL-C
    $ jobs -l
    $ kill %2 # Kill job with given job number
    $ kill %3
    $ jobs -l

Script (not on GitBash)
------

    $ script
    $ ls -l
    $ CTRL-D
    $ cat typescript

Record commands typed, commands with lots of outputs, trial-and-error when building software.

Add to lab notebook.

Send exact copy of command and error message to support or paste into a ticket.

Rework into a blog or tutorial.

bash_profile and .bashrc
------------------------

Useful for:

* Creating aliases.
* Setting user or application specirfic environment variables.
* Updating standard library and execution paths e.g. `PATH`.

    $ nano ~/bash_profile
    echo "Running .bash_profile"
    $ nano ~/bashrc
    echo "Running .bashrc"
    $ bash
    $ CTRL-D

`.bash_profile` is read when a login, shell is created.

`.bashrc` is read when an interactive, but non-login, shell is created.

Distinction is important when running applications that spawn new shells e.g. `mpiexec.hydra`.

Other shells have their own equivalents (e.g. `.profile`).

Shell power
-----------

Common words problem: 

* Read a text file.
* Identify the N most frequently-occurring words.
* Print out a sorted list of the words with their frequences.

Bentley, Knuth, McIlroy (1986) Programming pearls: a literate program Communications of the ACM, 29(6), pp471-483, June 1986 [doi:10.1145/5948.315654](http://dx.doi.org/10.1145/5948.315654)

Dr. Drang (2011) [More shell, less egg](http://www.leancrew.com/all-this/2011/12/more-shell-less-egg/), 4 December 2011. 

10 plus pages of Pascal ... or ... 1 line of shell:

    $ cat wordcount.sh
    $ ./wordcount.sh < books/war.txt
    $ ./wordcount.sh < books/war.txt 10

"A wise engineering solution would produce, or better, exploit-reusable parts." - Doug McIlroy
