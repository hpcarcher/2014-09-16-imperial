Bash Shell
==========

Bourne Again Shell. Syntax differs from other shells (ksh,csh,zsh) but fundamental concepts are the same.

Navigation
----------

    pwd    # Absolute path to current directory
    cd /   # Root directory
    cd ~   # Home directory - there's no place like ~
    ls -a  # Hidden files
    ls .   # Current directory
    ls ..  # Parent directory

Auto-completion (tab completion)
---------------

    gre    # Press TAB to see auto-completion options
    ls ma  # Press TAB to see auto-completion options for the file

Command history
---------------

* `CTRL-P` or up arrow - move back to older command.
* `CTRL-N` or down arrow - move forward to newer command.
* `CTRL-B` or left arrow - move left in line.
* `CTRL-F` or right arrow - move right in line.
* `CTRL-A` - jump to start of line.
* `CTRL-E` - jump to end of line.
* Type - edit command.

Reusing not retyping saves time.

Input and output redirection
----------------------------

    ls books/*.txt > txt_files.txt  # > redirects output (AKA standard output)
    cat txt_files.txt

    wc books/*.txt > words.txt
    cat words.txt

    cat > myscript.txt              # Echo standard input and redirect
    Blah
    CTRL-D

    ls *.cfg > output.txt
    cat output.txt

Question: why is this empty?

Answer: outputs and errors happen on two different streams.

    ls *.cfg 2> output.txt                  # 2 is standard error
    ls books/*.txt 1> output.txt            # 1 is standard output
    ls *.cfg *.txt *.png > output.txt 2>&1  # Capture both standard output and error

    ./interactive.sh
    cat config.properties                   # One line per interactive input
    ./interactive.sh < config.properties    # < redirects input (AKA standard input)
    ./interactive.sh < config.properties > out.txt 2>&1

Backticks
---------

    FILES=`ls books/*.txt`  # Contents of `` are executed before the enclosing command.
    echo $FILES
    for FILE in $FILES; do echo $FILE; done

    HOST=`hostname`
    echo HOST

    WHEREIWAS=`pwd`
    cd /
    cd $WHEREIWAS

Power of the pipe
-----------------

Count text files:

    find . -name '*.txt' > files.tmp  # find outputs list of files
    wc -l files.tmp                   # wc inputs list of files

Avoid temporary file:

    find . -name '*.txt' | wc -l      # | is a pipe

Question: what does this do?

    ls | grep s | wc -l

Answer: count the number of files with `s` in their name.

Good programming practice:

* Modular components with well-defined interfaces.
* High cohesion - elements belong together.
* Low coupling - depends on other components.
* "little pieces loosely joined" to create computational and data processing workflows.
* `history` + `grep` = 'search-for-command' function.
* Functions, methods, modules, classes, packages, libraries, scripts.

<p/>

    echo "Number of .txt files:" ; find . -name '*.txt' | wc -l  # ; runs each command separately

`tee` and `script`
------------------

    ls -l *.sh | tee log.txt     # Capture standard output mid-pipeline
    cat log.txt

    history | tee raw.txt | grep "tar" | tee filtered.txt
    ls -l *.sh | tee log.txt
    ls -l *.py | tee -a log.txt  # Append
    ls *.txt 2>&1 | tee log.txt

How tee works[ SVG](http://en.wikipedia.org/wiki/Tee_\(command\)#mediaviewer/File:Tee.svg)

    script  # Not GitBash
    ls -l
    CTRL-D
    cat typescript

Provenance:

* Record of commands typed, input parameters, output file names for lab notebook.
* Experiments when using command-line tools.
* Trial-and-error-and-fix when building software.
* Exact copies of commands and error messages for e-mails or bug reports.
* Rework into blogs, tutorials, FAQs.

Command history revisited
-------------------------

    history
    !NNNN  # Rerun Nth command in history
    history | grep 'wget'
    CTRL-R
    Type letter(s). CTRL-R to go
    (reverse-i-search)`;

    fc -l N     # Display command 10 onwards
    fc -l M N   # Display commands 10 to 20
    fc -l ssh   # Display commands from last 'ssh' command

    history -c  # Clear history e.g. you accidently type your password

Reusing not retyping, or up-arrowing through 10s of commands, saves time.

`source` versus `sh`
--------------------

    cat variables.sh
    ./variables.sh
    echo $EXAMPLE_DIR

    sh variables.sh
    echo $EXAMPLE_DIR

Question: why is the variable not set?

Answer: a new shell is spawned, commands are run, the shell is killed.

    source variables.sh  # Run the commands within the current shell
    echo $EXAMPLE_DIR

Kills the current shell if one of the commands is `exit`.

Packaging
---------

    mkdir tmp
    cd tmp
    cp ../books/*.txt .
    tar -cvzf books.tar.gz *txt  # TAR Create Verbose, TAR File, gZip

    rm *.txt
    tar -xvf books.tar.gz        # eXtract ... all over user's current directory!

    cd ..
    cp -r books books-1.1
    tar -cvzf books.tar.gz books-1.1  # ZIP up contents within directory

    mkdir unpack-nice
    cd unpack-nice
    mv ../books-1.1.tar.gz .  
    tar -tvf books-1.1.tar.gz  # lisT, without unpacking
    tar -xvf books-1.1.tar.gz  # eXtract

Security:

    ls -l books-1.1.tar.gz   # File size
    md5sum books-1.1.tar.gz  # MD5 checksum (hash that acts as fingerprint)

Add version number or date to directory and package for provenance.

Jobs
----

    count.sh > count1.out &  # Start a job in background
    count.sh > count2.out &
    count.sh > count3.out &
    cat count1.out
    cat count1.out
    cat count1.out
    jobs -l                   # Current jobs + is current, - is previous
    ps                        # Processes across all shells

    fg 2                      # Bring job to foreground
    CTRL-Z                    # Suspend job - not GitBash
    jobs -l

    bg 2                      # Restart job in background 
    jobs -l
    fg 1 
    CTRL-C
    jobs -l

    kill %2                   # Kill job with given job number
    jobs -l
    ps -A
    kill 3                    # Kill process with given process number
    jobs -l

Executables
-----------

    echo $PATH
    interactive.sh
    cd ..
    interactive.sh
    cd DIRECTORY
    PATH=~:$PATH  # Bash searches PATH for executables
    cd ..
    interactive.sh

    type git # built-in command which describes commands
    type ls # "is hashed" - cached so no need to re-search $PATH
    type python
    type interactive.sh
    type -t python  # Type e.g. "file"
    type -t type    # Type e.g. "builtin"
    type -a python  # All places in $PATH with this command

Wrong version of a compiler, interpreter, tool being used? Check the path.

.bash_profile and .bashrc
-------------------------

Set up aliases, environment variables for user or applications and library paths.

    nano ~/bash_profile
    echo "Running .bash_profile"
    nano ~/bashrc
    echo "Running .bashrc"
    bash
    CTRL-D

Create new login or GitBash shell.

* `.bashrc` is read when an interactive, non-login, shell is created
* `.bash_profile` is also read when a login shell is created
* Keep distinction in mind when running applications that spawn new shells e.g. `mpiexec.hydra`.

Exit codes
----------

    ls books
    echo $?  # 0 for OK
    ls none
    echo $?  # Non-zero for error

Clean up
--------

    rm *.out
    rm *.txt
    rm *.tmp

Little pieces loosely joined
----------------------------

Common words problem: 

* Read a text file.
* Identify the N most frequently-occurring words.
* Print out a sorted list of the words with their frequences.

10 plus pages of Pascal ... or ... 1 line of shell:

    cat wordcount.sh
    ./wordcount.sh < books/war.txt
    ./wordcount.sh < books/war.txt 10

"A wise engineering solution would produce, or better, exploit-reusable parts." - Doug McIlroy

* Bentley, Knuth, McIlroy (1986) Programming pearls: a literate program Communications of the ACM, 29(6), pp471-483, June 1986 [doi:10.1145/5948.315654](http://dx.doi.org/10.1145/5948.315654)
* Dr. Drang (2011) [More shell, less egg](http://www.leancrew.com/all-this/2011/12/more-shell-less-egg/), 4 December 2011. 
