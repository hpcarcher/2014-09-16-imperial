Version Control and Git
=======================

Introduction
============

Questions:

* Someone asks you, "can I have the code you used to create the data that you graphed in your conference paper?" How easy would it be for you to get this code for them?
* A reviewer asks you "I need this figure redone more clearly?" Could you recreate the figure using exactly the same code, configuration and data as the original? Or would you have to say "er, well, this is not quite the same figure..."?
* Someone tells you, "your laptop's just been stolen!" How much work have you lost?
* You're working with a colleague on a journal paper who storms into your office and shouts, "you've just deleted my analysis section". Would you have to apologise and ask them to write it again?
* You're working with colleagues on a code and you find that a function you wrote has been rewritten and you want to know why, how easy would it be to find this out? Now suppose the person who rewrote it left 3 months ago, how easy would it be then?

Cartoons:

* ["FINAL".doc](http://www.phdcomics.com/comics/archive.php?comicid=1531)
* [A story told in file names](http://www.phdcomics.com/comics/archive.php?comicid=1323)
* You can back-up a file only to realise later you have backed up a corrupted version.

Version control:

* AKA revision control.
* Preserves history of changes made to files or directories.
* Track Changes in Microsoft Word or file versions in DropBox or Google Drive
* [Wikipedia](http://en.wikipedia.org/wiki/Main_Page)
* ....but more powerful...changes to sets of files.

Uses:

* Source code.
* Scripts.
* Makefiles.
* Configuration files.
* Parameter sets.
* Data files.
* User documentation, manuals, conference papers, journal papers, book chapters, whether they be plain-text, LaTeX, XML, or whatever.

[Git](http://git-scm.com/):

* Git is not GitHub.
* Git is a version control tool.
* GitHub is a project hosting infrastructure, that uses Git.

Tracking our changes with a local repository
============================================

Create a local repository
-------------------------

Make sure you are not in any existing repository or any clone of the boot camp materials!

    git
    git --version
    git help
    git help --all
    git help checkout
    mkdir cookerybook
    cd cookerybook
    git init

Current directory is called the 'working directory'.

`.git` directory contains Git's configuration files.

    ls .git

Configure Git locally
---------------------

Set identity:

    git config --global user.name "Your Name"
    git config --global user.email "yourname@yourplace.org"

`--global` updates global configuration applied to all Git repository in your file system.

    cat ~/.gitconfig

Set default editor:

    git config --global core.editor nano
    git config --global core.editor vi
    git config --global core.editor xemacs
    git config -l                          # Alternative to cat ~/.config

Do not mess around with these files as you could lose work!

Add files and directories and recording changes
-----------------------------------------------

Write a recipe:

* Create a file `recipe.md`.
* [Markdown](http://daringfireball.net/projects/markdown/syntax) syntax.
* Add Ingredients and Cooking Instructions.

Current status of files in the repository:

    git status recipe.md

`Untracked` - files in working directory that Git is not managing.

Add file to 'staging area' (AKA 'index' or 'cache'):

    git add recipe.md
    git status recipe.md

`Changes to be committed` - content in the staging area, ready for Git to manage. Think of it as a loading dock.

    git commit

Git can deduce what was changed and when, and, using the configuration, who by, but not why. 

'Commit messages' provide the why:

* Messages like "made a change" or "commit 5" are redundant.
* A good commit message usually contains a one-line description followed by a longer explanation if necessary.

Git shows number of files changed and the number of lines inserted or deleted across all those files.

    git status recipe.md

`nothing to commit` - Git is managing everything in the working directory.

    git log

'Commit identifier' AKA 'revision number' uniquely identifies changes made in this commit, author, date, and commit message.

    git log --relative-date

Make updates to `recipe.md`.

    git status recipe.md

`Changes not staged for commit` section and `modified` marker - file managed by Git has been modified and changes are not yet commited.

    git add recipe.md
    git commit

Good commits are atomic - they consist of the smallest change that remains meaningful. 

For code, it's useful to commit changes that can be reviewed by someone else in under an hour.:

* Fagan (1976) discovered that a rigorous inspection can remove 60-90% of errors before the first test is run. M.E., Fagan (1976). [Design and Code inspections to reduce errors in program development](http://www.mfagan.com/pdfs/ibmfagan.pdf). IBM Systems Journal 15 (3): pp. 182-211.
* Cohen (2006) discovered that all the value of a code review comes within the first hour, after which reviewers can become exhausted and the issues they find become ever more trivial. J. Cohen (2006). [Best Kept Secrets of Peer Code Review](http://smartbear.com/SmartBear/media/pdfs/best-kept-secrets-of-peer-code-review.pdf). SmartBear, 2006. ISBN-10: 1599160676. ISBN-13: 978-1599160672.

Create a directory:

    mkdir images
    cd images

Download image of recipe from the web and put into `images` or use `wget`:

    wget http://www.cookuk.co.uk/images/slow-cooker-winter-vegetable-soup/smooth-soup.jpg

Add directory to repository:

    git add images
    git commit -m "Added images directory and soup image" images

Add link to recipe:

    [Soup](images/smooth-soup.jpg "My soup")

Commit:

    git commit -m "Added link to image of my soup." recipe.md

What (not) to store in the repository:

* Store anything that is created manually e.g. source code, scripts, Makefiles, plain-text documents, notes, LaTeX documents, configuration files, input files.
* This can include Word or Excel.
* Do not store anything created automatically by a compiler or tool e.g. object files, binaries, libraries, PDFs etc.
* These can be recreated from sources.
* Remove risk of auto-generated files becoming out of synch with manual ones.

Discard changes
---------------

Make and commit changes to `recipe.md`.

    git diff recipe.md

* `-` - a line was deleted. 
* `+` - a line was added. 
* A line that has been edited is shown as a removal of the old line and an addition of the updated line.

Throw away local changes and 'revert':

    git checkout -- recipe.md
    git status data-report.md

Look at history
---------------

Make and commit changes to `recipe.md`.

    git log
    git log recipe.md
    git diff COMMITID
    git diff OLDER_COMMITID NEWER_COMMITID

Rollback working directory to state of repository at first commit:

    git log
    git checkout COMMITID
    ls
    cat recipe.md

Return to current state:

    git checkout master
    ls

'undo' and 'redo' for directories and files.

More frequent commits increase the granularity of 'undo':

* Commit early, commit often.
* If you make a mistake (e.g. check in buggy code) then the old version is still there.

DropBox and GoogleDrive:

* Also preserve every version, but they delete old versions after 30 days, or, for GoogleDrive, 100 revisions. 
* DropBox allows old versions to be stored for longer but you have to pay. 
* Version control is only bounded by space available.

Use tags as nicknames for commit identifiers
--------------------------------------------

Commit identifiers are cryptic and meaningless to humans.

'Tags' allow commit identifiers to be named.

    git tag BOOT_CAMP
    git tag

Make and commit changes to `recipe.md`.

    git checkout BOOT_CAMP
    git checkout master

When to tag and why:

* When software is released e.g. `VERSION.1.2` so can retrieve this version to fix bugs.
* When configuration files and scripts are used to generate data for a paper e.g. `JPHYSCOMP.03.14` so can redo an analysis if paper comes back from reviewers with questions.

Tag naming, like variable, function, class and script names, should be:

* Easy-to-remember.
* Human-readable.
* Meaningful.
* Self-documenting.

Branches
--------

    git status recipe.md

What is `master`?

A branch - a set of related commits made to files the repository, each of which can be used and edited and updated concurrently. 

`master` is Git's default branch.

Create a new branch from any commit at any time.

When complete, merge the branch into another branch, or into `master`.

Question: why might this be useful?

* Suppose we develop some software and release this to our users. We then rewrite some functionality and add some more. Suppose a user finds a bug and we want to fix it and get this user a bug-fixed version. Our rewrites may not be complete and we have the choice of either releasing a half-complete, rewrites-in-progress version, or telling our user to wait until the next release comes out - which may cause them to lose interest if the bug is preventing them from using our software.
* Experiment with developing a new feature, or a refactoring, we're not sure we'll keep.
* Simultaneously prepare a paper for submission and add a new section for a future submission.

    git branch

`*` - current branch.

Create a new branch:

    git branch new_recipe_format
    git branch

Switch to new branch:

    git checkout new_recipe_format
    git branch

Two concurrent branches now exist and we can work on either.

Make and commit changes to `recipe.md`.

Switch between branches:

    git checkout master
    cat recipe.md
    git checkout new_recipe_format
    cat recipe.md
    git checkout master
    cat recipe.md

Once happy with our new format, we can merge changes from `new_recipe_format` into `master`:

    git merge new_recipe_format

Merging is done on a file-by-file basis, merging files line by line.

Make and commit changes to `recipe.md`.

    git checkout new_recipe_format

Make, commit and push changes to `recipe.md`, taking care to edit the same lines as above, but making different changes.

`CONFLICT` - changes can't be seamlessly merged because changes have been made to the same set of lines in the same files.

    git status

`Unmerged` - files which have conflicts.

    cat recipe.md

Conflict markup:

* `<<<<<<< HEAD` - conflicting lines local commit.
* `=======` - divider between conflicting regions.
* `>>>>>>> 71d34decd32124ea809e50cfbb7da8e3e354ac26` - conflicting lines from remote commit.

Conflict resolution - edit file and do one of:

* Keep the local version, which, here, is the one marked-up by `HEAD`.
* Keep the other version, which, here, is the one marked-up by the commit identifier.
* Or keep a combination of the two.

Remove all the markup.

    git add recipe.md
    git commit -m "Resolved confict in recipe.md by ..."
    git log

`Merge branch` entry.

Pretty-print history:

    git log --oneline --graph --decorate --all

Delete branch, once merged changes in:

    git branch -D new_recipe_format

DropBox and GoogleDrive don't support this ability.

No work is ever lost.

Branch images
-------------

We create our branch for the new feature.

    -c1---c2---c3                               master
                \
                 c4                             feature1

We can then continue developing our software in our default, or master, branch,

    -c1---c2---c3---c5---c6---c7                   master
                \
                 c4                                feature1

And, we can work on the new feature in the feature1 branch

    -c1---c2---c3---c5---c6---c7                   master
                \
                 c4---c8---c9                      feature1

We can then merge the feature1 branch adding new feature to our master branch (main program):

     -c1---c2---c3---c5---c6---c7--c10              master
                \                   /
                 c4---c8---c9------                 feature1

When we merge our feature1 branch with master git creates a new commit which contains merged files from master and feature1. After the merge we can continue developing. The merged branch is not deleted. We can continue developing (and making commits) in feature1 as well.

    -c1---c2---c3---c5---c6---c7--c10---c11--c12     master
                \                /
                 c4---c8---c9-------c13              feature1

One popular model is to have,

* A release branch, representing a released version of the code.
* A master branch, representing the most up-to-date stable version of the code.
* Various feature and/or developer-specific branches representing work-in-progress, new features etc.

For example,

               0.1      0.2        0.3
              c6---------c9------c17------            release
             /          /       /
     c1---c2---c3--c7--c8---c16--c18---c20---c21--    master
     |                      /
     c4---c10---c13------c15                          fred
     |                   /
     c5---c11---c12---c14---c19                       kate

There are different possible workflows when using Git for code development.

One of the examples may be when the master branch holds stable and tested code. If a bug is found by a user, a bug fix can be applied to the release branch, and then merged with the master branch. When a feature or developer-specific branch, is stable and has been reviewed and tested it can be merged with the master branch. When the master branch has been reviewed and tested and is ready for release, a new release branch can be created from it.

Summary
-------

* Keep track of changes like a lab notebook for code and documents.
* Roll back changes to any point in the history of changes - 'undo' and 'redo'.
* Use branches to work on concurrent versions of the repository.

Question: what problems or challenges do we still face in managing our files?

Answer:

* If we delete our repository not only have we lost our files we've lost all our changes!
* Suppose we're away from our usual computer, for example we've taken our laptop to a conference and are far from our workstation, how do we get access to our repository then?

Work from multiple locations with a remote repository
=====================================================

Repository hosting:

* Site or institution-specific repositories.
* [GitHub](http://github.com) - pricing plans to host private repositories.
* [Bitbucket](https://bitbucket.org) - free, private repositories to researchers. 
* [Launchpad](https://launchpad.net) 
* [GoogleCode](http://code.google.com)
* [SourceForge](http://sourceforge.net)

Version control plus integrated tools:

* Network graphs and time histories changes to repositories.
* Commit-triggered e-mails.
* Browsing code from within a web browser, with syntax highlighting.
* Software release management.
* Issue (ticket) and bug tracking.
* Wikis.
* Download.
* Varying permissions for various groups of users.
* Other service hooks e.g. to Twitter.

Get an account
--------------

* [Sign-up for free GitHub account](https://github.com/signup/free)
* [Sign-up for free BitBucket account](https://bitbucket.org/account/signup/)

Create new repository
---------------------

GitHub:

* Log in to [GitHub](https://github.com)
* Click on the Create a new repo icon on the top right, next to your user name
* Enter Repository name: `cookbook`
* Make sure the Public option is selected
* Make sure the Initialize this repository with a README is unselected
* Click Create Repository

BitBucket:

* Log in to [Bitbucket](https://bitbucket.com).
* Click on Create icon on top left, next to Bitbucket logo.
* Enter Repository name: `cookbook`.
* Check private repository option is ticked.
* Check repository type is `Git`.
* Check Initialize this repository with a README is unselected.
* Click Create Repository.

Question: is publicly visible code on BitBucket or GitHub open source?
Answer: yes, but only if it has an open source licence. Otherwise, by default, it is "all rights reserved".

'Push' `master` branch to GitHub:

    git remote add origin https://github.com/USERNAME/cookbook.git
    git push -u origin master

'Push' `master` branch to BitBucket:

    git remote add origin https://USERNAME@bitbucket.org/USERNAME/cookbook.git
    git push -u origin master

`origin` is an alias for repository URL.

`-u` sets local repository to track remote repository.

`master` branch content is copied across the remote repository, named via the alias `origin`, and a new `master` branch is created the remote repository.

Check master branch is now on GitHub:

* Click Code tab.
* Click Commits tab.
* Click Network tab.

Check master branch is now on BitBucket:

* Click Source tab.
* Click Commits tab.

Clone remote repository
-----------------------

Suppose something dramatic and dire happens:

    rm -rf cookerybook

Copy, or 'clone', the remote repository:

    $ git clone https://github.com/USERNAME/cookbook.git

    $ git clone https://USERNAME@bitbucket.org/USERNAME/cookbook.git

    $ cd cookbook
    $ git log
    $ ls -A

Question: where is the `cookerybook` directory?

Answer: `cookerybook` was the directory that held our local repository but was not a part of it.

Push changes to remote repository
---------------------------------

Make and commit changes to `recipe.md`.

    git push

Refresh web pages and check that changes are now in the remote repository.

Collaboration
=============

Form into pairs and swap GitHub / BitBucket user names.

One of you share your repository with your partner - we'll call you the Owner:

* GitHub - Owner click on the Settings tab, click on Collaborators, and add partner's GitHub name.
* BitBucket - Owner click on the Share link, and add partner's BitBucket name.

Both Owner and partner clone the Owner's repository e.g.

    git clone https://github.com/OWNERUSERNAME/cookbook.git
  
    git clone https://USERNAME@bitbucket.org/USERNAME/cookbook.git 

Owner make, commit and push changes to `recipe.md`.

Pull changes from a remote repository
-------------------------------------

Partner 'fetch' changes from remote repository:

    git fetch
    git diff origin/master

`diff` compares current, `master` branch, with `origin/master` branch which is the name of the `master` branch in `origin` which is the alias for the cloned repository, the one in our remote repository.

'Merge' changes into current repository, which merges the branches together:

    git merge origin/master
    cat recipe.md

Merging is done on a file-by-file basis, merging files line by line just as when we used branches locally.

Partner make, commit and push changes to `recipe.md`.

Owner 'fetch' changes from remote repository.

Partner 'pull' changes from remote repository:

    git pull
    cat recipe.md
    git log

`pull` does a `fetch` and a `merge` in one go. 

Exercise - collaborate
======================

Owner and partner alternatively pull, change, commit, push.

Owner and partner together:

* Both edit different lines of the same file, add it and commit it.
* Both push.
* Slower one pull then push.
* Both edit same lines of the same file, add it and commit it.
* Both push.
* Slower one pull, resolve conflicts, push.
* Repeat! 
* Try editing and adding new files and directories too.

Summary
-------

* Host repository remotely.
* Copy, or clone, remote repository onto local machine
* Make changes to local repository and push these to remote repository
* Fetch and merge, or pull, changes from remote repository into local repository
* Identify and resolve conflicts when the same file is edited within two repositories

Exercise - Copy the SWC material
================================

* Create a `bootcamp` repository on GitHub/BitBucket.
* Change into the directory you cloned at the start of the boot camp.
* Push this repository to your remote `bootcamp` repository.
* Keep using this repository throughout the rest of the boot camp!

Conclusion
==========

* Keep track of changes like a lab notebook for code and documents - ideas explored, fixes made, refactorings done, false paths explored - what was changed, who by, when and why.
* Roll back changes to any point in the history of changes to files and directories - "undo" and "redo" for files.
* Confidence to play around, experiment without deriving baroque naming schemes or risking losing your work.
* Back up entire history of changes in various locations.
* Work on files from multiple locations.
* Identify and resolve conflicts when the same file is edited within two repositories without losing any work.
* Collaboratively work on code or documents or any other files without any loss of work and with full provenance and accountability.
* Version control is just as useful, relevant, powerful, helpful, necessary for a solo researcher as for a team of researchers.

Now, consider again the initial questions:

* Someone asks you, "can I have the code you used to create the data that you graphed in your conference paper?" How easy would it be for you to get this code for them?
* A reviewer asks you "I need this figure redone more clearly?" Could you recreate the figure using exactly the same code, configuration and data as the original? Or would you have to say "er, well, this is not quite the same figure..."?
 * You can use your version control logs and tags to immediately retrieve the exact version of the code that you used.
* Someone tells you, "your laptop's just been stolen!" How much work have you lost?
 * Ideally you'll have lost no work as you push it regularly to a remote repository.
* You're working with a colleague on a journal paper who storms into your office and shouts, "you've just deleted my analysis section". Would you have to apologise and ask them to write it again?
 * Ask them to retrieve the previous version from the repository.
* You're working with colleagues on a code and you find that a function you wrote has been rewritten and you want to know why, how easy would it be to find this out? Now suppose the person who rewrote it left 3 months ago, how easy would it be then?
 * Look through the logs at the commit messages and, hopefully, these'll explain why it was changed.

"If you are not using version control then, whatever else you may be doing with a computer, you are not doing science" -- Greg Wilson
