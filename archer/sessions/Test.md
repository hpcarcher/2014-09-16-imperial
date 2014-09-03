Testing
=======

Question: how do you test?

* Compile the code ("if it builds, ship it")?
* Run on correct inputs, check results by visual inspection?
* Run on correct inputs, check results automatically?
* Run tests after every change or bug fix?
* Use continuous integration?
* Document manual testing?
* Run on inputs known to cause failures?
* Run to ensure fixed bugs don't reoccur?

Question: if you don't test, why not?

* "I don't write buggy code". Almost all code has bugs.
* "It's too hard". This can be a sign that the code is not well designed.
* "My code can't be tested". Why not?
* "It's not interesting"
* "It takes too much time and I've research to do"

Avoid embarrassment:

* [Geoffrey Chang](http://en.wikipedia.org/wiki/Geoffrey_Chang) had to retract 3 papers from [Science](http://www.sciencemag.org) due to a flipped sign bit.
* McKitrick and Michaels published an [erratum](http://www.int-res.com/articles/cr2004/27/c027p265.pdf) to a [Climate Research 26(2) 2004](http://www.int-res.com/abstracts/cr/v26/n2/p159-173/) paper due to a [problem](http://crookedtimber.org/2004/08/25/mckitrick-mucks-it-up/) caused by degrees and radians.
* Ariane 5 used Ariane 4 software. Ariane 5's new engines caused the code to produce a buffer overflow. Ariane 5 blew up!

Save money:

* Find a bug on your laptop for free before you submit a job to a charged-for HPC resource.
 
Save time:

* Spot bugs before you analyse data produced by your scripts.
* 1-10-100 rule.

Safety net:

* Fix bugs, optimise and parallelise without introducing (new) bugs.
* EPCC and Colon Cancer Genetics Group (CCGG) of MRC Human Genetics Unit at Western General Hospital Edinburgh optimised and parallelised FORTRAN genetics code.

Documentation:

* How to use, or not to use, scripts and code and what they do.

Verification - "Have we built it correctly?" Is it bug free, precise, accurate, and repeatable?

Validation - "Have we built the right thing?" Is it designed in such a way as to produce the answers we are interested in, data we want, etc?

Finding bugs before testing
---------------------------

Question: what is single most effective way of finding bugs?

Answer: 

* Fagan (1976) Rigorous inspection can remove 60-90% of errors before the first test is run. M.E., Fagan (1976). [Design and Code inspections to reduce errors in program development](http://www.mfagan.com/pdfs/ibmfagan.pdf). IBM Systems Journal 15 (3): pp. 182-211.
* Cohen (2006) The value of a code review comes within the first hour, after which reviewers can become exhausted and the issues they find become ever more trivial. J. Cohen (2006). [Best Kept Secrets of Peer Code Review](http://smartbear.com/SmartBear/media/pdfs/best-kept-secrets-of-peer-code-review.pdf). SmartBear, 2006. ISBN-10: 1599160676. ISBN-13: 978-1599160672.

Introducing tests from the outside-in
-------------------------------------

* Unit tests test small individual functions.
* Researchers inherit large codes, which may not have any tests. 
* Where to start with a unit test?
* Evolve tests from the outside-in.
* [Software Sustainability Institute](http://www.software.ac.uk) initial test framework for [FABBER](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FABBER) C++ image analysis software.

End-to-end tests and Python
----------------------------

Determine a flow pattern:

    python cfd.py configs/32x32.cfg 32x32.dat
    head 32x32.dat
    python cfd.py configs/32x32.cfg 32x32.dat quiet
    head 32x32.dat

Question: what is possibly the simplest test we could do? 

Answer: check there is an output file produced for a valid input file

Use Python to write end-to-end tests. Program being tested does not have to be Python. Can use the same approach using shell scripts.

Create `test_cfd_end_to_end.py`:

    import os
    import os.path

    def file_exists(filename):
      if (os.path.isfile(filename)):
        print "OK ", filename, "exists"
      else:
        print "FAIL ", filename, "does not exist"

    for f in os.listdir("."):
      if f.endswith(".dat"):
        os.remove(f)

    print "Test 32x32"
    os.system("python cfd.py configs/32x32.cfg 32x32.dat quiet")
    file_exists("32x32.dat")

Use functions as these commands will be called more than once - anticipate reuse.

    python test_cfd_end_to_end.py

<p/>

    print "Test 64x64"
    os.system("python cfd.py configs/64x64.cfg 64x64.dat quiet")
    file_exists("64x64.dat")

Tests that code, or functions, work with valid inputs are complemented by tests that they fail with invalid inputs.

Exercise 1 - write a test for no file
-------------------------------------

See [exercises](TestExercises.md).

Solution:

    def file_not_exists(filename):
      if (os.path.isfile(filename)):
        print "FAIL ", filename, "exists"
      else:
        print "OK ", filename, "does not exist"

    print "Test none"
    os.system("python cfd.py configs/none.cfg none.dat")
    file_not_exists("none.dat")

Check actual outputs against expected outputs
---------------------------------------------

Create expected outputs:

    make 32x32.dat
    make 64x64.dat
    make 64x64x20.dat
    make 96x96.dat

Or:

    python cfd.py configs/32x32.cfg 32x32.dat quiet
    python cfd.py configs/64x64.cfg 64x64.dat quiet
    python cfd.py configs/64x64x20.cfg 64x64x20.dat quiet
    python cfd.py configs/96x96.cfg 96x96.dat quiet

<p/>

    mkdir expected/
    mv *.dat expected/

    python cfd.py configs/64x64.cfg 64x64.dat
    diff 64x64.dat expected/64x64.dat              # Compare files for equality
    diff -q 64x64.dat expected/64x64.dat           # Return result only
    diff -q 64x64.dat expected/32x32.dat           # Return result only

`os.system` returns the command's exit code, `0` for OK, and non-zero for errors.

    def files_equal(file1,file2):
      cmd = "diff -q " + file1 + " " + file2
      result = os.system(cmd)
      if (result == 0):
        print "OK ", file1, "equals", file2
      else:
        print "FAIL ", file1, "does not equal", file2

    files_equal("32x32.dat", "expected/32x32.dat")
    files_equal("64x64.dat", "expected/64x64.dat")

Reduce duplicated code and loop over files:

    for f in os.listdir("configs"):
      if f.endswith(".cfg"):
        name = os.path.splitext(f)[0]
        cfg = os.path.join("configs",f)
        dat = name + ".dat"
        cmd = "python cfd.py " + cfg + " " + dat + " quiet"
        os.system(cmd)
        file_exists(dat)
        expected = os.path.join("expected", dat)
        files_equal(dat,expected)

Exercise 2 - recode `cfd.py`
----------------------------

See [exercises](TestExercises.md).

Solution:

    for iteration in range(num_iterations):
      update_stream_function(width, height, psi)

    for i in range(1, height+1):
      for j in range(1, width+1):
        tmp[i][j] = 0.25 * (psi[i+1][j]+psi[i-1][j]+psi[i][j+1]+psi[i][j-1])

    for i in range(1, height+1):
      for j in range(1, width+1):
        psi[i][j] = tmp[i][j]

Test structure
--------------

Test structure:

* Set-up expected outputs given known inputs e.g. `expected/.dat` files or `0` for the return code.
* Run component on known inputs.
* Check if actual outputs match expected outputs, or, for invalid inputs, that behaviour is as expected.

Regardless of whether it is a test of a:

* 10 line function.
* Component or library.
* Serial application running on a single processor.
* Parallel application running on a multiple processors.
* Automated or manual.

Data file meta-data
-------------------

Add meta-data to record the provenance of the output data file. 

    import datetime

    write_data(width, height, psi, config, dat_file)

    def write_data(width, height, psi, config, dat_file):

    out.write("# Created by: {0}\n".format(__file__))
    out.write("# Date: {0}\n".format(datetime.datetime.today()))
    out.write("# Grid size: {0} x {1}\n".format(width, height))
    out.write("# Inlet X: {0}\n".format(config['inlet_x']))
    out.write("# Inlet width: {0}\n".format(config['inlet_width']))
    out.write("# Outlet Y: {0}\n".format(config['outlet_y']))
    out.write("# Outlet width: {0}\n".format(config['outlet_height']))

    python cfd.py configs/64x64.cfg 64x64.dat
    head 64x64.dat
    python test_cfd_end_to_end.py

Question: what is the problem?

Answer: the meta-data. `diff` is too simplistic now. 

* Want finer-grained tests of equality between data files. 
* Use information about the file content and structure.
* Difference between syntactic and semantic content.

When 0.1 + 0.2 == 3.00000000004
-------------------------------

Question: what other problems might `diff` experience with data files?

Answer: floating point values.

    python
    a = 0.1
    b = 0.2
    print a + b
    print a + b == 0.3
    a + b

Simple tests for the equality of two floating point values are problematic due to imprecision in values.

Compare for equality within a given threshold, or delta e.g. *expected* and *actual* to be equal if *expected - actual < 0.000000000001*.

Python [nose](https://pypi.python.org/pypi/nose/) library includes functions for floating point equality.

    python
    from nose.tools import assert_almost_equal
    expected = 2
    expected = 2.000001
    actual = 2.0000000001
    assert_almost_equal(expected, actual, 0)
    assert_almost_equal(expected, actual, 1)
    assert_almost_equal(expected, actual, 2)
    assert_almost_equal(expected, actual, 3)
    assert_almost_equal(expected, actual, 4)
    assert_almost_equal(expected, actual, 5)
    assert_almost_equal(expected, actual, 6)

`nose.testing` uses absolute tolerance: abs(x, y) <= delta.

[Numpy](http://www.numpy.org/) `numpy.testing` uses relative tolerance: abs(x, y) <= delta * (max(abs(x), abs(y)). 

`data/` has files produced by the same software, with the same inputs, under the same configuration. 

Question: why might they differ?

Answer: they were run on different numbers on processors e.g. 2x1, 4x2 etc.

    diff -q data/2x1.dat data/data4x2.dat

<p/>

    import numpy as np

    def test_data_files_equal():
      file21 = np.loadtxt("data/data2x1.dat")
      file42 = np.loadtxt("data/data4x2.dat")
      np.testing.assert_equal(file21, file42)

    test_data_files_equal()

<p/>

    python test_cfd_end_to_end.py

<p/>

    assert_equal(expected, psi)
    np.testing.assert_allclose(file21, file42, rtol=0, atol=1e-7)

What is a suitable threshold for equality? That is application-specific - for some domains round to the nearest whole number, for others be far, far more accurate.

Testing at finer-granularities - towards unit tests
---------------------------------------------------

* Tests at varying levels of granularity should be written.
* Changes to a specific component can be tested before the component is integrated.
* Quicker to discover a problem when testing a 10 line function in isolation then testing it as part of an end-to-end application which may take 1 hour to run and may not even, depending upon the inputs, invoke that function. 
* Finest level of granularity is a unit test where a unit is the smallest testable part of an application e.g. function or module, method or class.

Exercise 3 - propose unit tests for `cfd.py` and `jacobi.py`
------------------------------------------------------------

See [exercises](TestExercises.md).

Some example solutions (there are many more)

`set_inlet_boundary`

* The values of the stream function on the top edge are initialised as
  expected. 
* An inlet that goes out of the bounds of the box results in an
  error.  

`set_outlet_boundary`

* The values of the stream function on the right edge are initialised
  as expected. 
* An outlet that goes out of the bounds of the box results in an
  error.  

`update_stream_function`

* Given a stream function with known values then, after the function
  call, the stream function contains new values equal to those we
  have pre-computed.
* Given a stream function that is all zeros then, after the function
  call, it is still all zeroes.
* Given a stream function that is all ones then, after the function
  call, it is still all zeroes.
* Given a stream function that is an empty list, then, after the
  call, it is still an empty list.

`write_data`

* Given zero width and height and an empty stream function then a
  file is saved which has width and height zero.
* Given a 1x1 grid and a 3x3 stream function that is all zeros then a
  file is saved that has contents:

<p/>

    1 1
    0  0  0.00000 0.00000 0.00000

* Given a 2x2 grid and a4x4 stream function that is all ones then a
  file is saved that has contents:

<p/>

    2 2
    0     0    0.00000    0.00000    0.00000
    0     1    0.00000    0.00000    0.00000
    1     0    0.00000    0.00000    0.00000
    1     1    0.00000    0.00000    0.00000

A unit test for `update_stream_function`
----------------------------------------

Create `test_cfd.py`:

    from jacobi import update_stream_function

Python [nose](https://pypi.python.org/pypi/nose/) library includes tests for equality, inequality, boolean values, thrown exceptions etc.

    from nose.tools import assert_equal

We'll be using floating points too:

    import numpy as np

    def test_empty_psi():
      psi=[]
      update_stream_function(0, 0,psi)
      assert_equal([], psi)

    def test_zero_psi():
      psi = [[0 for col in range(3)] for row in range(3)]
      expected = [[0 for col in range(3)] for row in range(3)]
      update_stream_function(1, 1, psi)
      np.testing.assert_allclose(expected, psi, rtol=0, atol=0.01)

    test_empty_psi()
    test_zero_psi()

<p/>

    python test_cfd.py

`nosetests` automatically finds, runs and reports on tests.

    nosetests test_cfd.py

`.` denotes successful test function calls.

Uses 'reflection' to find out the test functions - `test_` function, module and file prefixes.

Remove `test_` calls.

<p/>

    python test_cfd.py

<p/>

Looking at how the the stream function is updated, for each value not
on the boundaries of the box, it sums the four neighbours then
multiplies by 0.25. If the stream function is all ones then the centre
values will have value (1+1+1+1)*0.25 = 1. So given a stream function
of all ones we expect `update_stream_function` to be the identity
function:

    def test_ones_psi():
      psi = [[1 for col in range(5)] for row in range(5)]
      expected = [[1 for col in range(5)] for row in range(5)]
      update_stream_function(3, 3,psi)
      np.testing.assert_allclose(expected, psi, rtol=0, atol=0.01)

Another example with pre-calculated values. Given:

    # 0 0 1 0 0
    # 0 0 1 0 0
    # 0 0 1 1 1
    # 0 0 0 0 0
    # 0 0 0 0 0

Then we expect:

    # 0 0    1    0    0
    # 0 0.25 0.5  0.5  0
    # 0 0.25 0.5  0.5  1
    # 0 0    0.25 0.25 0
    # 0 0    0    0    0

Test function:

    def test_psi():
      psi = [[0 for col in range(5)] for row in range(5)]
      expected = [[0 for col in range(5)] for row in range(5)]
      psi[0][2] = 1
      psi[1][2] = 1
      psi[2][2] = 1
      psi[2][3] = 1
      psi[2][4] = 1
      expected[0][2] = 1
      expected[1][1] = 0.25
      expected[1][2] = 0.5
      expected[1][3] = 0.5
      expected[2][1] = 0.25
      expected[2][2] = 0.5
      expected[2][3] = 0.5
      expected[2][4] = 1
      expected[3][2] = 0.25
      expected[3][3] = 0.25
      update_stream_function(3, 3, psi)
      np.testing.assert_allclose(expected, psi, rtol=0, atol=0.01)

`nose` is an [xUnit test framework](http://en.wikipedia.org/wiki/XUnit). Others are JUnit, CUnit, google-test, FRUIT, pFUnit etc.

xUnit test report, standard format, convert to HTML, present online.

    nosetests --with-xunit test_cfd.py
    cat nosetests.xml

Defensive programming
---------------------

Suppose an invalid set of values are passed to `cfd` e.g. the inlet
width extends beyond the end of the box:

    python
    from cfd import cfd
    cfd(1000, 32, 20, 20, 5, 15, "out.dat")  
    CTRL-D

Assume that mistakes will happen and guard against them. Defensive programming.

    def cfd(iterations, edge, inlet_x, inlet_width, outlet_y, outlet_height, dat_file, quiet=True):

      assert inlet_x <= width
      assert inlet_x + inlet_width < width
      assert outlet_y <= height
      assert outlet_y + outlet_height < height

Try again:

    python
    from cfd import cfd
    cfd(1000, 32, 20, 20, 5, 15, "out.dat")  
    CTRL-D

Programs like Firefox  are full of assertions: 10-20% of their code is to check that the other 80-90% is working correctly.

For HPC resources, want to exit a.s.a.p rather than continue executing and burn up resource allocations.

Types:

* Precondition - must be true at the start of a function in order for it to work correctly.
* Postcondition - guaranteed to be true when a function finishes.
* Invariant - always true at a particular point inside a piece of code.

Help other developers understand program and whether their understanding matches the code. Users should never see these sorts of failure!

    from nose.tools import assert_raises
    from cfd import cfd

    def test_cfd_invalid_inlet_width():
      assert_raises(AssertionError, cfd, 1000, 32, 20, 20, 5, 15, "out.dat")  

Exercise 4 - implement unit tests for `cfd.py` and `jacobi.py`
--------------------------------------------------------------

See [exercises](TestExercises.md).

Allow 15-30 minutes.

Automated testing jobs
----------------------

Automated tests can be run:

* Manually.
* At regular intervals.
* Every time code is commited to revision control.

A simple automatic triggering of automated tests is via a Unix `cron` job.

A more advanced approach is via a continuous integration server. These trigger automated test runs and publish the results.

* [Muon Ion Cooling Experiment](http://www.mice.iit.edu/) (MICE) have a large number of tests written in Python. They use [Jenkins](http://jenkins-ci.org/) to build their code and trigger the running of the tests which are then [published online](http://test.mice.rl.ac.uk/).
* [Apache Hadoop Common Jenkins dashboard](https://builds.apache.org/job/Hadoop-Common-trunk/)

Tests are code
--------------

Review tests and avoid tests that:

* Pass when they should fail, false positives.
* Fail when they should pass, false negatives.
* Don't test anything. 

<p/>

    def test_critical_correctness():
      # TODO - will complete this tomorrow!
      pass

Conclusion
----------

Tests:

* Set-up expected outputs given known inputs.
* Run component on known inputs.
* Check if actual outputs match expected outputs.

When to test:

* Always!
* Early. Don't wait till after the code's been used to generate data for an important paper, or been given to someone else.
* Often. So any bugs can be identified a.s.a.p. Bugs are easier to fix if they're identified at the time the relevant code is being actively developed.
* Turn bugs into assertions or tests. Check that bugs do not reappear.

When to finish:

* "It is nearly impossible to test software at the level of 100 percent of its logic paths", fact 32 in R. L. Glass (2002) [Facts and Fallacies of Software Engineering](http://www.amazon.com/Facts-Fallacies-Software-Engineering-Robert/dp/0321117425) ([PDF](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.94.2037&rep=rep1&type=pdf)).
* But, no excuse for not testing anything.
* When do you finish proof reading a paper? Learn from experience. 

"If it's not tested, it's broken" - Bruce Eckel, in [Thinking in Java, 3rd Edition](http://www.mindview.net/Books/TIJ/).

["Testing is science"](http://maori.geek.nz/post/testing_your_code_is_doing_science) - Graham Jenson.
