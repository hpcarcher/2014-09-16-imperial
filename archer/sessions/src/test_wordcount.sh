#!/bin/sh

# File to check existence for.
test_file_exists() {
  if [ -f "$1" ]
  then
    echo "OK: $1 exists"
  else
    echo "FAILURE: $1 not found"
  fi
}

# File to check non-existence for.
test_file_not_exists() {
  if [ -f "$1" ]
  then
    echo "FAILURE: $1 exists"
  else
    echo "OK: $1 not found"
  fi
}

# $1 file to compare
# $2 file to compare
test_files_equal() {
  compare=`diff -rq $1 $2`
  if [ -z "$compare" ]; then
    echo "OK: $1 equals $2"
  else
    echo "FAILURE: $1 does not equal $2"
  fi
}

rm -f *.dat

echo "Test - no such file"
python wordcount.py no_such_file.txt none.dat
test_file_not_exists none.dat

for file in $(ls books/*.txt); do
  name=`basename $file .txt`
  output_file=$name.dat
  echo "Test - $file"
  python wordcount.py $file $output_file
  test_file_exists $output_file
  test_files_equal $output_file expected/$output_file
done
