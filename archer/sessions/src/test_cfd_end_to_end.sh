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
  compare=$(diff -rq $1 $2)
  if [ -z "$compare" ]; then
    echo "OK: $1 equals $2"
  else
    echo "FAILURE: $1 does not equal $2"
  fi
}

rm -f *.dat

for file in $(ls configs/*.cfg); do
  name=$(basename $file .cfg)
  output_file=$name.dat
  echo "Test - $file"
  python cfd.py $file $output_file quiet
  test_file_exists $output_file
  test_files_equal $output_file expected/$output_file
done

echo "Test - no such file"
python cfd.py no_such_file.cfg none.dat quiet
test_file_not_exists none.dat
