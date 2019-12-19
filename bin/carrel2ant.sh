#!/usr/bin/env bash

# carrel2ant.sh - given a directory with specifically shaped file names, output an AntConc preference file

# Eric Lease Morgan <eric_morgan@infomotions.com>
# December 14, 2019 - first cut


# configure
TEMPLATE='./etc/template.ant'

# sanity check
if [[ -z $1 ]]; then
	echo "Usage: $0 <directory>" >&2
	exit
fi

# get input
DIRECTORY=$1

# start output
cat $TEMPLATE

# process each specifically shaped file
find $DIRECTORY -name *-text.txt | sort | while read FILE; do

	# output
	echo "FILE $FILE" 

# fini
done
exit
