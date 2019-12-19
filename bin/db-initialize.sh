#!/usr/bin/env bash

# db-initialize.sh - given a directory and an identifier, create a database of a specific shape

# Eric Lease Morgan <eric_morgan@infomotions.com>
# December 11, 2019 - first cut; "Happy Birthday, Lincoln!"


SCHEMA='./etc/schema.sql'
DATABASE='reader.db'

# sanity check
if [[ -z $1 ]]; then

	echo "Usage: $0 <directory>" >&2
	exit

fi

# get input
DIRECTORY=$1

# initialize, do the work, and done
rm -rf "$DIRECTORY/$DATABASE"
cat $SCHEMA | sqlite3 "$DIRECTORY/$DATABASE"
exit