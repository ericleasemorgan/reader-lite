#!/usr/bin/env bash

# db-initialize.sh - given a directory and an identifier, create a database of a specific shape

# Eric Lease Morgan <eric_morgan@infomotions.com>
# December 11, 2019 - first cut; "Happy Birthday, Lincoln!"


SCHEMA='./etc/schema.sql'

# sanity check
if [[ -z $1 || -z $2 ]]; then

	echo "Usage: $0 <directory> <identifier>" >&2
	exit

fi

# get input
DIRECTORY=$1
IDENTIFIER=$2

# initialize, do the work, and done
rm -rf "$DIRECTORY/$IDENTIFIER.db"
cat $SCHEMA | sqlite3 "$DIRECTORY/$IDENTIFIER.db"
exit