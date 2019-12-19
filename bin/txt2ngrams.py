#!/usr/bin/env python

# txt2ngrams.py - given the number of tokens (n) and a plain text file, output phrases of n sizes

# Eric Lease Morgan <emorgan@nd.edu>
# December 11, 2019 - first cut; "Happy Birthday, Lincoln!"


# require
import spacy
import sys
import textacy
import textacy.preprocessing

# sanity check
if len( sys.argv ) != 3 :
	sys.stderr.write( 'Usage: ' + sys.argv[ 0 ] + " <number> <file>\n" )
	quit()

# get input
size  = int( sys.argv[ 1 ] )
file  = sys.argv[ 2 ]

# slurp up the text and normalize it
with open ( file ) as handle: text = handle.read()
text = textacy.preprocessing.normalize.normalize_quotation_marks( text )
text = textacy.preprocessing.normalize.normalize_hyphenated_words( text )
text = textacy.preprocessing.normalize.normalize_whitespace( text )

# initialize model
maximum = len( text ) + 1
model   = spacy.load( 'en', max_length=maximum )

# model the data; this needs to be improved
doc = model( text )

# output header
print( '\t'.join( [ 'ngram', 'ngram', 'ngram' ] ) )

# do the work
for ngrams in textacy.extract.ngrams( doc, size ) :
	
	# output
	ngrams = ngrams.text.lower().split( ' ' )
	print( '\t'.join( ngrams ) )

# done
exit()
