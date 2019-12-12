#!/usr/bin/env python

# extract-svo.py - given a plain text file, output the subject-verb-object combinations

# Eric Lease Morgan <emorgan@nd.edu>
# December 11, 20190 - first cut


# require
import spacy
import sys
import textacy
import textacy.preprocessing

# sanity check
if len( sys.argv ) != 2 :
	sys.stderr.write( 'Usage: ' + sys.argv[ 0 ] + " <file>\n" )
	quit()

# get input
file  = sys.argv[ 1 ]

# slurp up the text and normalize it
with open ( file ) as handle: text = handle.read()
text = textacy.preprocessing.normalize.normalize_quotation_marks( text )
text = textacy.preprocessing.normalize.normalize_hyphenated_words( text )
text = textacy.preprocessing.normalize.normalize_whitespace( text )

# initialize model
maximum = len( text ) + 1
nlp     = spacy.load( 'en', max_length=maximum )

# model
doc = nlp( text )

# output header and process each svo combination; do the work
print( 'subject', 'verb', 'object', sep='\t' )
for item in textacy.extract.subject_verb_object_triples( doc ) :

	# parse & output
	subject = item[ 0 ].text.lower()
	verb    = item[ 1 ].text.lower()
	object  = item[ 2 ].text.lower()
	print( subject, verb, object, sep='\t' )
	
# done
exit()
