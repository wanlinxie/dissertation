#! /usr/bin/env python


from __future__ import division, with_statement
import os
import re
import subprocess
import tempfile
import text.sentence
from nltk.corpus import conll2000
import nltk

conjunctions = set()

class Nltktagchunk(object):
    """Interface to the TagChunk tool.
    """
    def __init__(self, path='/path/to/project/tools/chunker/'):
        """Initialize the path to TagChunk and regular expressions for
        processing its BIO-tagged output format.
        """
        self.path = path

        self.token_re = re.compile(r"^(.*)_(.*)_(.)-(.*)$")

    def run_on_corpus(self, corpus):
        """Write sentences to a temporary file as strings of words, run
        TagChunk on the file and retrieve the tagged results, then delete the
        file.
        """
        # Check if the corpus consists of Sentences or MultiSentences, and
        # get a single list of Sentences either way
        sentences = []
        if corpus[0].__class__ == text.sentence.MultiSentence:
            for multisentence in corpus:
                # Collect the Sentence objects from each MultiSentence
                sentences.extend(multisentence.sentences)
        else:
            sentences = corpus
        
        train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP','VP','PP'])
        unigram_chunker = UnigramChunker(train_sents)

        strings_BIO = []
        
        for sentence in sentences:
            sentence_text = ' '.join(sentence.tokens)
            tags = [t[1] for t in nltk.pos_tag(sentence_text.split())]

            sentence_arr = sentence_text.split(" ")
            bio = unigram_chunker.tagger.tag(tags)
            
            temp = ""
            for i in range(0,len(sentence_arr)):
                if str(bio[i][1]) == "O":
                    temp += sentence_arr[i]+"_"+bio[i][0]+"_B-"+str(bio[i][1])+ " "
                else:
                    temp += sentence_arr[i]+"_"+bio[i][0]+"_"+str(bio[i][1])+" "

            temp = temp[:-1]
            strings_BIO.append(temp)

        #print(strings_BIO)

        # Process sentence
        for sentence, string_BIO in zip(sentences, strings_BIO):
            #print("debugging")
            pos_tags, chunks = self.process_BIO_string(string_BIO)

            sentence.add_token_tags(pos_tags, name='pos_tags',
                    annotator='chunker')

            sentence.add_span_tags(chunks, name='chunks',
                    annotator='chunker')

    def process_BIO_string(self, string_BIO):
        """Convert a sentence string consisting of space-separated words in
        the format word_POS_BIOtag-chunktype into a tuple of POS tags and a
        tuple-to-string dictionary of chunk spans.
        """
        tokens_BIO = string_BIO.split(' ')

        pos_tags = []
        chunks = {}
        current_span_start = None
        current_type = ''
        for i, token_BIO in enumerate(tokens_BIO):
            match_obj = re.match(self.token_re, token_BIO)
            if match_obj:
                word, POS_tag, BIO_tag, chunk_type = match_obj.groups()

                # Append POS to list of pos_tags
                pos_tags.append(POS_tag)
                if POS_tag == 'IN':
                    conjunctions.add(word.lower())

                # If an incomplete chunk exists and is ended by the current
                # token, finish the chunk and add it to the dictionary of
                # chunks
                if current_span_start and BIO_tag != 'I':
                    span = current_span_start, i-1
                    chunks[span] = current_type

                    # Clear current span for future iterations
                    current_span_start = None

                # If the current word begins a new chunk
                if BIO_tag == 'B' and chunk_type != 'O':
                    current_span_start = i
                    current_type = chunk_type
            else:
                print "ERROR: Can't parse BIO string", token_BIO
                raise Exception

        # If an incomplete chunk remains, wrap it up
        if current_span_start:
            span = current_span_start, len(tokens_BIO)-1
            chunks[span] = current_type

        return pos_tags, chunks

class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)
    
    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)