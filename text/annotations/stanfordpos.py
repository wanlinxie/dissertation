#! /usr/bin/env python
# Author: Wanlin Xie

from __future__ import division, with_statement
import os
import re
import subprocess
import tempfile
import text.sentence
import text.structure


class Stanfordpos(object):
    """Interface to the Stanford POS tagger.
    """
    def __init__(self,
            path='/path/to/project/tools/stanford-postagger-full-2018-10-16'):
        """Initialize the path to the Stanford tagger.
        """
        self.path = path
        self.token_re = re.compile(r"^(.*)_(.*)$") #????
        
    def run_on_corpus(self, corpus, show_output=False):
        """Write sentences to a temporary file as strings of words, run the
        Stanford parser on the file and retrieve the parsed results,
        then delete the file.
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

        # Generate a temporary file for the sentences
        f, temp_filepath = tempfile.mkstemp('.txt',
                                            'sents_',
                                            self.path,
                                            text=True)

        # Add in sentence terminators to prevent accidental sentence merging
        try:
            # Write sentences out to the temporary file as strings of words
            with os.fdopen(f, 'w') as temp_file:
                for sentence in sentences:
                    sentence_text = ' '.join(sentence.tokens)
                    print>>temp_file, sentence_text

            classname = 'edu.stanford.nlp.tagger.maxent.MaxentTagger'
            model_path = self.path+"/models/wsj-0-18-bidirectional-distsim.tagger"
            process = subprocess.Popen(['java',
                                        '-mx500m',
                                        '-cp', self.path + '/*',
                                        classname,
                                        '-model',
                                        model_path,
                                        '-textFile',
                                        temp_filepath],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if show_output:
                print stdout
                print stderr
        finally:
            # Delete temporary file
            #print("Dont delete")
            os.remove(temp_filepath)

        # Split output into strings of per-sentence parses
        parse_strings = stdout.split('\n');
        parse_strings.pop()

        # Process sentence
        for sentence, parse_string in zip(sentences, parse_strings):

            pos_tags = self.process_parse(parse_string)

            sentence.add_token_tags(pos_tags, name='pos_tags',
                    annotator='stanfordpos')

    def process_parse(self, parse):
        """Process the Stanford pos format.
        """
        tokens = parse.split(' ')
        pos_tags = []

        for i, token in enumerate(tokens):
            match_obj = re.match(self.token_re,token)
            if match_obj:
                word, POS_tag = match_obj.groups()
                pos_tags.append(POS_tag)
            else:
                print "ERROR: Can't parse string", token
                raise Exception
        #print(pos_tags)
        return pos_tags
