#!/usr/bin/env python

import numpy as np


class FusionInstance(object):

	def __init__(self, file_id, output_sent, input_sents):

		self.file_id = file_id # int
		self.output_sent = output_sent # string
		self.input_sents = input_sents # list of strings

	def get_input_len(self):
		return len(input_sents)

	def get_length_ratio(self):
		return np.mean([len(input_sent.split()) 
			for input_sent in self.input_sents]) / \
				len(self.output_sent.split())

	def print_summary(self):
		print("file id: ", self.file_id, '\n')
		for input_sent in self.input_sents:
			print ("IN: ", input_sent, '\n')

		print("OUT: ", output_sent, '\n\n')