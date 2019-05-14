#!/usr/bin/env python

import instance
import pickle
import unicodedata

#testfile = './test_data.pkl'

class FusionCorpus(object): # collections of Fusion Instances

	def __init__(self, filepath):
		self.filepath = filepath
		
	
	def parse_from_pkl(self):
		fusions = []
		f = open(self.filepath,'rb')
		raw = pickle.load(f)
		#print(raw)
		for i in raw:
			#print(i[0], i[1], i[2])
			fusions.append(instance.FusionInstance(
				file_id = i[0],
				output_sent = i[1],
				input_sents = i[2]))
		f.close()
		return fusions
	
	@staticmethod
	def normalize(text):
		return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore') \
		if isinstance(text, unicode) \
		else text

	def export_instances(self, corpus): 
		fusion_instances = self.parse_from_pkl()
		#print("length of fusion instances:", len(fusion_instances))

		# Append fusion instances to the corpus
		for instance in fusion_instances:
			output_sents = [self.normalize(instance.output_sent)]
			print(output_sents)
			input_sents = [[self.normalize(sent)] for sent in instance.input_sents]
			print(input_sents)
			corpus.add_instance(input_sents, output_sents)

		num_instances = len(corpus.instances)


		train_test = int(num_instances*0.8)
		print("train_test", train_test)
		corpus.set_slices(train=(0,train_test), \
			test=(train_test, num_instances))

		if num_instances == 0:
			print("Zero corpus instances")
		else:
			print ' '.join(("Train:", str(len(corpus.train_instances)),
				str(len(corpus.train_instances) / float(num_instances) * 100) + '%'))
			print ' '.join(("Test:", str(len(corpus.test_instances)),
				str(len(corpus.test_instances) / float(num_instances) * 100) + '%'))

def create_empty_corpus(tcorpus, ecorpus):
	train_test = len(ecorpus.instances)

	for i in tcorpus.test_instances:
		e_input_sents = [[sent.raw] for sent in i.input_sents]
		e_output_sents = [sent.raw for sents in i.gold_sentences for sent in sents.sentences]
		ecorpus.add_instance(e_input_sents,e_output_sents)

	num_instances = len(ecorpus.instances)
	ecorpus.set_slices(train=(0,train_test), \
		test=(train_test, num_instances))

	if num_instances == 0:
		print("Zero corpus instances when splitting")
	else:
		print ' '.join(("Train:", str(len(ecorpus.train_instances)),
			str(len(ecorpus.train_instances) / float(num_instances) * 100) + '%'))
		print ' '.join(("Test:", str(len(ecorpus.test_instances)),
			str(len(ecorpus.test_instances) / float(num_instances) * 100) + '%'))

def create_split_corpus(tcorpus, scorpus, sarr):
	for i in tcorpus.train_instances:
		check = False
		for g_s in i.gold_sentences:
			for s in g_s.sentences:
				for t in s.tokens:
					if t.lower() in sarr:
						check = True
		if check:
			s_input_sents = [[sent.raw] for sent in i.input_sents]
			s_output_sents = [sent.raw for sents in i.gold_sentences for sent in sents.sentences]
			scorpus.add_instance(s_input_sents,s_output_sents)

	train_test = len(scorpus.instances)
	# add all test instances at end
	for i in tcorpus.test_instances:
		s_input_sents = [[sent.raw] for sent in i.input_sents]
		s_output_sents = [sent.raw for sents in i.gold_sentences for sent in sents.sentences]
		scorpus.add_instance(s_input_sents,s_output_sents)

	num_instances = len(scorpus.instances)
	scorpus.set_slices(train=(0,train_test), \
		test=(train_test, num_instances))

	if num_instances == 0:
		print("Zero corpus instances when splitting")
	else:
		print ' '.join(("Train:", str(len(scorpus.train_instances)),
			str(len(scorpus.train_instances) / float(num_instances) * 100) + '%'))
		print ' '.join(("Test:", str(len(scorpus.test_instances)),
			str(len(scorpus.test_instances) / float(num_instances) * 100) + '%'))




#if __name__ == '__main__':
#	fus_corpus = FusionCorpus(testfile)

	

	

