import argparse
import json
import torch
import codecs

def convert_instance_to_idx_seq(word_insts, word2idx):
	''' Mapping words to idx sequence. '''
	idx_seqs = []
	for inst in word_insts:
		idx_seq = []
		for token in inst:
			try:
				idx_seq.append(word2idx[token])
			except:
				idx_seq.append(word2idx['<UNK>'])
		idx_seqs.append(idx_seq)

	return idx_seqs

def read_image_instances_from_file(inst_file):
	''' Convert file into lists '''
	
	image_insts = []
	f = codecs.open(inst_file, encoding="utf-8").readlines()
	for sent in f:
		image_insts += [sent.strip()]

	return image_insts

def read_instances_from_file(inst_file, max_sent_len):
	''' Convert file into word seq lists '''

	word_insts = []
	trimmed_sent_count = 0
	with open(inst_file) as f:
		for sent in f:
			words = sent.split()
			if len(words) > max_sent_len:
				trimmed_sent_count += 1
			word_inst = words[:max_sent_len]

			if word_inst:
				word_insts += [['<BOS>'] + word_inst + ['<EOS>']]
			else:
				word_insts += [None]

	print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

	if trimmed_sent_count > 0:
		print('[Warning] {} instances are trimmed to the max sentence length {}.'
			.format(trimmed_sent_count, max_sent_len))

	return word_insts
			

def main():
	''' Main function '''

	parser = argparse.ArgumentParser()
	parser.add_argument('-train_img', required=True)
	parser.add_argument('-train_src', required=True)
	parser.add_argument('-train_tgt', required=True)
	parser.add_argument('-valid_img', required=True)
	parser.add_argument('-valid_src', required=True)
	parser.add_argument('-valid_tgt', required=True)
	parser.add_argument('-src_vocab', required=True)
	parser.add_argument('-tgt_vocab', required=True)
	parser.add_argument('-save_data', required=True)
	parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)

	opt = parser.parse_args()
	opt.max_token_seq_len = opt.max_word_seq_len + 2

	# Training set
	train_img_insts = read_image_instances_from_file(opt.train_img)
	train_src_word_insts = read_instances_from_file(opt.train_src, opt.max_word_seq_len)
	train_tgt_word_insts = read_instances_from_file(opt.train_tgt, opt.max_word_seq_len)

	if not (len(train_img_insts) == len(train_src_word_insts) == len(train_tgt_word_insts)):
		print('[Warning] The training instance count is not equal!')
		exit()

	# Validation set
	valid_img_insts = read_image_instances_from_file(opt.valid_img)
	valid_src_word_insts = read_instances_from_file(opt.valid_src, opt.max_word_seq_len)
	valid_tgt_word_insts = read_instances_from_file(opt.valid_tgt, opt.max_word_seq_len)

	if not (len(valid_img_insts) == len(valid_src_word_insts) == len(valid_tgt_word_insts)):
		print ('[Warning] The validation instance count is not equal')
		exit()

	# Build vocabulary
	src_word2idx = json.load(open(opt.src_vocab))
	tgt_word2idx = json.load(open(opt.tgt_vocab))

	# word to index
	print('[Info] Convert source word instances into sequences of word index.')
	train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
	valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

	print('[Info] Convert target word instances into sequences of word index.')
	train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
	valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

	data = {
		'settings': opt,
		'dict': {
			'src': src_word2idx,
			'tgt': tgt_word2idx},
		'train': {
			'img': train_img_insts,
			'src': train_src_insts,
			'tgt': train_tgt_insts},
		'valid': {
			'img': valid_img_insts, 
			'src': valid_src_insts,
			'tgt': valid_tgt_insts}}

	print('[Info] Dumping the processed data to pickle file', opt.save_data)
	torch.save(data, opt.save_data)
	print('[Info] Finish.')
	

if __name__ == '__main__':
	main()
