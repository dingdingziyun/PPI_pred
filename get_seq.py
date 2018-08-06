
def get_org_ppi(file_name, org_id):
	l=[]
	#org_id='10090'
	biogrid_file=open(file_name).read().split('\n')
	for lines in biogrid_file:
		if len(lines.split('\t'))==24:
			orgA=lines.split('\t')[15]
			orgB=lines.split('\t')[16]
			#print orgA+'\t'+orgB
			if orgA==org_id and orgB==org_id and lines.split('\t')[12]=='physical' and ((lines.split('\t')[5]+'\t'+lines.split('\t')[6]) not in l) and ((lines.split('\t')[6]+'\t'+lines.split('\t')[5]) not in l):
				#print lines.split('\t')[7]+'\t'+lines.split('\t')[8]
				l.append(lines.split('\t')[5]+'\t'+lines.split('\t')[6])
	return l

def get_seq_dic(dictionary_name):
	seq_dic=dict()
	dict_file=open(dictionary_name).read().split('//\n')[0:-1]
	for prot in dict_file:
		if prot.find('DR   TAIR')!=-1:
			prot_id=prot.split('DR   TAIR; ')[1].split('\n')[0].split('; ')[1].replace('.','')
			#print prot_id
			prot_seq=''.join(prot.split('SQ   SEQUENCE')[1].split('\n')[1:]).replace(' ','').replace('X','').replace('U','')
			#print prot_seq
			seq_dic[prot_id]=prot_seq
	return seq_dic
#change path and org_id
ppi_list=get_org_ppi('../dataset/BIOGRID-MV-Physical-3.4.158.tab2.txt','3702')
mouse_seq_dic=get_seq_dic('../dataset/uniprot-proteome%3AUP000006548.txt')
#print mouse_seq_dic
#print len(ppi_list)


#change output path
#change org_name
def get_sequence(ppi_list, output_seq, output_id):
	output_seq=open(output_seq,'w')
	output_id=open(output_id,'w')
	for ppi in ppi_list:
		if ppi.split()[0].upper() in mouse_seq_dic and ppi.split()[1].upper() in mouse_seq_dic:
			if len(mouse_seq_dic[ppi.split()[0].upper()])>=50 and len(mouse_seq_dic[ppi.split()[1].upper()])>=50:
				output_seq.write(mouse_seq_dic[ppi.split()[0].upper()]+','+mouse_seq_dic[ppi.split()[1].upper()]+'\n')
				output_id.write(ppi.split()[0].upper()+'\t'+ppi.split()[1].upper()+'\n')
	output_seq.close()
	output_id.close()

neg_pairs=open('../dataset/neg_pairs').read().split('\n')

get_sequence(ppi_list, '../dataset/ara_seq', '../dataset/ara_id')
get_sequence(neg_pairs,'../dataset/neg_seq','../dataset/neg_id')
