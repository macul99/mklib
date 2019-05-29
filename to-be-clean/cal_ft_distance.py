import numpy as np

# load feature and calculate their distance
ref_ft_path='/home/macul/Pictures/debug1/0c34d8897e954592bd75517a6d261c49.txt'

ft_path_list=[	'/home/macul/Pictures/debug1/1k1f2371f23ub5gaerhiwavscgk3cqfa.txt',
				'/home/macul/Pictures/debug1/3m91nyyub67c72820al0a7coa53wx1mv.txt',
				'/home/macul/Pictures/debug1/7c7aw2ot03f4t8r2cagetqkyq97ckzav.txt',
				'/home/macul/Pictures/debug1/a3hafdjumjra3ue597x5h3n6xdgr8u0u.txt',
				'/home/macul/Pictures/debug1/ooj07m5zfdd0f197jaeay5a6bj6krv5a.txt',
				'/home/macul/Pictures/debug1/qh7jmt6seeaesmdymjamibs93c5af75u.txt',
				'/home/macul/Pictures/debug1/vdfnn8m82msebq4rh0x507uvgnibfbdd.txt',
				'/home/macul/Pictures/debug1/wbzlvrg92u87dgldgzk897g6e2vo25x2.txt',
				'/home/macul/Pictures/debug1/wl9t31djmef09iw5n1cnntx4ax4ut7ob.txt',
				'/home/macul/Pictures/debug1/xwxn39hrci0notv469blcfa8buaikq6v.txt',
				'/home/macul/Pictures/debug1/zeoaeo0qfwzquy8v28a9vj1rr1q4bsea.txt']
ref_ft=[]

with open(ref_ft_path) as f:

	for v in f.read().split():
		ref_ft += [float(v)]

ft=[]

for i in range(len(ft_path_list)):
	tmp_ft=[]
	with open(ft_path_list[i]) as f:
		for v in f.read().split():
			tmp_ft += [float(v)]
	ft += [tmp_ft]

ref_ft=np.array(ref_ft)
ft=np.array(ft)


for i in range(ft.shape[0]):
	#print('angle: ', np.arccos(np.dot(ref_ft,ft[i]))/np.pi*180)
	print('norm: ', np.linalg.norm(ft[i]))

#print(ft)
