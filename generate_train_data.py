import os, sys, math,random
import numpy as np

cell=sys.argv[1]
data_ratio=sys.argv[2]

data_file='%s/' % cell
save_dir='preprocess/data/%s/' % cell

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def hic_matrix_extraction(res=10000,norm_method='NONE'):
    chrom_list = list(range(1,23))#chr1-chr22
    hr_contacts_dict={}
    for each in chrom_list:
        # hr_hic_file = data_file+'data/%s/intra_%s/chr%d_10k_intra_%s.txt'%(DPATH,norm_method,each,norm_method)
        hr_hic_file = data_file+'intra_%s/chr%d_10k_intra_%s.txt'%(norm_method,each,norm_method)
        chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open('chromosome.txt').readlines()}
        mat_dim = int(math.ceil(chrom_len['chr%d'%each]*1.0/res))
        hr_contact_matrix = np.zeros((mat_dim,mat_dim))
        for line in open(hr_hic_file).readlines():
            idx1, idx2, value = int(line.strip().split('\t')[0]),int(line.strip().split('\t')[1]),float(line.strip().split('\t')[2])
            if idx2/res>=mat_dim or idx1/res>=mat_dim:
                continue
            else:
                hr_contact_matrix[int(idx1/res)][int(idx2/res)] = value
        hr_contact_matrix+= hr_contact_matrix.T - np.diag(hr_contact_matrix.diagonal())
        hr_contacts_dict['chr%d'%each] = hr_contact_matrix
    lr_contacts_dict={}
    for each in chrom_list:
        # lr_hic_file = data_file+'data/%s/intra_%s/chr%d_10k_intra_%s_downsample_ratio%s.txt'%(DPATH,norm_method,each,norm_method,data_ratio)
        lr_hic_file = data_file+'intra_%s/chr%d_10k_intra_%s_downsample_ratio%s.txt'%(norm_method,each,norm_method,data_ratio)
        chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open('chromosome.txt').readlines()}
        mat_dim = int(math.ceil(chrom_len['chr%d'%each]*1.0/res))
        lr_contact_matrix = np.zeros((mat_dim,mat_dim))
        for line in open(lr_hic_file).readlines():
            idx1, idx2, value = int(line.strip().split('\t')[0]),int(line.strip().split('\t')[1]),float(line.strip().split('\t')[2])
            if idx2/res>=mat_dim or idx1/res>=mat_dim:
                continue
            else:
                lr_contact_matrix[int(idx1/res)][int(idx2/res)] = value
        lr_contact_matrix+= lr_contact_matrix.T - np.diag(lr_contact_matrix.diagonal())
        lr_contacts_dict['chr%d'%each] = lr_contact_matrix

    nb_hr_contacts={item:sum(sum(hr_contacts_dict[item])) for item in hr_contacts_dict.keys()}
    nb_lr_contacts={item:sum(sum(lr_contacts_dict[item])) for item in lr_contacts_dict.keys()}
    
    return hr_contacts_dict,lr_contacts_dict,nb_hr_contacts,nb_lr_contacts

hr_contacts_dict,lr_contacts_dict,nb_hr_contacts,nb_lr_contacts = hic_matrix_extraction()

def crop_hic_matrix_by_chrom(chrom,size=40 ,thred=200):
    #thred=2M/resolution
    distance=[]
    crop_mats_hr=[]
    crop_mats_lr=[]    
    row,col = hr_contacts_dict[chrom].shape
    if row<=thred or col<=thred:
        print('HiC matrix size wrong!')
        sys.exit()
    def quality_control(mat,thred=0.05):
        if len(mat.nonzero()[0])<thred*mat.shape[0]*mat.shape[1]:
            return False
        else:
            return True
        
    for idx1 in range(0,row-size,size):
        for idx2 in range(0,col-size,size):
            if abs(idx1-idx2)<thred:
                if quality_control(lr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]):
                    distance.append([idx1-idx2,chrom])

                    lr_contact = lr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                    hr_contact = hr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                    
                    crop_mats_lr.append(lr_contact)
                    crop_mats_hr.append(hr_contact)
    crop_mats_hr = np.concatenate([item[np.newaxis,:] for item in crop_mats_hr],axis=0)
    crop_mats_lr = np.concatenate([item[np.newaxis,:] for item in crop_mats_lr],axis=0)
    return crop_mats_hr,crop_mats_lr,distance
def data_split(chrom_list):
    random.seed(100)
    distance_all=[]
    assert len(chrom_list)>0
    hr_mats,lr_mats=[],[]
    for chrom in chrom_list:
        crop_mats_hr,crop_mats_lr,distance = crop_hic_matrix_by_chrom(chrom,size=40 ,thred=200)
        distance_all+=distance
        hr_mats.append(crop_mats_hr)
        lr_mats.append(crop_mats_lr)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_mats=hr_mats.transpose((0,2,3,1))
    lr_mats=lr_mats.transpose((0,2,3,1))
    return hr_mats,lr_mats,distance_all

hr_mats_train,lr_mats_train,distance_train = data_split(['chr%d'%idx for idx in list(range(1,18))])
hr_mats_test,lr_mats_test,distance_test = data_split(['chr%d'%idx for idx in list(range(18,23))])
np.savez(save_dir+'train_data_raw_ratio%s.npz'%(data_ratio), train_lr=lr_mats_train,train_hr=hr_mats_train,distance=distance_train)
np.savez(save_dir+'test_data_raw_ratio%s.npz'%(data_ratio), test_lr=lr_mats_test,test_hr=hr_mats_test,distance=distance_test)