import wml.img_utils as wmli
import argparse
import os.path as osp
import os
import wml.wml_utils as wmlu

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('save_dir', type=str, help='source imgs directory')
    parser.add_argument('datadir_root', type=str, help='output imgs directory')
    parser.add_argument('-d','--dataset', type=str, help='output img fmt')
    parser.add_argument('--gpu', type=str, default="0", help='gpu')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    save_dir = args.save_dir
    datadir_root = args.datadir_root
    dataset = args.dataset
    gpu = args.gpu
    if dataset in ['can','walnuts']:
        pass
    else:
        os.chdir("./glass2")
        data_path = osp.join(datadir_root,"datasets")
        aug_path = osp.join(datadir_root,"dtd")
        cmd = f"python eval.py --results_path {save_dir} --gpu {gpu} --seed 0 --test ckpt net -b auto --pretrain_embed_dimension 512 --target_embed_dimension 512 --patchsize 3 --meta_epochs 640  --dsc_layers 2 --dsc_hidden 1024 --pre_proj 1 --mining 1 --noise 0.015 --radius 0.75 --p 0.5 --step 20 --limit 392 "\
            f"dataset --distribution 0 --mean 0.5 --std 0.1 --fg 1 --rand_aug 1 --resize 512 --batch_size 5 --align -1 --data_path {data_path} --aug_path {aug_path} -d {dataset}"
        print(f"command: {cmd}")
        os.system(cmd)



    