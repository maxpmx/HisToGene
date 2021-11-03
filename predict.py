import torch
from torch.utils.data import DataLoader
from utils import *
from vis_model import SpatialTransformer
import warnings
from dataset import ViT_HER2ST, ViT_SKIN
from tqdm import tqdm
warnings.filterwarnings('ignore')


MODEL_PATH = ''

# device = 'cpu'
def model_predict(model, test_loader, adata=None, attention=True, device = torch.device('cpu')): 
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, exp, center in tqdm(test_loader):

            patch, position = patch.to(device), position.to(device)
            
            pred = model(patch, position)


            if preds is None:
                preds = pred.squeeze()
                ct = center
                gt = exp
            else:
                preds = torch.cat((preds,pred),dim=0)
                ct = torch.cat((ct,center),dim=0)
                gt = torch.cat((gt,exp),dim=0)
                
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct

    return adata, adata_gt

def sr_predict(model, test_loader, attention=True,device = torch.device('cpu')):
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, center in tqdm(test_loader):
            
            patch, position = patch.to(device), position.to(device)
            pred = model(patch, position)
            
            if preds is None:
                preds = pred.squeeze()
                ct = center
            else:
                preds = torch.cat((preds,pred),dim=0)
                ct = torch.cat((ct,center),dim=0)
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct


    return adata

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # for fold in [5,11,17,26]:
    for fold in range(12):
        # fold=30
        # tag = '-vit_skin_aug'
        # tag = '-cnn_her2st_785_32_cv'
        tag = '-vit_her2st_785_32_cv'
        # tag = '-cnn_skin_134_cv'
        # tag = '-vit_skin_134_cv'
        ds = 'HER2'
        # ds = 'Skin'

        print('Loading model ...')
        # model = STModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = VitModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = STModel.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt") 
        model = SpatialTransformer.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt")
        model = model.to(device)
        # model = torch.nn.DataParallel(model)
        print('Loading data ...')

        # g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        g = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))

        # dataset = SKIN(train=False,ds=ds,fold=fold)
        dataset = ViT_HER2ST(train=False,mt=False,sr=True,fold=fold)
        # dataset = ViT_SKIN(train=False,mt=False,sr=False,fold=fold)
        # dataset = VitDataset(diameter=112,sr=True)

        test_loader = DataLoader(dataset, batch_size=16, num_workers=4)
        print('Making prediction ...')

        adata_pred, adata = model_predict(model, test_loader, attention=False)
        # adata_pred = sr_predict(model,test_loader,attention=True)

        adata_pred.var_names = g
        print('Saving files ...')
        adata_pred = comp_tsne_km(adata_pred,4)
        # adata_pred = comp_umap(adata_pred)
        print(fold)
        print(adata_pred)

        adata_pred.write('processed/test_pred_'+ds+'_'+str(fold)+tag+'.h5ad')
        # adata_pred.write('processed/test_pred_sr_'+ds+'_'+str(fold)+tag+'.h5ad')

        # quit()

if __name__ == '__main__':
    main()

