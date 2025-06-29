import numpy
import torch
import itertools
import pandas
import util.autoencoder as ae
import util.dopnet as dp
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from util.ml import get_k_folds_list
import os

dataset_path = 'dataset/expanded_formula.xlsx'
max_dops = 5
n_folds = 3

os.makedirs('save/loss', exist_ok=True)
os.makedirs('save/pred', exist_ok=True)
os.makedirs('save/emb', exist_ok=True)

df = pandas.read_excel(dataset_path)
columns = df.columns.tolist()

target_name_map = {2: 'S', 3: 'sigma', 4: 'PF', 5: 'tc', 6: 'zt'}

for target_idx in [4]:
    target_name = target_name_map[target_idx]
    init_lr = 1e-3 if target_name in ['sigma', 'S', 'PF'] else 1e-1
    print(f"\n=========== Training on: {columns[target_idx]} ===========")

    dataset = dp.load_dataset(dataset_path, comp_idx=0, target_idx=target_idx, max_dops=max_dops, cond_idx=[1])

    if target_name == 'PF':
        for d in dataset:
            d.target *= 1e5

    rand_idx = numpy.random.permutation(len(dataset))
    rand_dataset = [dataset[idx] for idx in rand_idx]
    k_folds = get_k_folds_list(rand_dataset, k=n_folds)

    list_test_mae, list_test_rmse, list_test_r2 = [], [], []
    list_preds, list_embs = [], []
    train_losses, test_losses = [], []

    for k in range(n_folds):
        print(f'------ Fold [{k + 1}/{n_folds}] ------')

        dataset_train = list(itertools.chain(*(k_folds[:k] + k_folds[k + 1:])))
        targets_train = numpy.array([x.target for x in dataset_train]).reshape(-1, 1)
        dop_dataset_train = dp.get_dataset(dataset_train, max_dops)
        data_loader_train = DataLoader(dop_dataset_train, batch_size=32, shuffle=True)
        data_loader_calc = DataLoader(dop_dataset_train, batch_size=32)

        dataset_test = k_folds[k]
        targets_test = numpy.array([x.target for x in dataset_test]).reshape(-1, 1)
        dop_dataset_test = dp.get_dataset(dataset_test, max_dops)
        data_loader_test = DataLoader(dop_dataset_test, batch_size=32)

        emb_host = ae.Autoencoder(dataset[0].host_feat.shape[0], 64).cuda()
        optimizer_emb = torch.optim.Adam(emb_host.parameters(), lr=1e-3, weight_decay=1e-5)

        for epoch in range(300):
            train_loss_emb = ae.train(emb_host, data_loader_train, optimizer_emb)
            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/300]  Emb Loss: {train_loss_emb:.4f}')

        host_embs_train = ae.test(emb_host, data_loader_calc)
        host_embs_test = ae.test(emb_host, data_loader_test)

        dop_dataset_train.host_feats = host_embs_train
        dop_dataset_test.host_feats = host_embs_test
        data_loader_train = DataLoader(dop_dataset_train, batch_size=32, shuffle=True)
        data_loader_test = DataLoader(dop_dataset_test, batch_size=32)

        pred_model = dp.DopNet(
            host_embs_train.shape[1],
            dataset[0].dop_feats.shape[1],
            dim_out=1,
            max_dops=max_dops
        ).cuda()

        optimizer = torch.optim.SGD(pred_model.parameters(), lr=init_lr, weight_decay=1e-7)
        criterion = torch.nn.L1Loss()

        for epoch in range(1000):
            if (epoch + 1) % 200 == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.5

            train_loss = dp.train(pred_model, data_loader_train, optimizer, criterion)
            preds_test = dp.test(pred_model, data_loader_test).cpu().numpy()
            test_loss = mean_absolute_error(targets_test, preds_test)
            train_losses.append((k, epoch + 1, train_loss))
            test_losses.append((k, epoch + 1, test_loss))

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/1000]  Train: {train_loss:.4f}  Test: {test_loss:.4f}')

        preds_test = dp.test(pred_model, data_loader_test).cpu().numpy()
        embs_test = dp.emb(pred_model, data_loader_test).cpu().numpy()

        targets_test_out = targets_test
        preds_test_out = preds_test

        list_test_mae.append(mean_absolute_error(targets_test_out, preds_test_out))
        list_test_rmse.append(numpy.sqrt(mean_squared_error(targets_test_out, preds_test_out)))
        list_test_r2.append(r2_score(targets_test_out, preds_test_out))

        idx_test = numpy.array([x.idx for x in dataset_test]).reshape(-1, 1)
        list_preds.append(numpy.hstack([idx_test, targets_test_out, preds_test_out]))
        list_embs.append(numpy.hstack([idx_test, targets_test, embs_test]))

    df_loss = pandas.DataFrame(train_losses, columns=['fold', 'epoch', 'train_loss'])
    df_loss_test = pandas.DataFrame(test_losses, columns=['fold', 'epoch', 'test_loss'])
    df_loss_all = pandas.merge(df_loss, df_loss_test, on=['fold', 'epoch'])
    df_loss_all.to_csv(f'save/loss/loss_{target_name}.csv', index=False)

    pandas.DataFrame(numpy.vstack(list_preds), columns=['Index', 'Real', 'Pred']).to_csv(
        f'save/pred/preds_{target_name}.csv', index=False
    )
    pandas.DataFrame(numpy.vstack(list_embs)).to_csv(
        f'save/emb/embs_{target_name}.csv', header=False, index=False
    )

    print(f'Target {columns[target_idx]} summary:')
    print('MAE :', numpy.mean(list_test_mae))
    print('RMSE:', numpy.mean(list_test_rmse))
    print('R2  :', numpy.mean(list_test_r2))
