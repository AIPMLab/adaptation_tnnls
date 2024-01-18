import configargparse
import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from FocalLoss import FocalLoss
# from FocalLoss import FocalLossWithSmoothing

def get_parser(source, target, model, backbone):
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    # parser.add("--config", is_config_file=True, help='DSAN/DSAN.yaml',default="BNM/BNM.yaml")
    # parser.add("--config", is_config_file=True, help='DSAN/DSAN.yaml', default="DeepCoral/DeepCoral.yaml")
    parser.add("--config", is_config_file=True, help='DSAN/DSAN.yaml', default=model)
    # parser.add("--config", is_config_file=True, help='DSAN/DSAN.yaml', default="DAN/DAN.yaml")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)

    # network related
    parser.add_argument('--backbone', type=str, default=backbone)
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, default='./datasets/Multi/')
    # parser.add_argument('--data_dir', type=str, default='./datasets/BrainTumor/BT2')
    parser.add_argument('--src_domain', type=str, default=source)
    parser.add_argument('--tgt_domain', type=str, default=target)

    # training related
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False,
                        help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')
    return parser


def set_random_seed(seed):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True,
        num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True,
        num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return source_loader, target_train_loader, target_test_loader, n_class


def get_model(args):
    model = models.TransferNet(
        args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter,
        use_bottleneck=args.use_bottleneck).to(args.device)
    return model


def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
                                nesterov=False
                                )
    # optimizer = torch.optim.Adam(params,lr= args.lr,weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay,momentum=args.momentum,nesterov=True
    #                             )
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (
        -args.lr_decay))
    return scheduler


def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)

    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)

    acc = 100. * correct / len_target_dataset

    return acc, test_loss.avg



def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args, name):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch

    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0
    log = []

    # 创建空列表，以便存储训练和测试时的输出
    all_outputs_train = []
    # Focal = FocalLoss()
    for e in range(1, args.n_epoch + 1):
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)
        criterion = torch.nn.MSELoss()
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        for _ in range(n_batch):
            data_source, label_source = next(iter_source)
            data_target, _ = next(iter_target)
            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)
            # features = model.base_network(data_source)
            # features = model.bottleneck_layer(features)
            # pre = model.classifier_layer(features)
            clf_loss, transfer_loss = model(data_source, data_target, label_source)
            # clf_loss = Focal(pre, label_source)
            loss = clf_loss + args.transfer_loss_weight * transfer_loss
            # if stop > 1:
            #     contras = get_model(args)
            #     contras.load_state_dict(torch.load('./Model/Contras.pth'))
            #     features2 = contras.base_network(data_source)
            #     features2 = contras.bottleneck_layer(features2)
            #     pred = contras.classifier_layer(features2)
            #     mseloss = criterion(pre, pred)
            #     loss = loss + mseloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())

        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
            e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)


        # Test
        stop += 1
        test_acc, test_loss= test(model, target_test_loader, args)
        log.append([test_acc.item()])
        info += ', test_loss {:.4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        np_log = np.array(log, dtype=float)
        np.savetxt(name, np_log, delimiter=',', fmt='%.6f')
        if best_acc < test_acc:
            best_acc = test_acc
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)
        # torch.save(model, './Model/Contras.pth')
    print('Transfer result: {:.4f}'.format(best_acc))



def main(source, target,name, model, backbone):
    parser = get_parser(source, target, model, backbone)
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(0)
    source_loader, target_train_loader, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    optimizer = get_optimizer(model, args)

    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args, name)


if __name__ == "__main__":
    main('train', 'valid', name='1.csv', model= 'DANN/dann.yaml', backbone = 'resnet34')
    main('train', 'valid', name='2.csv', model= 'DAN/DAN.yaml', backbone = 'resnet34')
    main('train', 'valid', name='3.csv', model='DeepCoral/DeepCoral.yaml', backbone='resnet34')
    # main('train', 'valid', name='4.csv', model='DAN/DAN.yaml', backbone='googlenet')
    # main('train', 'valid', name='5.csv', model='DANN/dann.yaml', backbone='vgg')
    # main('train', 'valid', name='6.csv', model='DAN/DAN.yaml', backbone='vgg')
    # main('train', 'valid', name='7.csv', model='DANN/dann.yaml', backbone='resnet34')
    # main('train', 'valid', name='8.csv', model='DAN/DAN.yaml', backbone='resnet34')
    # main('train', 'valid', name='9.csv', model='DANN/dann.yaml', backbone='resnet50')
    # main('train', 'valid', name='10.csv', model='DAN/DAN.yaml', backbone='resnet50')
    # main('train', 'valid', name='11.csv', model='DANN/dann.yaml', backbone='resnet101')
    # main('train', 'valid', name='12.csv', model='DAN/DAN.yaml', backbone='resnet101')
    # main('train', 'valid', name='13.csv', model='DANN/dann.yaml', backbone='densenet')
    # main('train', 'valid', name='14.csv', model='DAN/DAN.yaml', backbone='densenet')
    # main('train', 'valid', name='15.csv', model='DANN/dann.yaml', backbone='shuffle')
    # main('train', 'valid', name='16.csv', model='DAN/DAN.yaml', backbone='shuffle')
    # main('train', 'valid', name='17.csv', model='DANN/dann.yaml', backbone='efficient')
    # main('train', 'valid', name='18.csv', model='DAN/DAN.yaml', backbone='efficient')
    # main('train', 'valid', name='19.csv', model='DANN/dann.yaml', backbone='mnasnet')
    # main('train', 'valid', name='20.csv', model='DAN/DAN.yaml', backbone='mnasnet')
    # main('webcam', 'amazon', name='WtoA.csv')
    # main('amazon', 'dslr', name='AtoD.csv')
    # main('dslr', 'amazon', name='DtoA.csv')
    # main('webcam', 'dslr', name='WtoD.csv')
    # main('dslr', 'webcam', name='DtoW.csv')



