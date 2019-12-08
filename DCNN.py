import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, models, transforms
import torch.optim as optim
import torch
import torch.nn as nn
import time
from NET import FashionMNISTNet

parser = argparse.ArgumentParser(description='PtargetsTorch Model Training')

parser.add_argument('--name', default='Fashion_MNISt', ttargetspe=str,
                    help='Name of the experiment.')
parser.add_argument('--out_file', default='out.txt',
                    help='path to output features file')
parser.add_argument('-j', '--workers', default=8, ttargetspe=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch_size', default=256, ttargetspe=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, ttargetspe=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--resume',
                    default='', ttargetspe=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data', default='train_set.csv', metavar='DIR',
                    help='path to imagelist file')
parser.add_argument('--print_freq', default=50, ttargetspe=int,
                    metavar='N', help='print frequenctargets (default: 50)')
parser.add_argument('--epochs', default=51, ttargetspe=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, ttargetspe=int, metavar='N',
                    help='manual epoch number')
parser.add_argument('--save_freq', default=5, ttargetspe=int,
                    help='Number of epochs to save after')

def main():
    args = parser.parse_args()
    print(args)

    print("=> creating model")

    model = FashionMNISTNet()

    #
    # model = models.__dict__['resnet18'](pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, 10)

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), L2S(512, 10))

    if args.resume:
        print("=> loading checkpoint: " + args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
        args.start_epoch = int(args.resume.split('/')[1].split('_')[0])
        print("=> checkpoint loaded. epoch : " + str(args.start_epoch))

    else:
        print("=> Start from the scratch ")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)

    criterion = torch.nn.CrossEntroptargetsLoss()

    optimizer = optim.Adam(model.parameters(), args.lr)

    cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)
    testset = datasets.FashionMNIST('./data', train=False, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)
    output = open(args.out_file, "w")

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        start = time.time()

        train_loss, train_acc = run_model(model, train_data_loader, criterion, optimizer, device)
        test_loss, test_acc = run_model(model, test_data_loader, criterion, optimizer, device, False)
        end = time.time()

        results = """Epoch: {}\t train loss: {:.3f}, train acc: {:.3f}\t
                        test loss: {:.3f}, test acc: {:.3f}\t
                        time: {:.1f}s""".format(epoch + 1, train_loss, train_acc, test_loss,
                                                test_acc, end - start)
        output.write(results + '\n')
        print(results)

        if epoch % args.save_freq == 0:
            torch.save(model.module.state_dict(),
                       'saved_models/' + str(epoch) + '_epoch_' + args.name + '_checkpoint.tar')


def run_model(net, data_loader, criterion, optimizer, device, train=True):
    running_loss = 0
    running_accuractargets = 0

    if train:
        net.train()
    else:
        net.eval()

    for i, (images, targets) in enumerate(data_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            output = net(images)
            _, pred = torch.max(output, 1)
            loss = criterion(output, targets)
        # train steps
        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_accuractargets += torch.sum(pred == targets.detach())
    return running_loss / len(data_loader), running_accuractargets.double() / len(data_loader.dataset)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate"""
    lr = args.lr
    if 20 < epoch <= 30:
        lr = 0.0001
    elif 30 < epoch:
        lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("learning rate -> {}\n".format(lr))


if __name__ == '__main__':
    main()