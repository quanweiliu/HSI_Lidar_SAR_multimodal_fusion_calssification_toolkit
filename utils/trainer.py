from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import time


# train for one epoch to learn unique features
def train(epoch, net, net_head, criterion, train_loader, test_loader, train_optimizer, args):
    total_loss = 0
    total_num = 0
    train_correct = 0
    test_correct = 0
    train_bar = tqdm(train_loader)
    
    start_time = time.time()
    net.train()
    net_head.train()
    for pos_1, pos_2, label in train_bar:

        label = label - 1
        pos_1, label = pos_1.to(args.device), label.to(args.device)
        out = net_head(net(pos_1))
        loss = criterion(out, label)

        pred = out.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()
        train_accuracy = 100. * train_correct.item() / len(train_loader.dataset)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size
        average_loss = total_loss / total_num
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} TRA: {:.4f}'.format(\
                                epoch, args.epochs, round(average_loss, 4), round(train_accuracy, 4)))

    if epoch % args.log_interval == 0:
        net.eval()
        net_head.eval()
        for t_pos_1, t_pos_2, t_label in test_loader:
            with torch.no_grad():
                t_label = t_label - 1
                t_pos_1, t_label = t_pos_1.to(args.device), t_label.to(args.device)
                out_e = net_head(net(t_pos_1))

                t_pred = out_e.data.max(1, keepdim=True)[1]
                test_correct += t_pred.eq(t_label.data.view_as(t_pred)).sum()
                test_accuracy = 100. * test_correct / len(test_loader.dataset)
                test_accuracy = round(test_accuracy.item(), 4)
        # print("linear test_accuracy", round(test_accuracy.item(), 4)) 
    else:
        test_accuracy = 0.0
    train_time = time.time() - start_time

    return round(average_loss, 4), round(train_accuracy, 4), test_accuracy


# train for one epoch to learn unique features
def train_SD_SSISO(epoch, net, net_head, criterion, train_loader, test_loader, train_optimizer, args):
    train_correct = 0
    test_correct = 0
    total_loss = 0
    # train_bar = tqdm(train_loader)
    
    start_time = time.time()
    net.train()
    net_head.train()
    for pos_1, label in train_loader:

        label = label - 1
        pos_1, label = pos_1.to(args.device), label.to(args.device)
        out = net_head(net(pos_1))
        loss = criterion(out, label)

        pred = out.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()
        train_accuracy = 100. * train_correct.item() / len(train_loader.dataset)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_loss += loss.item()  # 累计 batch 的损失

    average_loss = total_loss / len(train_loader)

    if epoch % args.log_interval == 0:
        net.eval()
        net_head.eval()
        for t_pos_1, t_label in test_loader:
            with torch.no_grad():
                t_label = t_label - 1
                t_pos_1, t_label = t_pos_1.to(args.device), t_label.to(args.device)
                out_e = net_head(net(t_pos_1))

                t_pred = out_e.data.max(1, keepdim=True)[1]
                test_correct += t_pred.eq(t_label.data.view_as(t_pred)).sum()
                test_accuracy = 100. * test_correct / len(test_loader.dataset)
                test_accuracy = round(test_accuracy.item(), 4)
        # print("linear test_accuracy", round(test_accuracy.item(), 4)) 
    else:
        test_accuracy = 0.0
    train_time = time.time() - start_time
    print('Train Epoch: [{}/{}] Loss: {:.4f} TRA: {:.4f} TEA: {:.4f} TIME: {:.4f}'.format(\
                    epoch, args.epochs, round(average_loss, 4), round(train_accuracy, 4), \
                        round(test_accuracy, 4), round(train_time, 2)))
    
    return round(average_loss, 4), round(train_accuracy, 4), test_accuracy, round(train_time, 2)


# train for one epoch to learn unique features
def train_SD_SSISO2(epoch, net, criterion, train_loader, test_loader, train_optimizer, args):
    train_correct = 0
    test_correct = 0
    total_loss = 0
    # train_bar = tqdm(train_loader)
    
    start_time = time.time()
    net.train()
    for pos_1, label in train_loader:

        label = label - 1
        pos_1, label = pos_1.to(args.device), label.to(args.device)
        out = net(pos_1)
        loss = criterion(out, label)

        pred = out.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()
        train_accuracy = 100. * train_correct.item() / len(train_loader.dataset)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_loss += loss.item()  # 累计 batch 的损失

    average_loss = total_loss / len(train_loader)

    if epoch % args.log_interval == 0:
        net.eval()
        for t_pos_1, t_label in test_loader:
            with torch.no_grad():
                t_label = t_label - 1
                t_pos_1, t_label = t_pos_1.to(args.device), t_label.to(args.device)
                out_e = net(t_pos_1)

                t_pred = out_e.data.max(1, keepdim=True)[1]
                test_correct += t_pred.eq(t_label.data.view_as(t_pred)).sum()
                test_accuracy = 100. * test_correct / len(test_loader.dataset)
                test_accuracy = round(test_accuracy.item(), 4)
        # print("linear test_accuracy", round(test_accuracy.item(), 4)) 
    else:
        test_accuracy = 0.0
    train_time = time.time() - start_time
    print('Train Epoch: [{}/{}] Loss: {:.4f} TRA: {:.4f} TEA: {:.4f} TIME: {:.4f}'.format(\
                    epoch, args.epochs, round(average_loss, 4), round(train_accuracy, 4), \
                        round(test_accuracy, 4), round(train_time, 2)))
    
    return round(average_loss, 4), round(train_accuracy, 4), test_accuracy, round(train_time, 2)


# train for one epoch to learn unique features
def train_SSISO(epoch, net, net_head, criterion, train_loader, test_loader, train_optimizer, args):
    train_correct = 0
    test_correct = 0
    total_loss = 0
    # train_bar = tqdm(train_loader)
    
    start_time = time.time()
    net.train()
    net_head.train()
    for pos_1, pos_2, label in train_loader:

        label = label - 1
        pos_1, label = pos_1.to(args.device), label.to(args.device)
        out = net_head(net(pos_1))
        loss = criterion(out, label)

        pred = out.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()
        train_accuracy = 100. * train_correct.item() / len(train_loader.dataset)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_loss += loss.item()  # 累计 batch 的损失

    average_loss = total_loss / len(train_loader)

    if epoch % args.log_interval == 0:
        net.eval()
        net_head.eval()
        for t_pos_1, t_pos_2, t_label in test_loader:
            with torch.no_grad():
                t_label = t_label - 1
                t_pos_1, t_label = t_pos_1.to(args.device), t_label.to(args.device)
                out_e = net_head(net(t_pos_1))

                t_pred = out_e.data.max(1, keepdim=True)[1]
                test_correct += t_pred.eq(t_label.data.view_as(t_pred)).sum()
                test_accuracy = 100. * test_correct / len(test_loader.dataset)
                test_accuracy = round(test_accuracy.item(), 4)
        # print("linear test_accuracy", round(test_accuracy.item(), 4)) 
    else:
        test_accuracy = 0.0
    train_time = time.time() - start_time
    print('Train Epoch: [{}/{}] Loss: {:.4f} TRA: {:.4f} TEA: {:.4f} TIME: {:.4f}'.format(\
                    epoch, args.epochs, round(average_loss, 4), round(train_accuracy, 4), \
                        round(test_accuracy, 4), round(train_time, 2)))
    
    return round(average_loss, 4), round(train_accuracy, 4), test_accuracy, round(train_time, 2)


# # train for one epoch to learn unique features
# def train_SMIMO(epoch, net, criterion, train_loader, test_loader, optimizer, args):
#     train_correct = 0
#     test_correct = 0
#     # train_bar = tqdm(train_loader)

#     start_time = time.time()
#     net.train()
#     for pos_1, pos_2, label in train_loader:

#         label = label - 1
#         pos_1  = pos_1.to(args.device)
#         pos_2  = pos_2.to(args.device)
#         label = label.to(args.device)

#         out1, out2, out3 = net(pos_1, pos_2)
#         loss1 = criterion(out1, label)
#         loss2 = criterion(out2, label)
#         loss3 = criterion(out3, label)
#         loss = loss1 + loss2 + loss3
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


#         out = out1 + out2 + out3
#         # pred = torch.max(out, 1)[1].squeeze()
#         pred = out.data.max(1, keepdim=True)[1]
#         train_correct += pred.eq(label.data.view_as(pred)).sum()
#         train_accuracy = 100. * train_correct.item() / len(train_loader.dataset)

#         # total_num += args.batch_size
#         # total_loss += loss.item() * args.batch_size
#         # average_loss = total_loss / total_num
#         # train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} TRA: {:.4f}'.format(\
#         #                         epoch, args.epochs, round(average_loss, 4), round(train_accuracy, 4)))

#     if epoch % args.log_interval == 0:
#         net.eval()
#         for t_pos_1, t_pos_2, t_label in test_loader:
#             with torch.no_grad():
#                 t_label = t_label - 1
#                 t_pos_1 = t_pos_1.to(args.device)
#                 t_pos_2 = t_pos_2.to(args.device)
#                 t_label = t_label.to(args.device)

#                 out_e1, out_e2, out_e3 = net(t_pos_1, t_pos_2)
#                 out_e = out_e1 + out_e2 + out_e3

#                 t_pred = out_e.data.max(1, keepdim=True)[1]
#                 test_correct += t_pred.eq(t_label.data.view_as(t_pred)).sum()
#                 test_accuracy = 100. * test_correct / len(test_loader.dataset)
#                 test_accuracy = round(test_accuracy.item(), 4)
#         # print("linear test_accuracy", round(test_accuracy.item(), 4)) 
#     else:
#         test_accuracy = 0.0

#     train_time = time.time() - start_time
#     print('Train Epoch: [{}/{}] Loss: {:.4f} TRA: {:.4f} TEA: {:.4f} TIME: {:.4f}'.format(\
#                     epoch, args.epochs, round(loss.cpu().item(), 4), round(train_accuracy, 4), \
#                         round(test_accuracy, 4), round(train_time, 2)))
    
#     return round(loss.cpu().item(), 4), round(train_accuracy, 4), test_accuracy, round(train_time, 2)


# train for one epoch to learn unique features
def train_SMIMO(epoch, net, superhead, criterion, train_loader, test_loader, optimizer, args):
    train_correct = 0
    test_correct = 0
    total_loss = 0
    # train_bar = tqdm(train_loader)

    start_time = time.time()
    net.train()
    for pos_1, pos_2, label in train_loader:

        label = label - 1
        pos_1  = pos_1.to(args.device)
        pos_2  = pos_2.to(args.device)
        label = label.to(args.device)

        _out1, _out2 = net(pos_1, pos_2)
        out1, out2, out3 = superhead(_out1, _out2)
        out = out1 + out2 + out3

        loss1 = criterion(out1, label)
        loss2 = criterion(out2, label)
        loss3 = criterion(out3, label)
        loss = loss1 + loss2 + loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # 累计 batch 的损失
        # pred = torch.max(out, 1)[1].squeeze()
        pred = out.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()
        train_accuracy = 100. * train_correct.item() / len(train_loader.dataset)

    average_loss = total_loss / len(train_loader)

    if epoch % args.log_interval == 0:
        net.eval()
        with torch.no_grad():
            for t_pos_1, t_pos_2, t_label in test_loader:
                t_label = t_label - 1
                t_pos_1 = t_pos_1.to(args.device)
                t_pos_2 = t_pos_2.to(args.device)
                t_label = t_label.to(args.device)

                _out1, _out2 = net(t_pos_1, t_pos_2)
                out_e1, out_e2, out_e3 = superhead(_out1, _out2)
                out_e = out_e1 + out_e2 + out_e3

                t_pred = out_e.data.max(1, keepdim=True)[1]
                test_correct += t_pred.eq(t_label.data.view_as(t_pred)).cpu().sum()

        test_accuracy = (100. * test_correct / len(test_loader.dataset)).item()
        # print("linear test_accuracy", round(test_accuracy.item(), 4)) 
    else:
        test_accuracy = 0.0

    train_time = time.time() - start_time
    print('Train Epoch: [{}/{}] Loss: {:.4f} TRA: {:.4f} TEA: {:.4f} TIME: {:.4f}'.format(\
                    epoch, args.epochs, round(average_loss, 4), round(train_accuracy, 4), \
                        round(test_accuracy, 4), round(train_time, 2)))
    
    return round(average_loss, 4), round(train_accuracy, 4), round(test_accuracy, 4), round(train_time, 2)


# train for one epoch to learn unique features
def train_SMIMO2(epoch, net, criterion, train_loader, test_loader, optimizer, args):
    train_correct = 0
    test_correct = 0
    total_loss = 0.0
    # train_bar = tqdm(train_loader)

    start_time = time.time()
    net.train()
    for pos_1, pos_2, label in train_loader:

        label = label - 1
        pos_1  = pos_1.to(args.device)
        pos_2  = pos_2.to(args.device)
        label = label.to(args.device)

        out1, out2, out3 = net(pos_1, pos_2)
        # print(out1.shape, out2.shape, out3.shape, pos_1.shape, pos_2.shape)
        loss = criterion(out1, label) + \
                1 * torch.mean(torch.pow(out2 - pos_1, 2)) + \
                1 * torch.mean(torch.pow(out3 - pos_2, 2))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # 累计 batch 的损失
        # pred = torch.max(out, 1)[1].squeeze()
        pred = out1.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()
        train_accuracy = (100. * train_correct / len(train_loader.dataset)).item()

    average_loss = total_loss / len(train_loader)

    if epoch % args.log_interval == 0:
        net.eval()
        with torch.no_grad():
            for t_pos_1, t_pos_2, t_label in test_loader:
                t_label = t_label - 1
                t_pos_1 = t_pos_1.to(args.device)
                t_pos_2 = t_pos_2.to(args.device)
                t_label = t_label.to(args.device)

                out_e, _out2, _out3 = net(t_pos_1, t_pos_2)

                t_pred = out_e.data.max(1, keepdim=True)[1]
                test_correct += t_pred.eq(t_label.data.view_as(t_pred)).cpu().sum()

        test_accuracy = (100. * test_correct / len(test_loader.dataset)).item()
        # print("linear test_accuracy", round(test_accuracy.item(), 4)) 
    else:
        test_accuracy = 0.0

    train_time = time.time() - start_time
    print('Train Epoch: [{}/{}] Loss: {:.4f} TRA: {:.4f} TEA: {:.4f} TIME: {:.4f}'.format(\
                    epoch, args.epochs, round(average_loss, 4), round(train_accuracy, 4), \
                    round(test_accuracy, 4), round(train_time, 2)))
    
    return round(average_loss, 4), round(train_accuracy, 4), round(test_accuracy, 4), round(train_time, 2)


# train for one epoch to learn unique features
def train_SMIMO3(epoch, net, criterion, train_loader, test_loader, optimizer, args):
    train_correct = 0
    test_correct = 0
    total_loss = 0.0
    # train_bar = tqdm(train_loader)

    start_time = time.time()
    net.train()
    for pos_1, pos_2, label in train_loader:

        label = label - 1
        pos_1  = pos_1.to(args.device)
        pos_2  = pos_2.to(args.device)
        label = label.to(args.device)

        out1, out2, out3 = net(pos_1, pos_2)
        # print(out1.shape, out2.shape, out3.shape, pos_1.shape, pos_2.shape)
        loss = criterion(out1, label) + \
                1 * torch.mean(torch.pow(out2 - out1, 2)) + \
                1 * torch.mean(torch.pow(out3 - out1, 2))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # 累计 batch 的损失
        # pred = torch.max(out, 1)[1].squeeze()
        pred = out1.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()
        train_accuracy = (100. * train_correct / len(train_loader.dataset)).item()

    average_loss = total_loss / len(train_loader)

    if epoch % args.log_interval == 0:
        net.eval()
        with torch.no_grad():
            for t_pos_1, t_pos_2, t_label in test_loader:
                t_label = t_label - 1
                t_pos_1 = t_pos_1.to(args.device)
                t_pos_2 = t_pos_2.to(args.device)
                t_label = t_label.to(args.device)

                out_e, _out2, _out3 = net(t_pos_1, t_pos_2)

                t_pred = out_e.data.max(1, keepdim=True)[1]
                test_correct += t_pred.eq(t_label.data.view_as(t_pred)).cpu().sum()

        test_accuracy = (100. * test_correct / len(test_loader.dataset)).item()
        # print("linear test_accuracy", round(test_accuracy.item(), 4)) 
    else:
        test_accuracy = 0.0

    train_time = time.time() - start_time
    print('Train Epoch: [{}/{}] Loss: {:.4f} TRA: {:.4f} TEA: {:.4f} TIME: {:.4f}'.format(\
                    epoch, args.epochs, round(average_loss, 4), round(train_accuracy, 4), \
                    round(test_accuracy, 4), round(train_time, 2)))
    
    return round(average_loss, 4), round(train_accuracy, 4), round(test_accuracy, 4), round(train_time, 2)


# train for one epoch to learn unique features
def train_SMISO(epoch, net, criterion, train_loader, test_loader, optimizer, args):
    train_correct = 0
    test_correct = 0
    total_loss = 0.0
    # train_bar = tqdm(train_loader)

    start_time = time.time()
    net.train()
    for pos_1, pos_2, label in train_loader:

        label = label - 1
        pos_1  = pos_1.to(args.device)
        pos_2  = pos_2.to(args.device)
        label = label.to(args.device)

        out = net(pos_1, pos_2)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # pred = torch.max(out, 1)[1].squeeze()
        pred = out.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()
        train_accuracy = 100. * train_correct.item() / len(train_loader.dataset)

        total_loss += loss.item()  # 累计 batch 的损失

    average_loss = total_loss / len(train_loader)

    if epoch % args.log_interval == 0:
        net.eval()
        for t_pos_1, t_pos_2, t_label in test_loader:
            with torch.no_grad():
                t_label = t_label - 1
                t_pos_1 = t_pos_1.to(args.device)
                t_pos_2 = t_pos_2.to(args.device)
                t_label = t_label.to(args.device)

                out_e = net(t_pos_1, t_pos_2)

                t_pred = out_e.data.max(1, keepdim=True)[1]
                test_correct += t_pred.eq(t_label.data.view_as(t_pred)).sum()
                test_accuracy = 100. * test_correct / len(test_loader.dataset)
                test_accuracy = round(test_accuracy.item(), 4)
        # print("linear test_accuracy", round(test_accuracy.item(), 4)) 
    else:
        test_accuracy = 0.0

    train_time = time.time() - start_time
    print('Train Epoch: [{}/{}] Loss: {:.4f} TRA: {:.4f} TEA: {:.4f} TIME: {:.4f}'.format(\
                    epoch, args.epochs, round(average_loss, 4), round(train_accuracy, 4), \
                    round(test_accuracy, 4), round(train_time, 2)))
    
    return round(average_loss, 4), round(train_accuracy, 4), test_accuracy, round(train_time, 2)


# train for one epoch to learn unique features
def train_MMISO(epoch, net, criterion, train_loader, test_loader, optimizer, args):
    train_correct = 0
    test_correct = 0
    total_loss = 0.0
    # train_bar = tqdm(train_loader)

    start_time = time.time()
    net.train()
    for batch_idx, (data11, data12, data13, data21, data22, data23, label) in enumerate(train_loader):
        label = label - 1
        data11 = data11.to(args.device)
        data12 = data12.to(args.device)
        data13 = data13.to(args.device)
        data21 = data21.to(args.device)
        data22 = data22.to(args.device)
        data23 = data23.to(args.device)
        label = label.to(args.device)
        # print(data11.shape, data21.shape, data12.shape, data22.shape, data13.shape, data23.shape)
        out = net(data11, data21, data12, data22, data13, data23)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # pred = torch.max(out, 1)[1].squeeze()
        pred = out.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()
        train_accuracy = 100. * train_correct.item() / len(train_loader.dataset)

        total_loss += loss.item()  # 累计 batch 的损失

    average_loss = total_loss / len(train_loader)

    if epoch % args.log_interval == 0:
        net.eval()
        with torch.no_grad():
            for batch_idx, (data11, data12, data13, data21, data22, data23, t_label) in enumerate(test_loader):
                t_label = t_label - 1
                data11 = data11.to(args.device)
                data21 = data21.to(args.device)
                data12 = data12.to(args.device)
                data22 = data22.to(args.device)
                data13 = data13.to(args.device)
                data23 = data23.to(args.device)
                t_label = t_label.to(args.device)
                # print(data11.shape, data21.shape, data12.shape, data22.shape, data13.shape, data23.shape)
                out_e = net(data11, data21, data12, data22, data13, data23)

                t_pred = out_e.data.max(1, keepdim=True)[1]
                test_correct += t_pred.eq(t_label.data.view_as(t_pred)).sum()
                test_accuracy = 100. * test_correct.item() / len(test_loader.dataset)
        # print("linear test_accuracy", round(test_accuracy.item(), 4)) 
    else:
        test_accuracy = 0.0

    train_time = time.time() - start_time
    print('Train Epoch: [{}/{}] Loss: {:.4f} TRA: {:.4f} TEA: {:.4f} TIME: {:.4f}'.format(\
                    epoch, args.epochs, round(loss.cpu().item(), 4), round(train_accuracy, 4), \
                    round(test_accuracy, 4), round(train_time, 2)))
    
    return round(average_loss, 4), round(train_accuracy, 4), round(test_accuracy, 4), round(train_time, 2)



def cal_loss(f_s, f_t, reduction='sum'):
    p_s = F.log_softmax(f_s, dim=1)
    p_t = F.softmax(f_t, dim=1)
    loss = F.kl_div(p_s, p_t, reduction=reduction) / f_t.shape[0]
    return loss


# train for one epoch to learn unique features
def train_MMIMO(epoch, net, criterion, train_loader, test_loader, optimizer, args):
    train_correct = 0
    test_correct = 0
    lambda3 = 0.3
    lambda4 = 1
    lambda1 = 5
    lambda2 = 0.1
    total_loss = 0.0
    # train_bar = tqdm(train_loader)

    start_time = time.time()
    net.train()
    for batch_idx, (data11, data12, data13, data21, data22, data23, label) in enumerate(train_loader):
        label = label - 1
        data11 = data11.to(args.device)
        data12 = data12.to(args.device)
        data13 = data13.to(args.device)
        data21 = data21.to(args.device)
        data22 = data22.to(args.device)
        data23 = data23.to(args.device)
        label = label.to(args.device)
        # print(data11.shape, data21.shape, data12.shape, data22.shape, data13.shape, data23.shape)
        batch_pred, x_cls_cnn,x_cls_trans, \
            x1_out, x2_out, con_loss, x1c_out, \
                loss_ml, x_fuse1, x_fuse2, x_transfusion = net(data11, data21, data12, data22, data13, data23)
        loss = criterion(batch_pred, label)  + lambda1*con_loss

        if args.distillation==1:
            # if datasetname=='Augsburg':
            #     loss += 0.3*criterion(x1_out, batch_target)
            #     loss += 0.3*criterion(x2_out, batch_target)
            #     loss += 0.3*criterion(x1c_out, batch_target)
                
            #     # distillation loss
            #     loss += cal_loss(x1_out,batch_pred)    #蒸馏损失
            #     loss += cal_loss(x2_out,batch_pred)
            #     loss += cal_loss(x1c_out,batch_pred)
            # # cross-entropy loss
            # else:

            # three classification loss
            loss += lambda3 * criterion(x1_out, label)
            loss += lambda3 * criterion(x2_out, label)
            loss += lambda3 * criterion(x1c_out, label)
            # loss += lambda3*criterion(x_cls_cnn, batch_target)
            # loss += lambda3*criterion(x_cls_trans, batch_target)
            
            # distillation loss
            loss += lambda4 * cal_loss(x1_out, batch_pred)    #蒸馏损失
            loss += lambda4 * cal_loss(x2_out, batch_pred)
            loss += lambda4 * cal_loss(x1c_out, batch_pred)
            # loss += lambda4*cal_loss(x_cls_cnn,batch_pred)
            # loss += lambda4*cal_loss(x_cls_trans,batch_pred)

            # mutual information loss
            # if datasetname=='Augsburg':
            #     loss  +=  loss_ml * 0.01 #* adjust(0, 1, epoch, num_epochs)
            # else:
            #     loss  +=  loss_ml * 0.1
            loss  +=  loss_ml * lambda2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # pred = torch.max(out, 1)[1].squeeze()
        pred = batch_pred.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()
        train_accuracy = 100. * train_correct.item() / len(train_loader.dataset)
        total_loss += loss.item()  # 累计 batch 的损失
    average_loss = total_loss / len(train_loader)


    if epoch % args.log_interval == 0:
        net.eval()
        with torch.no_grad():
            for batch_idx, (data11, data12, data13, data21, data22, data23, t_label) in enumerate(test_loader):
                t_label = t_label - 1
                data11 = data11.to(args.device)
                data21 = data21.to(args.device)
                data12 = data12.to(args.device)
                data22 = data22.to(args.device)
                data13 = data13.to(args.device)
                data23 = data23.to(args.device)
                t_label = t_label.to(args.device)
                # print(data11.shape, data21.shape, data12.shape, data22.shape, data13.shape, data23.shape)
                batch_pred, x_cls_cnn, x_cls_trans, x1_out, \
                    x2_out, con_loss, x1c_out, loss_ml, \
                        x_fuse1, x_fuse2, x_transfusion  = net(data11, data21, data12, data22, data13, data23)
                if args.pred_flag == 'o_fuse':
                    choice_pred = batch_pred
                elif args.pred_flag == 'o_1':
                    choice_pred = x1_out
                elif args.pred_flag == 'o_2':
                    choice_pred = x2_out
                elif args.pred_flag == 'o_cnn':
                    choice_pred = x_cls_cnn    
                elif args.pred_flag == 'o_trans':
                    choice_pred = x_cls_trans   
                loss = criterion(choice_pred, t_label)  + con_loss #+ criterion(batch_pred_2, batch_target)
                t_pred = choice_pred.data.max(1, keepdim=True)[1]
                test_correct += t_pred.eq(t_label.data.view_as(t_pred)).sum()
                test_accuracy = 100. * test_correct.item() / len(test_loader.dataset)
        # print("linear test_accuracy", round(test_accuracy.item(), 4)) 
    else:
        test_accuracy = 0.0

    train_time = time.time() - start_time
    print('Train Epoch: [{}/{}] Loss: {:.4f} TRA: {:.4f} TEA: {:.4f} TIME: {:.4f}'.format(\
                    epoch, args.epochs, round(average_loss, 4), round(train_accuracy, 4), \
                    round(test_accuracy, 4), round(train_time, 2)))
    
    return round(average_loss, 4), round(train_accuracy, 4), round(test_accuracy, 4), round(train_time, 2)