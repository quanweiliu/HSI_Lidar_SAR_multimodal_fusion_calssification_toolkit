import os
import torch

def save_weights(train_loss, test_loss, best_loss, best_acc, test_accuracy, epoch, model, super_head, optimizer, args):
    # if train_loss < best_loss:
    #     best_loss = train_loss
    #     if super_head != None:
    #         torch.save({
    #                 "epoch": epoch,
    #                 "model": model.state_dict(),
    #                 "super_head": super_head.state_dict(),
    #                 "optimizer": optimizer.state_dict()}, 
    #                 os.path.join(args.result_dir, "train_loss.pth"))
    #     else:
    #         torch.save({
    #                 "epoch": epoch,
    #                 "model": model.state_dict(),
    #                 "optimizer": optimizer.state_dict()}, 
    #                 os.path.join(args.result_dir, "train_loss.pth"))


    if test_loss < best_loss:
        best_loss = test_loss
        if super_head != None:
            torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "super_head": super_head.state_dict(),
                    "optimizer": optimizer.state_dict()}, 
                    os.path.join(args.result_dir, "test_loss.pth"))
        else:
            torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, 
                    os.path.join(args.result_dir, "test_loss.pth"))


    # if best_acc < test_accuracy:
    #     best_acc = test_accuracy
    #     if super_head != None:
    #         torch.save({
    #                 "epoch": epoch,
    #                 "model": model.state_dict(),
    #                 "super_head": super_head.state_dict(),
    #                 "optimizer": optimizer.state_dict()}, 
    #                 os.path.join(args.result_dir, "test_acc.pth"))
    #     else:
    #         torch.save({
    #                 "epoch": epoch,
    #                 "model": model.state_dict(),
    #                 "optimizer": optimizer.state_dict()}, 
    #                 os.path.join(args.result_dir, "test_acc.pth"))
            

    # if train_loss < best_loss and best_acc <= test_accuracy:
    #     best_loss = train_loss
    #     best_acc = test_accuracy
    #     if super_head != None:
    #         torch.save({
    #                 "epoch": epoch,
    #                 "model": model.state_dict(),
    #                 "super_head": super_head.state_dict(),
    #                 "optimizer": optimizer.state_dict()}, 
    #                 os.path.join(args.result_dir, "model.pth"))
    #     else:
    #         torch.save({
    #                 "epoch": epoch,
    #                 "model": model.state_dict(),
    #                 "optimizer": optimizer.state_dict()}, 
    #                 os.path.join(args.result_dir, "model.pth"))
            
    return best_loss, best_acc