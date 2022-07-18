import numpy as np
import time

def print_losses(total_epoch, epoch, iter, iter_per_epoch, losses, print_keys=False):

    if print_keys:
        header_str = 'epoch [%d/%d]\t\t\tloss' % (epoch,total_epoch)

        for key, value in losses.items():
            if key != 'loss':
                if len(key) < 5:
                    key_str = key + ' ' * (5 - len(key))
                    header_str += '\t\t%s' % (key_str)
                else:
                    header_str += '\t\t%s' % (key[0:5])

        now = time.localtime()
        d_h_s = "%02d:%02d:%02d" % (now.tm_hour,now.tm_min,now.tm_sec)
        print(header_str,'\t\t',d_h_s)

    loss_str = 'epoch [%d/%d] %05d/%05d: \t%.4f' % (epoch,total_epoch, iter, iter_per_epoch, np.mean(losses['loss']))

    for key, value in losses.items():
        if key != 'loss':
            loss_str += '\t\t%.4f' % (np.mean(value))

    print(loss_str)
