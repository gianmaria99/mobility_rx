import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tikzplotlib as tik

def load_data(path, fc, relative=True, metric='fd'):
    if relative:
        err = np.load(path + metric + '_rel_fc%s.npy'%(fc))
    else:
        err = np.load(path + metric + '_abs_fc%s.npy'%(fc))
    f = open(path + 'info.txt', 'r')
    info = {}
    info['n_static'] = f.readline().split('[')[1].split(']')[0].split(', ')
    info['interval'] = f.readline().split('[')[1].split(']')[0].split(', ')
    info['aoa'] = f.readline().split('[')[1].split(']')[0].split(', ')
    info['snr'] = f.readline().split('[')[1].split(']')[0].split(', ')
    return err, info

def print_error(path, fc, relative=True, metric='fd'):
    err, info = load_data(path, fc, relative, metric)
    for n in range(len(info['n_static'])):
        for i in range(len(info['interval'])):
            for a in range(len(info['aoa'])):
                for s in range(len(info['snr'])):
                    print('With parameters: n_static=%s, interval=%s, aoa=%s, snr=%s '%(info['n_static'][n], info['interval'][i], info['aoa'][a], info['snr'][s]))
                    print(metric + ' median error: ' + str(np.median(err[:,s,a,i,n], axis=0)))
                    print(metric + ' average error: ' + str(np.mean(err[:,s,a,i,n], axis=0)))
                    print(metric + ' error std: ' + str(np.std(err[:,s,a,i,n], axis=0)) + '\n')

def plot_boxplot(path, errors, xlabel, ylabel, xticks, title, name=''):
    """
        Plot boxplots using seaborn for the given list of errors.
    """
    plt.figure(figsize=(12,8))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid()
    ax = sns.boxplot(data=errors, orient='v', palette='rocket', showfliers=False)
    plt.xticks(np.arange(len(xticks)), xticks)
    plt.title(title)
    #plt.ylim(top=60,bottom=0)
    plt.savefig(path+name+'.png')
    tik.save(path+name+'.tex')
    plt.show()
    plt.close()

def plot_fd(path, fc, relative=True, metric='fd', snr=20, aoa=5, interval=48, n_static=2):
    err, info = load_data(path, fc, relative, metric)
    if not any(info['snr'].astype(np.float)==snr):
        print('selected SNR value is not available, the available SNR values are: ' + info['snr'])
    if not any(info['aoa'].astype(np.float)==aoa):
        print('selected AoA std value is not available, the available AoA std are: ' + info['aoa'])
    if not any(info['interval'].astype(np.float)==interval):
        print('selected interval value is not available, the available interval values are: ' + info['interval'])
    if not any(info['n_static'].astype(np.float)==n_static):
        print('selected number of static paths is not available, the numbers of available static paths are: ' + info['n_static'])

print_error('cir_estimation_sim/data/test/', 60)