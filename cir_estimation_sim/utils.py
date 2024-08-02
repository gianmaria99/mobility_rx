import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tikzplotlib as tik

def load_data(path, fc, relative=True, metric='fd'):
    if relative:
        try:
            err = np.load(path + metric + '_rel_fc%s.npy'%(fc))
        except:
            print('There are no relative %s errors for fc=%s'%(metric, fc))
    else:
        try:
            err = np.load(path + metric + '_abs_fc%s.npy'%(fc))
        except:
            print('There are no absolute %s errors for fc=%s'%(metric, fc))

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

def plot_boxplot(path, errors, xlabel, ylabel, xticks, title, save, name=''):
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
    if save:
        plt.savefig(path+name+'.png')
        tik.save(path+name+'.tex')
    plt.show()
    plt.close()

def argcommon_el(a, b):
    result = [ind for ind,i in enumerate(a) if i in b]
    return result

def plot(path, fc, relative=True, metric='fd', snr=[20], aoa=[5], interval=[48], n_static=[2], save=False):
    err, info = load_data(path, fc, relative, metric)
    for s in snr:
        if not any(np.array([int(j) for j in info['snr']])==s):
            print('selected SNR value %s dB is not available, the available SNR values are: %s'%(s, info['snr']))
    for a in aoa:
        if not any(np.array([int(j) for j in info['aoa']])==a):
            print('selected AoA std value %s° is not available, the available AoA std are: %s'%(a, info['aoa']))
    for i in interval:
         if not any(np.array([int(j) for j in info['interval']])==i):
            print('selected interval value %s ms is not available, the available interval values are: %s'%(i, info['interval']))
    for n in n_static:
        if not any(np.array([int(j) for j in info['n_static']])==n):
            print('selected number of static paths %s is not available, the numbers of available static paths are: %s'%(n, info['n_static']))
    s = argcommon_el([int(j) for j in info['snr']], snr)
    a = argcommon_el([int(j) for j in info['aoa']], aoa)
    i = argcommon_el([int(j) for j in info['interval']], interval)
    n = argcommon_el([int(j) for j in info['n_static']], n_static)
    err = err[:,s,a,i,n]
    var_ind = []
    for j,el in enumerate([s,a,i,n]):
        if len(el)>1:
            var_ind.append(j)
        
    if len(var_ind)>1:
        print('Please vary one parameter at a time')
        return
    var_ind = var_ind[0]
    if relative:
        y_label = 'normalized ' + metric + ' error'
    else:
        y_label = 'absolute ' + metric + ' error'
    if var_ind==0:
        x_label = 'SNR [dB]'
        x_ticks = np.array([int(j) for j in info['snr']])[s]
        title = 'Varying SNR'
        name = 'var_snr_fc%s'%(fc)
    if var_ind==1:
        x_label = 'AoA std [°]'
        x_ticks = np.array([int(j) for j in info['aoa']])[a]
        title = 'Varying AoA'
        name = 'var_aoa_fc%s'%(fc)
    if var_ind==2:
        x_label = 'Aggregation window KT [ms]'
        x_ticks = np.array([int(j) for j in info['interval']])[i]
        title = 'Varying interval'
        name = 'var_interval_fc%s'%(fc)
    if var_ind==3:
        x_label = 'No. static paths S'
        x_ticks = np.array([int(j) for j in info['n_static']])[n]
        title = 'Varying number of static paths'
        name = 'var_nstatic_fc%s'%(fc)
        
    plot_boxplot(path, err, x_label, y_label, x_ticks, title, save, name)

if __name__=='__main__':        
    print_error('cir_estimation_sim/data/test/', 60)
    plot('cir_estimation_sim/data/test/', 60, snr=[20], aoa=[5], interval=[2,48], n_static=[2])
