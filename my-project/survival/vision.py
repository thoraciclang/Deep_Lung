import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def plot_train_curve(L, title='Training Curve'):
	x = range(1, len(L) + 1)
	plt.plot(x, L)
	# no ticks
	plt.xticks([])
	plt.title(title)
	plt.show()

def plot_surv_func(T, surRates, id):
	plt.plot(T, np.transpose(surRates))
	plt.show()
	plt.savefig('./fig_LCCS/surv_'+str(id)+'.png')
	plt.close()