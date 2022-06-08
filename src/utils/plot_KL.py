import os
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# grep 'Epoch----->' -r webs-0.7.log  | awk -F' .. ' '{print $3}' >webs-0.7-CL
# grep 'Epoch----->' -r webs-0.7.log  | awk -F' .. ' '{print $4}' >webs-0.7-KL
# grep 'Epoch----->' -r webs-0.7.log  | awk -F' .. ' '{print $5}' >webs-0.7-RC
# grep 'Epoch----->' -r webs-0.7.log  | awk -F' .. ' '{print $6}' >webs-0.7-ELBO

ds='20ng'
# ds='tmn'
# ds='webs'
# ds='reuters'
algs = ['Non-CL(ETM)', 'CNTM']
taus = [0, 0.7]
spot_sets = []
for i, alg in enumerate(algs):
    file = os.path.dirname(__file__) + '/../../KL-logs/' + ds + '-' + str(taus[i]) + '-KL'
    if os.path.exists(file) is False:
        continue

    with open(file, encoding='utf-8') as f:
        alg_spots = DataFrame(columns=['x', 'y'])
        epoch = 0
        for line in f:
            prefix = 'KL_theta:'
            if line.startswith(prefix) is False:
                continue
            v = float(line[len(prefix):].strip())
            alg_spots = alg_spots.append(Series([epoch, v], index=['x', 'y']), ignore_index=True)
            epoch += 1
        spot_sets.append(alg_spots)

fig = plt.figure()
subplot = fig.add_subplot(1, 1, 1)

if ds == '20ng' or ds == 'reuters':
    subplot.set_xlim(0, 999)
    subplot.set_ylim(0, 18)
elif ds == 'tmn':
    subplot.set_xlim(0, 999)
    subplot.set_ylim(0, 6)
elif ds == 'webs':
    subplot.set_xlim(0, 999)
    subplot.set_ylim(0, 3.5)
# subplot.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.xlabel("Training Epoch (" + ds.upper() + ")")
plt.ylabel("KL theta")

for i, alg_spots in enumerate(spot_sets):
    label = algs[i]
    marker = "."
    style = '-'
    if i == 0:
        color = 'green'
    elif i == 1:
        color = 'red'
    elif i == 2:
        color = 'yellow'
    elif i == 3:
        color = 'black'
    elif i == 4:
        color = 'magenta'
    elif i == 5:
        color = 'purple'
    elif i == 6:
        color = 'blue'
    else:
        continue

    subplot.plot(alg_spots.x, alg_spots.y, marker=marker, color=color, linestyle=style, label=label)

    subplot.legend(loc=7)

plt.legend()
plt.show()
exit(0)