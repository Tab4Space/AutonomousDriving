import matplotlib.pyplot as plt
import numpy as np

sub001_csv = np.loadtxt('./_data/sub_001/action.csv', dtype=np.float32, encoding="utf-8", delimiter=',')
sub001_len = 5000
sub001_steer = sub001_csv[:5000, 0]

#plt.plot(sub001_steer)
#plt.show()

sub002_csv = np.loadtxt('./_data/sub_002/action.csv', dtype=np.float32, encoding="utf-8", delimiter=',')
sub002_len = 5000
sub002_steer = sub002_csv[:5000, 0]

#plt.plot(sub002_steer)
#plt.show()

plt.plot(range(sub001_len), sub001_steer, color='#FF8000', linestyle='solid', label='Expert')
plt.plot(range(sub002_len), sub002_steer, color='#0080FE', linestyle='dashed', label='Model')
plt.legend(loc='upper right')
plt.xlabel('Step')
plt.ylabel('Steering Value')
# plt.legend(sub002_steer)
plt.show()