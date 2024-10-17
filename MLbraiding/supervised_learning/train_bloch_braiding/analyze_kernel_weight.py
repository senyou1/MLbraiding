from matplotlib import pyplot as plt

from mode import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L=201
model = Tudui(L).to(device)
model.load_state_dict(torch.load('../train_bloch_braiding/checkpoint.pt'))
model = model.to(device)
print(model)
weight_keys=model.state_dict().keys()
print(weight_keys)
for key in weight_keys:
    if "num_batches_tracked" in key:
        continue
    weight_t=model.state_dict()[key].cpu().numpy()
    print(weight_t.shape)
    # read a kernel information
    # weight_t=weight_t[39,:,:,:]
    weight_mean=weight_t.mean()
    weight_std = weight_t.std(ddof=0)
    weight_min = weight_t.min()
    weight_max = weight_t.max()
    print("mean is {:.2f},std is {:.2f},max is {:.2f},min is {:.2f}".format(weight_mean,
                                                            weight_std,
                                                            weight_max,
                                                            weight_min))

    plt.close()
    weight_vec=np.reshape(weight_t,[-1])
    plt.hist(weight_vec,bins=50)
    plt.title(key)
    plt.show()