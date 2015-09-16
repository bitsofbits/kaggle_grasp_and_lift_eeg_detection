from . import net_stf7

def create_net(train_source, test_source, **kwargs): 
    return net_stf7.create_net(train_source, test_source, filter0_width=9, filter1_num=64, filter2_num=128, **kwargs)
