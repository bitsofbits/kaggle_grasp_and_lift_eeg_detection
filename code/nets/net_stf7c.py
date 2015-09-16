from . import net_stf7

def create_net(train_source, test_source, **kwargs): 
    return net_stf7.create_net(train_source, test_source, filter0_width=21, filter1_num=128, filter2_num=256, **kwargs)


