from . import net_stf7i

def create_net(train_source, test_source, **kwargs): 
    return net_stf7i.create_net(train_source, test_source, filter0_width=64, **kwargs)


