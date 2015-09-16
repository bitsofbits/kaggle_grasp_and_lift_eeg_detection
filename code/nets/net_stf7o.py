from . import net_stf7m

def create_net(train_source, test_source, **kwargs): 
    return net_stf7m.create_net(train_source, test_source, filter0_width=5, **kwargs)


