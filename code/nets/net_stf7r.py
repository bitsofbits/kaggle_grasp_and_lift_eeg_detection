from . import net_stf7m

def create_net(train_source, test_source, **kwargs): 
    return net_stf7m.create_net(train_source, test_source, filter1_width=4, filter2_width=4, **kwargs)


