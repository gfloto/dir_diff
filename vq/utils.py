import yaml
from taming.models.vqgan import VQModel

# return model given yaml file
def get_model(yaml_path):
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ddconfig = config['model']['params']['ddconfig']
    lossconfig = config['model']['params']['lossconfig']
    n_embed = config['model']['params']['n_embed']
    embed_dim = config['model']['params']['embed_dim']
    disc_start = config['model']['params']['lossconfig']['params']['disc_start']
    print(f'embed dim: {embed_dim}\t num embed: {n_embed}')

    return VQModel(ddconfig, lossconfig, n_embed, embed_dim)