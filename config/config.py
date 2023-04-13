import json

# Load config.json
with open('./config/config.json') as f:
    _sweep_config = json.load(f)

def sweep_config(args):
    # please refer below
    # https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
    s_config = _sweep_config
    
    if not 'project' in s_config:
        s_config['project'] = args.project

    if not 'entity' in s_config:
        s_config['entity'] = args.entity
        
    return s_config