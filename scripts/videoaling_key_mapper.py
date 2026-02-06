import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description='Map keys in a PyTorch checkpoint and save to a new file.')
    parser.add_argument('--input', default='VideoReward/checkpoint-11352/model.pth', help='Path to the input PTH file')
    parser.add_argument('--output', default='VideoReward-update_key/checkpoint-11352/model.pth', help='Path to the output PTH file')
    args = parser.parse_args()

    # Load the checkpoint
    checkpoint = torch.load(args.input, map_location='cpu')
    
    # Extract state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Apply key mapping based on the rules from tmp.py
    new_state_dict = {}
    for key in state_dict:
        if key.startswith('base_model.model.visual.'):
            new_key = key.replace('base_model.model.visual.', 'base_model.model.model.visual.', 1)
        elif key.startswith('base_model.model.model.'):
            new_key = key.replace('base_model.model.model.', 'base_model.model.model.language_model.', 1)
        else:
            new_key = key
        new_state_dict[new_key] = state_dict[key]

    # Save the new state_dict
    torch.save(new_state_dict, args.output)
    print(f'Mapped checkpoint saved to {args.output}')

if __name__ == '__main__':
    main()