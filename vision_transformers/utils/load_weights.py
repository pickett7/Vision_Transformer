import torch

def load_pretrained_state_dict(model, model_name='vit_ti_p16_224'):

    # model_vit_tiny = torch.load('trained_models/vit_pretrained/model/vit_tiny_patch16_224.pth')
    # print(model_vit_tiny)
    # NOTE: This loads timm weights.
    weights = torch.load('trained_models/vit_pretrained/model/vit_tiny_patch16_224.pth')
    # Model's current state dictionary.
    state_dict = model.state_dict()

    if model_name == 'vit_ti_p16_224' or model_name == 'vit_ti_p16_384':
        print("Loading timm weight")
        state_dict['cls_token'] = weights['cls_token']
        state_dict['pos_embedding'] = weights['pos_embed']
        state_dict['patches.patch.weight'] = weights['patch_embed.proj.weight']
        state_dict['patches.patch.bias'] = weights['patch_embed.proj.bias']
        
        for i in range(12):
            state_dict[f"transformer.layers.{i}.0.norm.weight"] = weights[f"blocks.{i}.norm1.weight"]
            state_dict[f"transformer.layers.{i}.0.norm.bias"] = weights[f"blocks.{i}.norm1.bias"]
            state_dict[f"transformer.layers.{i}.0.fn.qkv.weight"] = weights[f"blocks.{i}.attn.qkv.weight"]
            state_dict[f"transformer.layers.{i}.0.fn.qkv.bias"] = weights[f"blocks.{i}.attn.qkv.bias"]
            state_dict[f"transformer.layers.{i}.0.fn.out.0.weight"] = weights[f"blocks.{i}.attn.proj.weight"]
            state_dict[f"transformer.layers.{i}.0.fn.out.0.bias"] = weights[f"blocks.{i}.attn.proj.bias"]
            state_dict[f"transformer.layers.{i}.1.norm.weight"] = weights[f"blocks.{i}.norm2.weight"]
            state_dict[f"transformer.layers.{i}.1.norm.bias"] = weights[f"blocks.{i}.norm2.bias"]
            state_dict[f"transformer.layers.{i}.1.fn.mlp_net.0.weight"] = weights[f"blocks.{i}.mlp.fc1.weight"]
            state_dict[f"transformer.layers.{i}.1.fn.mlp_net.0.bias"] = weights[f"blocks.{i}.mlp.fc1.bias"]
            state_dict[f"transformer.layers.{i}.1.fn.mlp_net.3.weight"] = weights[f"blocks.{i}.mlp.fc2.weight"]
            state_dict[f"transformer.layers.{i}.1.fn.mlp_net.3.bias"] = weights[f"blocks.{i}.mlp.fc2.bias"]
            
        state_dict['ln.weight'] = weights['norm.weight']
        state_dict['ln.bias'] = weights['norm.bias']
        # MAYBE no need to load head weights.
        state_dict['mlp_head.weight'] = weights['head.weight']
        state_dict['mlp_head.bias'] = weights['head.bias']
        model.load_state_dict(state_dict)
        return model




