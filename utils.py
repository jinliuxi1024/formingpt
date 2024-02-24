def _adapt_key(key: str) -> str:
    key_map = {
        'wte': 'token_embedding',
        'wpe': 'position_embedding',
        'h': 'blocks',
        'attn': 'attention',
        'ln_f': 'layer_norm',
        'ln_1': 'input_layer_norm',
        'ln_2': 'intermediate_layer_norm',
        'c_attn': 'input_projection',
        'c_fc': 'input_projection',
        'c_proj': 'output_projection',
        'mlp': 'feedforward',
        'lm_head': 'language_model_head',
    }
    items = key.split('.')
    new_items = []
    for item in items:
        if item in key_map:
            new_items.append(key_map[item])
        else:
            new_items.append(item)
    return '.'.join(new_items)

def adapt_huggingface_transformers_state_dict(state_dict: dict):
    new_state_dict = {}
    transpose_names = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    remove_names = ['.attn.bias', '.attn.masked_bias']
    for key, value in state_dict.items():
        if any(key.endswith(name) for name in remove_names):
            continue
        # 如果key以remove_names中的任意一个结尾，就跳过这个key
        new_key = _adapt_key(key)
        if any(key.endswith(name) for name in transpose_names):
            new_state_dict[new_key] = value.t()
            # 如果key以transpose_names中的任意一个结尾，就将value转置后赋值给new_state_dict
        else:
            new_state_dict[new_key] = value
            # 否则直接赋值给new_state_dict
    return new_state_dict

# 从总体上看，这两个函数的作用是将state_dict的key进行一些修改，使得state_dict的key和minigpt的key对应





