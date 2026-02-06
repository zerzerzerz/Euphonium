# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import torch
from torch.utils.data import Dataset
import json
import os
import random


class PromptEmbeddingDataset(Dataset):
    def __init__(
        self, json_path, num_latent_t, cfg_rate,
    ):
        # data_merge_path: video_dir, latent_dir, prompt_embed_dir, json_path
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.datase_dir_path = os.path.dirname(json_path)
        #self.video_dir = os.path.join(self.datase_dir_path, "video")
        #self.latent_dir = os.path.join(self.datase_dir_path, "latent")
        self.prompt_embed_dir = os.path.join(self.datase_dir_path, "prompt_embed")
        self.prompt_attention_mask_dir = os.path.join(
            self.datase_dir_path, "prompt_attention_mask"
        )
        with open(self.json_path, "r") as f:
            self.data_anno = json.load(f)
        # json.load(f) already keeps the order
        # self.data_anno = sorted(self.data_anno, key=lambda x: x['latent_path'])
        self.num_latent_t = num_latent_t
        # just zero embeddings [256, 4096]
        self.uncond_prompt_embed = torch.zeros(256, 4096).to(torch.float32)
        # 256 zeros
        self.uncond_prompt_mask = torch.zeros(256).bool()
        self.lengths = [
            data_item["length"] if "length" in data_item else 1
            for data_item in self.data_anno
        ]

    def __getitem__(self, idx):
        #latent_file = self.data_anno[idx]["latent_path"]
        prompt_embed_file = self.data_anno[idx]["prompt_embed_path"]
        prompt_attention_mask_file = self.data_anno[idx]["prompt_attention_mask"]
        if random.random() < self.cfg_rate:
            prompt_embed = self.uncond_prompt_embed
            prompt_attention_mask = self.uncond_prompt_mask
        else:
            # Support both absolute and relative paths for backward compatibility
            if os.path.isabs(prompt_embed_file):
                prompt_embed_path = prompt_embed_file
            else:
                prompt_embed_path = os.path.join(self.prompt_embed_dir, prompt_embed_file)
            
            if os.path.isabs(prompt_attention_mask_file):
                prompt_attention_mask_path = prompt_attention_mask_file
            else:
                prompt_attention_mask_path = os.path.join(
                    self.prompt_attention_mask_dir, prompt_attention_mask_file
                )
            
            prompt_embed = torch.load(
                prompt_embed_path,
                map_location="cpu",
                weights_only=True,
            )
            prompt_attention_mask = torch.load(
                prompt_attention_mask_path,
                map_location="cpu",
                weights_only=True,
            )
        return prompt_embed, prompt_attention_mask, self.data_anno[idx]['caption']

    def __len__(self):
        return len(self.data_anno)


def prompt_embedding_collate_function(batch):
    # return latent, prompt, latent_attn_mask, text_attn_mask
    # latent_attn_mask: # b t h w
    # text_attn_mask: b 1 l
    # needs to check if the latent/prompt' size and apply padding & attn mask
    prompt_embeds, prompt_attention_masks, caption = zip(*batch)
    
    # Check if all sequences have the same length
    seq_lengths = [emb.shape[0] for emb in prompt_embeds]
    if len(set(seq_lengths)) > 1:
        # Different lengths detected, need to pad
        max_seq_len = max(seq_lengths)
        hidden_dim = prompt_embeds[0].shape[1]
        
        padded_prompt_embeds = []
        padded_prompt_attention_masks = []
        
        for emb, mask in zip(prompt_embeds, prompt_attention_masks):
            seq_len = emb.shape[0]
            if seq_len < max_seq_len:
                # Pad embeddings with zeros
                padding = torch.zeros(max_seq_len - seq_len, hidden_dim, dtype=emb.dtype)
                padded_emb = torch.cat([emb, padding], dim=0)
                
                # Pad attention mask with False (or 0)
                mask_padding = torch.zeros(max_seq_len - seq_len, dtype=mask.dtype)
                padded_mask = torch.cat([mask, mask_padding], dim=0)
            else:
                padded_emb = emb
                padded_mask = mask
            
            padded_prompt_embeds.append(padded_emb)
            padded_prompt_attention_masks.append(padded_mask)
        
        prompt_embeds = torch.stack(padded_prompt_embeds, dim=0)
        prompt_attention_masks = torch.stack(padded_prompt_attention_masks, dim=0)
    else:
        # All same length, can directly stack
        prompt_embeds = torch.stack(prompt_embeds, dim=0)
        prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)
    
    return prompt_embeds, prompt_attention_masks, caption


if __name__ == "__main__":
    dataset = PromptEmbeddingDataset("data/rl_embeddings/videos2caption.json", num_latent_t=28, cfg_rate=0.0)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=prompt_embedding_collate_function
    )
    for prompt_embed, prompt_attention_mask, caption in dataloader:
        print(
            prompt_embed.shape,
            prompt_attention_mask.shape,
            caption
        )
        import pdb

        pdb.set_trace()
