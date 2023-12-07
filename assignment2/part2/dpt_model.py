"""Defines the VisualPrompting model (based on CLIP)"""
from pprint import pprint
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.nn.functional import normalize


import warnings


def load_clip_to_cpu(cfg):
    """Loads CLIP model to CPU."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class DeepPromptCLIP(nn.Module):
    """Modified CLIP module to support prompting."""
    def __init__(self, args, dataset, template="This is a photo of {}"):
        super(DeepPromptCLIP, self).__init__()
        classnames = dataset.classes

        print(f"Loading CLIP (backbone: {args.arch})")
        clip_model = self.load_clip_to_cpu(args)
        clip_model.to(args.device)

        # hack to make model as float() (This is a CLIP hack)
        if args.device == "cpu":
            clip_model = clip_model.float()


        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        print("List of prompts:")
        pprint(prompts)

        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(args.device)


        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Hint: You can use the code from clipzs.py here!

        # Instructions:
        # - Given a list of prompts, compute the text features for each prompt.
        # - Return a tensor of shape (num_prompts, 512).

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
        text_features = normalize(text_features, dim=-1)
        #######################
        # END OF YOUR CODE    #
        #######################

        self.text_features = text_features
        self.clip_model = clip_model
        self.logit_scale = self.clip_model.logit_scale.exp().detach()

        self.injection_layer = args.injection_layer

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Hint: consider the shape required for the deep prompt to be compatible with the CLIP model 
        # Hint: CLIP uses different datatypes for CPU (float32) and GPU (float16)
        # Hint: use args.prompt_num to specify the number of deep prompts to use

        # args.prompt_num defines number of tokens to be added to the input tokens
        # 1 is just batch size of 1.
        # embd_size is the CLIP transformer block input size
        # dim=[args.prompt_num, 1, embd_size]
        embd_size = 768
        self.deep_prompt = nn.Parameter(torch.randn(args.prompt_num, 1, embd_size)) # force it have same type as the data.
        #######################
        # END OF YOUR CODE    #
        #######################


    def forward(self, image):
        """Forward pass of the model."""
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Implement the forward function. This is not exactly the same as
        # the model_inferece function in clipzs.py! Please see the steps below.

        # Steps:
        # - Compute the image features using the CLIP model (be sure use the custom_encode_image function).
        # - Normalize the image features.
        # - Compute similarity logits between the image features and the text features.
        # - You need to multiply the similarity logits with the logit scale (clip_model.logit_scale).
        # - Return logits of shape (batch size, number of classes).
        image_features = self.custom_encode_image(image)
        # Note: Ensure that the feature vectors are normalized (to unit length) before computing the similarity. 
        # This is crucial for the dot product to accurately represent cosine similarity.
        image_features = normalize(image_features, dim=-1)
        # print("image_features shape: ", image_features.shape)
        # print("text_features shape: ", self.text_features.shape)
        logits_per_image = self.logit_scale * image_features @ self.text_features.t()
        # print("logits_per_image shape: ", logits_per_image.shape)
        return logits_per_image
        #######################
        # END OF YOUR CODE    #
        #######################

    def custom_encode_image(self, x):
        """Encode image using CLIP model and add deep prompts."""
        # cf. https://github.com/openai/CLIP/blob/main/clip/model.py#L223

        x = x.type(self.clip_model.dtype)
        image_encoder = self.clip_model.visual
        
        x = image_encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + image_encoder.positional_embedding.to(x.dtype)
        x = image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Implement the part of the code where the deep prompt is injected into the CLIP model.
        # The custom_encode_image function largely follows the code from the CLIP repository.
        # You only need to modify the code responsible for running the transformer blocks.

        # Steps:
        # - Iterate over the transformer blocks (image_encoder.transformer.resblocks).
        # - Inject the deep prompt at the specified layer (self.injection_layer).

        # Hint: Beware of the batch size (the deep prompt is the same for all images in the batch).
        for i, block in enumerate(image_encoder.transformer.resblocks):
            if i == self.injection_layer:
                # x batch dim = [sequence_size, batch_size, emedding_size]
                # deep prompt has the dim = [args.prompt_num, 1, emded_size]
                # expand the deep prompt to have the dims [(args.prompt_num, batch_size, emded_size]
                # to match the dim of the batch x
                prompt = self.deep_prompt.to(x.dtype).expand(-1, x.size(1), -1)
                # append the prompt to x so that we will have the extra prompt tokens. 
                # final x batch dim = [sequence_size + args.prompt_num, batch_size, emded_size]
                x = torch.cat((x, prompt), dim=0)
            x = block(x)
        #######################
        # END OF YOUR CODE    #
        #######################

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = image_encoder.ln_post(x[:, 0, :])

        if image_encoder.proj is not None:
            x = x @ image_encoder.proj

        return x

    def load_clip_to_cpu(self, args):
        """Loads CLIP model to CPU."""
        backbone_name = args.arch
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url, args.root)
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = clip.build_model(state_dict or model.state_dict())
        return model

    @torch.no_grad()
    def visualize_prompt(self, method):
        """Visualizes the prompt."""
        warnings.warn("Deep prompts are not supported for visualization.")
