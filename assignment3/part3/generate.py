import torch
from transformers import GPT2Tokenizer


def generate(
        model: torch.nn.Module, 
        model_type: str, 
        prompt: str='', 
        num_samples: int=10, 
        n_steps: int=20, 
        do_sample: bool=True, 
        top_k: int = 50, 
        temperature: float = 1.0,
        device: str = 'cpu', 
        verbose: bool = True    
    ):
    """ Generates text samples using a pre-trained GPT model. This function takes a pre-trained model and generates a specified number 
    of text samples based on a given prompt. It allows for customization of the generation process through various parameters like the number 
    of samples, the number of steps (tokens) to generate, sampling strategy, and others.

    Attributes:
        model (torch.nn.Module): The pre-trained GPT model used for text generation.
        model_type (str): The type of GPT model used, necessary for the tokenizer.
        prompt (str, optional): The initial text prompt to start the generation. Defaults to an empty string for unconditional generation.
        num_samples (int, optional): The number of text samples to generate. Defaults to 10.
        n_steps (int, optional): The number of tokens to generate for each sample. Defaults to 20.
        do_sample (bool, optional): Whether to use sampling; set to False for deterministic generation. Defaults to True.
        top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to 50.
        temperature (float, optional): The value used to module the next token probabilities. Defaults to 1.0.
        device (str, optional): The device (e.g., 'cpu' or 'cuda') on which to perform the computation. Defaults to 'cpu'.
        verbose (bool, optional): If True, prints each generated sample. Defaults to True.

    Notes:
        - The function assumes the use of a GPT2Tokenizer from the Hugging Face transformers library.
        - The function is designed to handle both conditional and unconditional text generation based on the provided prompt.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    if prompt == '': 
        # to create unconditional samples...
        # huggingface/transformers tokenizer special cases these strings
        prompt = '<|endoftext|>'
    encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    x = encoded_input['input_ids']
    
    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=n_steps, do_sample=do_sample, top_k=top_k, temperature=temperature)
    
    # Decode the predicted outputs 
    decoded_outputs = []
    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        decoded_outputs.append(out)

        if verbose:
            print('-'*80)
            print(out)


if __name__ == "__main__":
    from gpt import GPT
    
    # Create model
    model_type = 'gpt2'
    model = GPT.from_pretrained(model_type=model_type)
    device = next(model.parameters()).device

    # Hparams
    num_samples=10
    num_generated_tokens=15
    do_sample=True
    temperature = 1.0

    # Generate sentence given initial prompt
    prompt = 'Yesterday I went to the '
    generate(
        prompt=prompt,
        model=model,
        model_type=model_type,
        num_samples=num_samples,
        n_steps=num_generated_tokens,
        do_sample=do_sample,
        temperature=temperature,
        device=device,
    )