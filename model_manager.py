import transformers
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
import re



class GPTJ():
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.model = self.load_and_configure()


    def load_and_configure(self):
        config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
        model = GPTJForCausalLM(config)
        self.add_adapters(model)
        print("adapters successfully added")

        # Delete unnecessary variables to conserve memory
        del config


        # Load the model's state dictionary from the checkpoint
        print("starting to load state dict")
        state_dict = torch.load(self.model_path, map_location=self.device)
        print("state dict loaded")
        model.load_state_dict(state_dict)
        print("state dict loaded to model")
        del state_dict

        model.to(self.device)

        # Clear GPU memory cache
        torch.cuda.empty_cache()
        return model

    def add_adapters(self, model):
      adapter_dim=16
      # assert adapter_dim > 0

      for module in model.modules():
          if isinstance(module, FrozenBNBLinear):
              module.adapter = nn.Sequential(
                  nn.Linear(module.in_features, adapter_dim, bias=False),
                  nn.Linear(adapter_dim, module.out_features, bias=False),
              )
              nn.init.zeros_(module.adapter[1].weight)
          elif isinstance(module, FrozenBNBEmbedding):
              module.adapter = nn.Sequential(
                  nn.Embedding(module.num_embeddings, adapter_dim),
                  nn.Linear(adapter_dim, module.embedding_dim, bias=False),
              )
              nn.init.zeros_(module.adapter[1].weight)


    def generate_verse(self, prompt):
        print("...getting prediction...")
        with torch.no_grad():
            result_length = 75
            prompt = "~ " + prompt + " =1G->2G= "
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            beam_outputs = self.model.generate(inputs["input_ids"],
                max_length=result_length,
                top_k=50, top_p=0.95,
                do_sample=True, temperature=0.7, pad_token_id=50256,
                num_return_sequences=10)

            lines = []
            for beam in beam_outputs:
                text = self.tokenizer.decode(beam, skip_special_tokens=True)
                line = text.split(" =1G->2G= ")[1]
                line = line[:line.find(" ~")]
                if line not in lines:
                    lines.append(line.strip("'").strip('"'))

            filtered_lines = [self.replace_n_word(line) for line in lines] #remove n-word
            filtered_lines = [line for line in filtered_lines if self.has_characters(line)]

            if len(filtered_lines) == 0:
                return self.get_prediction(prompt)  # Recursively retry if no suitable lines found

            # Delete unnecessary variables to conserve memory
            del inputs, beam_outputs, lines

            return filtered_lines

    def replace_n_word(self, line):
        # Define a regular expression pattern to match n-word and hyphenated cases
        pattern = r'\b\w*-?nigga\w*\b'
        # Use re.search to find the pattern in the line
        line = re.sub(pattern,"*****", line, flags=re.IGNORECASE)
        # returns line with stars if present or w no modifications if word not found
        return line
    
    def has_characters(self,line):
    # Strip removes leading and trailing whitespaces including tabs and newlines
        return False if line.strip() == "" else True


class FrozenBNBLinear(nn.Module):
    def __init__(self, weight, absmax, code, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
        self.bias = bias

    def forward(self, input):
        output = torch.clone(DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias))
        if self.adapter:
            output += self.adapter(input)
        return output

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


class DequantizeAndLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor,
                absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        assert not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2] and not ctx.needs_input_grad[3]
        input, weights_quantized, absmax, code = ctx.saved_tensors
        # grad_output: [*batch, out_features]
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias


class FrozenBNBEmbedding(nn.Module):
    def __init__(self, weight, absmax, code):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None

    def forward(self, input, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable
            weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
            output = F.embedding(input, weight_deq, **kwargs)
        if self.adapter:
            output += self.adapter(input)
        return output

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "FrozenBNBEmbedding":
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"


def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20):
    assert chunk_size % 4096 == 0
    code = None
    chunks = []
    absmaxes = []
    flat_tensor = matrix.view(-1)
    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size: (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code=code)
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)

    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    absmax = torch.cat(absmaxes)
    return matrix_i8, (absmax, code)


def convert_to_int8(model):
    """Convert linear and embedding modules to 8-bit with optional adapters"""
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print(name, child)
                setattr(
                    module,
                    name,
                    FrozenBNBLinear(
                        weight=torch.zeros(child.out_features, child.in_features, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    FrozenBNBEmbedding(
                        weight=torch.zeros(child.num_embeddings, child.embedding_dim, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                    )
                )

class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self.attn)
        convert_to_int8(self.mlp)


class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)


class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)

transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J





