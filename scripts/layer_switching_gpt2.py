import warnings

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, init
from torch.nn.parameter import Parameter
from transformers import PretrainedConfig, Conv1D
from transformers.modeling_gpt2 import Attention, MLP, GPT2PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


class LayerSwitchingGPT2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.GPT2Model`.
    It is used to instantiate an GPT-2 model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the GPT-2 `small <https://huggingface.co/gpt2>`__ architecture.

    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.


    Args:
        vocab_size (:obj:`int`, optional, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the different tokens that
            can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.GPT2Model`.
        type_vocab_size (:obj:`int`, optional, defaults to None):
            The vocabulary size of the `token_type_ids` passed into :class:`~transformers.GPT2Model`.
            :obj:`None` will disable token_type embeddings.
        n_positions (:obj:`int`, optional, defaults to 1024):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        n_ctx (:obj:`int`, optional, defaults to 1024):
            Dimensionality of the causal mask (usually same as n_positions).
        n_embd (:obj:`int`, optional, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_language_specific_attention_layers (:obj:`int`, optional, defaults to 0):
            Number of attention layers (from the bottom up) to be language specific
        language_specific_input_embeds (:obj:`bool`, optional, defaults to False):
            Whether or not to use separate input embeddings for each language
        language_specific_prediction_heads (:obj:`bool`, optional, defaults to False):
            Whether or not to use separate prediction heads for each language
        semantic_concepts (:obj:`int`, optional, defaults to None):
            Number of semantic concepts.
            If None, then universal latent semantic embeddings will not be used.
        language_specific_transformation (:obj:`bool`, optional, defaults to False):
            Whether or not to use language specific transformations
        n_languages (:obj:`int`, optional, defaults to 1):
            Number of languages for language specific layers
        n_layer (:obj:`int`, optional, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (:obj:`int`, optional, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (:obj:`int`, optional, defaults to None):
            Dimensionality of the inner feed-forward layers. :obj:`None` will set it to 4 times n_embd
        activation_function (:obj:`str`, optional, defaults to 'gelu'):
            Activation function selected in the list ["relu", "swish", "gelu", "tanh", "gelu_new"].
        resid_pdrop (:obj:`float`, optional, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (:obj:`int`, optional, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (:obj:`float`, optional, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (:obj:`float`, optional, defaults to 1e-5):
            The epsilon to use in the layer normalization layers
        initializer_range (:obj:`float`, optional, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example::

        >>> from transformers import GPT2Model, GPT2Config

        >>> # Initializing a GPT2 configuration
        >>> configuration = GPT2Config()

        >>> # Initializing a model from the configuration
        >>> model = GPT2Model(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "gpt2"

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_language_specific_attention_layers=0,
        language_specific_input_embeds=False,
        language_specific_prediction_heads=False,
        language_specific_transformation=False,
        semantic_concepts=None,
        n_languages=1,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_language_specific_attention_layers = n_language_specific_attention_layers
        self.language_specific_input_embeds = language_specific_input_embeds
        self.language_specific_prediction_heads = language_specific_prediction_heads
        self.language_specific_transformation = language_specific_transformation
        self.semantic_concepts = semantic_concepts
        self.n_languages = n_languages
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer



class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False, n_languages=1):
        super().__init__()
        self.n_languages = n_languages
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = nn.ModuleList([Attention(hidden_size, n_ctx, config, scale) for _ in range(n_languages)])
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # if config.add_cross_attention:
        #     self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
        #     self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        language=0,
    ):
        assert language < self.n_languages

        attn_outputs = self.attn[language](
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + hidden_states

        # if encoder_hidden_states is not None:
        #     # add one self-attention block for cross-attention
        #     assert hasattr(
        #         self, "crossattention"
        #     ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
        #     cross_attn_outputs = self.crossattention(
        #         self.ln_cross_attn(hidden_states),
        #         attention_mask=attention_mask,
        #         head_mask=head_mask,
        #         encoder_hidden_states=encoder_hidden_states,
        #         encoder_attention_mask=encoder_attention_mask,
        #         output_attentions=output_attentions,
        #     )
        #     attn_output = cross_attn_outputs[0]
        #     # residual connection
        #     hidden_states = hidden_states + attn_output
        #     outputs = outputs + cross_attn_outputs[1:]  # add cross attentions if we output attention weights

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        outputs = [hidden_states] + outputs
        return outputs  # hidden_states, present, (cross_attentions, attentions)


class SemanticEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,):
        super().__init__()

        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        init.normal_(self.weight)

    def forward(
            self,
            input_embeddings,
    ):
        assert input_embeddings.size()[-1] == self.weight.size()[1]
        temp = torch.matmul(input_embeddings, self.weight.t())
        temp = torch.softmax(temp, dim=2)
        return torch.matmul(temp, self.weight)


class LayerSwitchingGPT2Model(GPT2PreTrainedModel):

    config_class = LayerSwitchingGPT2Config

    def __init__(self, config):
        super().__init__(config)

        if config.semantic_concepts is not None:
            self.lse = SemanticEmbedding(config.semantic_concepts, config.n_embd)

        if config.language_specific_input_embeds:
            self.wte = nn.ModuleList([
                nn.Embedding(config.vocab_size, config.n_embd)
                for _ in range(config.n_languages)
            ])
        else:
            self.wte = nn.ModuleList([
                nn.Embedding(config.vocab_size, config.n_embd)
            ])

        if self.config.language_specific_transformation:
            self.language_specific_transformations = nn.ModuleList([
                nn.Linear(self.config.n_embd, self.config.n_embd, bias=False)
                for _ in range(config.n_languages)
            ])

        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        self.drop = nn.Dropout(config.embd_pdrop)

        self.h = nn.ModuleList()
        for i in range(config.n_layer):
            if i < config.n_language_specific_attention_layers:
                self.h.append(Block(config.n_ctx, config, scale=True, n_languages=config.n_languages))
            else:
                self.h.append(Block(config.n_ctx, config, scale=True, n_languages=1))

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()


    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D, SemanticEmbedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    # @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="gpt2",
    #     output_type=BaseModelOutputWithPast,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        language=0,
        **kwargs,
    ):
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            if self.config.language_specific_input_embeds:
                inputs_embeds = self.wte[language](input_ids)
            else:
                inputs_embeds = self.wte[0](input_ids)

        if self.config.language_specific_transformation:
            inputs_embeds = torch.tanh(self.language_specific_transformations[language](inputs_embeds))

        if self.config.semantic_concepts is not None:
            inputs_embeds = inputs_embeds + self.lse(inputs_embeds)

        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                # encoder_hidden_states=encoder_hidden_states,
                # encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                language=language if i < self.config.n_language_specific_attention_layers else 0,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions = all_attentions + (outputs[2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class GPT2LayerSwitchingLMHeadModel(GPT2PreTrainedModel):

    config_class = LayerSwitchingGPT2Config

    def __init__(self, config):
        super().__init__(config)

        if self.config.language_specific_input_embeds != self.config.language_specific_prediction_heads:
            assert not config.tie_word_embeddings

        assert config.n_languages is not None and config.n_languages > 0, 'n_languages must be positive'

        self.transformer = LayerSwitchingGPT2Model(config)

        if config.language_specific_prediction_heads:
            # create LM head for each language
            self.lm_heads = nn.ModuleList([nn.Linear(config.n_embd, config.vocab_size, bias=False) for _ in range(config.n_languages)])
        else:
            self.lm_heads = nn.ModuleList([nn.Linear(config.n_embd, config.vocab_size, bias=False)])

        self.init_weights()

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        if self.config.tie_word_embeddings:
            for input_embeddings, output_embeddings in zip(self.get_input_embeddings(), self.get_output_embeddings()):
                self._tie_or_clone_weights(output_embeddings, input_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_heads

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        language=0,
        position_ids=None,
        head_mask=None,
        # inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            language=language,
        )
        hidden_states = transformer_outputs[0]

        if self.config.language_specific_prediction_heads:
            lm_logits = self.lm_heads[language](hidden_states)
        else:
            lm_logits = self.lm_heads[0](hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
