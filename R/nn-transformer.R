#' @include nn.R
NULL

nn_transformer_encoder_layer <- nn_module(
  classname = "nn_transformer_encoder",
  initialize = function(
      d_model,
      num_heads,
      dim_feedforward = 2048,
      dropout = 0.1,
      activation = "relu",
      layer_norm_eps = 1e-5,
      batch_first = FALSE,
      norm_first = FALSE,
      bias = TRUE) {
    self$self_attn <- nn_multihead_attention(
      embed_dim = d_model,
      num_heads = num_heads,
      dropout = dropout,
      bias = bias,
      batch_first = batch_first
    )

    if(!activation %in% c('relu', 'gelu')) {
      value_error("Only 'relu' and 'gelu' supported for `activation`")
    }

    self$linear1 <- nn_linear(in_features = d_model, out_features = dim_feedforward, bias = bias)
    self$dropout <- nn_dropout(p = dropout)
    self$linear2 <- nn_linear(in_features = dim_feedforward, out_features = d_model, bias = bias)

    self$norm_first <- norm_first
    self$norm1 <- nn_layer_norm(normalized_shape = d_model, eps = layer_norm_eps, elementwise_affine = bias)
    self$norm2 <- nn_layer_norm(normalized_shape = d_model, eps = layer_norm_eps, elementwise_affine = bias)
    self$dropout1 <- nn_dropout(p = dropout)
    self$dropout2 <- nn_dropout(p = dropout)

    self$activation <- switch(activation,
      "relu" = function(x) nn_relu(x),
      "gelu" = function(x) nn_gelu(x),
      stop("Unsupported activation function")
    )
  },
  forward = function(x, src_mask = NULL, src_key_padding_mask = NULL) {
    x
  },
  sa_block = function(x, attn_mask, key_padding_mask) {
    # this takes in the args from foward and runs through multi head attention
    out <- self$self_attn(
      query = x,
      key = x,
      value = x,
      attn_mask = attn_mask,
      key_padding_mask = key_padding_mask,
      need_weights = FALSE
    )[[1]]
    self$dropout1(out)
  },
  ff_block = function(x) {
    out <- self$linear1(x)
    out <- self$activation(out)
    out <- self$dropout(out)
    out <- self$linear2(out)
    self$dropout2(out)
  }
)
