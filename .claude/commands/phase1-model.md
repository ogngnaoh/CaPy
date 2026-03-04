Implement all model components for CaPy. Create these files:

1. src/models/encoders.py
   MolecularEncoder: 5-layer GIN (hidden_dim=300), global mean pool, projection MLP (300->512->256), L2 normalize. Use torch_geometric.nn.GINConv with batch norms.
   TabularEncoder: configurable input_dim, 3 MLP layers (input->512, 512->512, 512->512) with LayerNorm, ReLU, Dropout(0.1). Residual connections on layers 2-3. Projection head (512->256), L2 normalize.
   MorphologyEncoder = TabularEncoder(input_dim=config)
   ExpressionEncoder = TabularEncoder(input_dim=config)

2. src/models/losses.py
   info_nce(z_a, z_b, temperature): symmetric InfoNCE loss. Compute similarity matrix z_a @ z_b.T / temperature. Subtract max for numerical stability. CrossEntropy both directions. Average.

3. src/models/capy.py
   CaPy(nn.Module) with mol_encoder, morph_encoder, expr_encoder, learnable log_temperature (init log(1/0.07)). Temperature clamped to [0.01, 10.0]. compute_loss returns weighted sum of 3 pairwise InfoNCE losses + per-pair loss dict.

4. tests/test_models.py
   Shape checks: all encoders output [batch_size, 256]. Normalization check: output norms approximately 1.0. Gradient flow: all parameters have non-None gradients after backward. Device placement: works on CPU.

5. tests/test_losses.py
   Perfect matching loss approximately log(batch_size). Random embeddings produce higher loss. Symmetry: info_nce(a,b,t) == info_nce(b,a,t). Batch size 1 does not crash.

After implementation, use the architect-explainer agent to explain: why GIN is provably powerful (WL test connection), why L2 normalize BEFORE similarity, why learnable temperature outperforms fixed, what projection heads do (SimCLR finding), why residual connections in the MLP encoders.
