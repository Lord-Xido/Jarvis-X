# Geometric and mathematical mechanism

Define a periodic physical lattice \(\Omega=(\mathbb Z/16\mathbb Z)^3\) with \(V=4096\) uint8 voxels and a coarse lattice \(\Lambda=(\mathbb Z/8\mathbb Z)^3\) with \(C=512\) uint8 latents. Phases \(p=o_x+2o_y+4o_z\in[0,7]\).

**Encoder:** Train eight logits \(u_p\) with normalized softmax weights \(\alpha_p = e^{u_p}/\sum_q e^{u_q}\); then store \(Z_c = \mathrm{round}(255\sum_p \alpha_p X_{c,p}/255)\).

**Decoder:** For each of eight phases, learn a 3×3×3 periodic coarse-neighborhood kernel and a bias:
\[
h_{c,p}=b_p+\sum_{\delta\in\{-1,0,1\}^3}w_{p,\delta}(Z_{c+\delta}-Z_c)/255,
\quad
\hat X_{c,p}=\mathrm{clip}_{[0,1]}(Z_c/255+0.32\tanh(h_{c,p})).
\]
The complete model has 8 + 8(27+1) = 232 parameters. Adam minimizes voxelwise squared error across six synthetic periodic input volumes. A straight-through surrogate of the quantized encoder is used for gradients (\(\partial Q/\partial z \approx 1\)); decoder gradients propagate through the 27 coarse neighbor sites to the encoder weights. These are not the exact derivatives of quantization.

**Feedback:** \(S_{t+1}=0.5S_t+0.5\hat X\) for six iterations from the untrained prediction. In exact arithmetic \(\|S_t-\hat X\|_\infty=2^{-t}\|S_0-\hat X\|_\infty\). The code checks this up to numerical tolerance and enforces finite values in [0,1]. The feedback step need not improve RMSE.

**Exact codec:** Round the neural prediction \(q_i = \mathrm{round}(255\hat X_i)\); save signed 16-bit residual \(r_i=X_i-q_i\). Reconstruct \(q_i+r_i=X_i\) exactly, validate range and CRC32. The neural packet includes the complete binary32 model (928 bytes), 512 latent bytes and Zstandard level-3 compressed residuals. Compare a common 20-byte-header representation for four candidates: raw, Zstandard-3 raw, learned residual, mean-coarse residual. Select minimum actual packet bytes; independently reconstruct every candidate and verify equality.

**Packet:** little endian: magic "JX3D" [4], version [u8=1], method [u8=0..3], flags [u16=0], original count [u32=4096], payload count [u32], source CRC32 [u32], then payload. CRC32 is not authentication. Floating point prediction equality across CPUs is not guaranteed without a fixed-point decode contract.

**Default measured case** (synthetic holdouts, GCC -O3; 85 epochs): held-out A nearest RMSE 0.0327608, trilinear 0.0174948, joint trained 0.0019361; learned packet 2730 bytes versus raw-Zstandard 3812 bytes. Held-out B trained RMSE 0.0018917, learned packet 2710 bytes versus raw-Zstandard 3822. On independent random voxels the learned predictor worsens (RMSE 0.2782073 vs 0.2723641 nearest). On block-shaped data raw Zstandard is best (161-byte packet). One-second execution observed locally but not guaranteed elsewhere; no real clinical 3D dataset was used.
