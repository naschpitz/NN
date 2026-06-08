#ifndef NN_CLI_AUGMENT_CPP_CL
#define NN_CLI_AUGMENT_CPP_CL

//===================================================================================================================//
// NN-CLI_Augment.cpp.cl
//
// GPU image-augmentation kernels, mirroring NN-CLI_ImageLoader's CPU transforms.
//
// Data layout: a batch of N images, each NCHW-flat and normalised to [0, 1].
//   sample s, channel c, pixel (y, x)  ->  index  s*(C*H*W) + c*(H*W) + y*W + x
//
// Per-sample randomness is drawn on the host (coin flips + parameters), exactly as
// the CPU path does, and passed in as per-sample arrays. The kernels are therefore
// deterministic given their parameters. Geometric transforms read `src` and write
// `dst` (the host ping-pongs the two batch buffers); element-wise transforms run
// in place. A disabled transform is simply not enqueued.
//
// "apply" arrays are per-sample int flags (1 = apply, 0 = pass through). Out-of-place
// kernels always write every output pixel (copying through when apply == 0), so the
// ping-pong stays consistent.
//===================================================================================================================//

//-- Helpers ---------------------------------------------------------------------------------------------------------//

inline float aug_clamp01(float v)
{
  return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// Bilinear sample with black (0) fill for out-of-bounds; used by rotation/translation/scaling.
inline float aug_sample_bilinear_black(global const float* src, ulong base, long H, long W, float sy, float sx)
{
  long x0 = (long)floor(sx);
  long y0 = (long)floor(sy);
  long x1 = x0 + 1;
  long y1 = y0 + 1;
  float fx = sx - (float)x0;
  float fy = sy - (float)y0;

  float v00 = (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) ? src[base + (ulong)(y0 * W + x0)] : 0.0f;
  float v10 = (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H) ? src[base + (ulong)(y0 * W + x1)] : 0.0f;
  float v01 = (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H) ? src[base + (ulong)(y1 * W + x0)] : 0.0f;
  float v11 = (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) ? src[base + (ulong)(y1 * W + x1)] : 0.0f;

  return (1.0f - fx) * (1.0f - fy) * v00 + fx * (1.0f - fy) * v10 + (1.0f - fx) * fy * v01 + fx * fy * v11;
}

// Bilinear sample with edge clamp; used by elastic deformation.
inline float aug_sample_bilinear_clamp(global const float* src, ulong base, long H, long W, float sy, float sx)
{
  long x0 = (long)floor(sx);
  long y0 = (long)floor(sy);
  long x1 = x0 + 1;
  long y1 = y0 + 1;
  float fx = sx - (float)x0;
  float fy = sy - (float)y0;

  long cx0 = clamp(x0, (long)0, W - 1);
  long cx1 = clamp(x1, (long)0, W - 1);
  long cy0 = clamp(y0, (long)0, H - 1);
  long cy1 = clamp(y1, (long)0, H - 1);

  float v00 = src[base + (ulong)(cy0 * W + cx0)];
  float v10 = src[base + (ulong)(cy0 * W + cx1)];
  float v01 = src[base + (ulong)(cy1 * W + cx0)];
  float v11 = src[base + (ulong)(cy1 * W + cx1)];

  return (1.0f - fx) * (1.0f - fy) * v00 + fx * (1.0f - fy) * v10 + (1.0f - fx) * fy * v01 + fx * fy * v11;
}

// Hash-based per-element PRNG -> uniform [0,1). Deterministic given (seed, gid).
inline uint aug_hash(uint x)
{
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

inline float aug_rand01(uint seed, ulong gid)
{
  uint h = aug_hash(seed ^ aug_hash((uint)(gid & 0xffffffffU)) ^ aug_hash((uint)(gid >> 32)));
  return (float)(h & 0x00ffffffU) / (float)0x01000000U;
}

//===================================================================================================================//
//-- Elastic displacement field generation (GPU) --//
//===================================================================================================================//

// Fill per-pixel displacement fields with uniform noise in [-1, 1]. nElements = N*H*W.
kernel void aug_field_random(global float* dx, global float* dy, uint seed, ulong total)
{
  ulong gid = get_global_id(0);
  if (gid >= total)
    return;

  dx[gid] = 2.0f * aug_rand01(seed, gid * 2UL) - 1.0f;
  dy[gid] = 2.0f * aug_rand01(seed, gid * 2UL + 1UL) - 1.0f;
}

// Separable Gaussian blur of a per-sample field (edge-clamped). One pass (horizontal
// when horizontal != 0, else vertical); the host runs it twice with a temp buffer.
kernel void aug_field_blur(global const float* in, global float* out, global const float* gauss, int radius, ulong H,
                           ulong W, int horizontal, ulong total)
{
  ulong gid = get_global_id(0);
  if (gid >= total)
    return;

  ulong hw = H * W;
  ulong s = gid / hw;
  ulong p = gid % hw;
  long y = (long)(p / W);
  long x = (long)(p % W);
  ulong base = s * hw;

  float acc = 0.0f;
  for (int k = -radius; k <= radius; k++) {
    long yy = y, xx = x;
    if (horizontal)
      xx = clamp(x + k, (long)0, (long)W - 1);
    else
      yy = clamp(y + k, (long)0, (long)H - 1);
    acc += in[base + (ulong)(yy * (long)W + xx)] * gauss[k + radius];
  }

  out[gid] = acc;
}

//===================================================================================================================//
//-- Geometric transforms (out-of-place; one work-item per output element) --//
//===================================================================================================================//

// Mirror along the vertical axis. nElements = N*C*H*W.
kernel void aug_flip(global const float* src, global float* dst, global const int* apply, ulong C, ulong H, ulong W,
                     ulong total)
{
  ulong gid = get_global_id(0);
  if (gid >= total)
    return;

  ulong chw = C * H * W;
  ulong s = gid / chw;
  ulong rem = gid % chw;
  ulong hw = H * W;
  ulong c = rem / hw;
  ulong p = rem % hw;
  ulong y = p / W;
  ulong x = p % W;
  ulong base = s * chw + c * hw;

  if (apply[s])
    dst[base + y * W + x] = src[base + y * W + (W - 1 - x)];
  else
    dst[gid] = src[gid];
}

// Rotation about the image centre (bilinear, black fill). angleRad per sample.
kernel void aug_rotate(global const float* src, global float* dst, global const int* apply,
                       global const float* angleRad, ulong C, ulong H, ulong W, ulong total)
{
  ulong gid = get_global_id(0);
  if (gid >= total)
    return;

  ulong chw = C * H * W;
  ulong s = gid / chw;
  ulong rem = gid % chw;
  ulong hw = H * W;
  ulong c = rem / hw;
  ulong p = rem % hw;
  long y = (long)(p / W);
  long x = (long)(p % W);
  ulong base = s * chw + c * hw;

  if (!apply[s]) {
    dst[gid] = src[gid];
    return;
  }

  float a = angleRad[s];
  float ca = cos(a);
  float sa = sin(a);
  float cx = (float)W / 2.0f;
  float cy = (float)H / 2.0f;

  float sx = ca * ((float)x - cx) + sa * ((float)y - cy) + cx;
  float sy = -sa * ((float)x - cx) + ca * ((float)y - cy) + cy;

  dst[base + (ulong)(y * (long)W + x)] = aug_sample_bilinear_black(src, base, (long)H, (long)W, sy, sx);
}

// Integer translation (nearest, black fill). dx/dy per sample (pixels).
kernel void aug_translate(global const float* src, global float* dst, global const int* apply, global const int* dxs,
                          global const int* dys, ulong C, ulong H, ulong W, ulong total)
{
  ulong gid = get_global_id(0);
  if (gid >= total)
    return;

  ulong chw = C * H * W;
  ulong s = gid / chw;
  ulong rem = gid % chw;
  ulong hw = H * W;
  ulong c = rem / hw;
  ulong p = rem % hw;
  long y = (long)(p / W);
  long x = (long)(p % W);
  ulong base = s * chw + c * hw;

  if (!apply[s]) {
    dst[gid] = src[gid];
    return;
  }

  long srcX = x - dxs[s];
  long srcY = y - dys[s];

  dst[base + (ulong)(y * (long)W + x)] =
    (srcX >= 0 && srcX < (long)W && srcY >= 0 && srcY < (long)H) ? src[base + (ulong)(srcY * (long)W + srcX)] : 0.0f;
}

// Scale about the centre (bilinear, black fill), matching the CPU resize + centre-crop.
kernel void aug_scale(global const float* src, global float* dst, global const int* apply, global const float* scales,
                      ulong C, ulong H, ulong W, ulong total)
{
  ulong gid = get_global_id(0);
  if (gid >= total)
    return;

  ulong chw = C * H * W;
  ulong s = gid / chw;
  ulong rem = gid % chw;
  ulong hw = H * W;
  ulong c = rem / hw;
  ulong p = rem % hw;
  long y = (long)(p / W);
  long x = (long)(p % W);
  ulong base = s * chw + c * hw;

  if (!apply[s]) {
    dst[gid] = src[gid];
    return;
  }

  float scale = scales[s];
  long newH = (long)round((float)H * scale);
  long newW = (long)round((float)W * scale);
  long offsetY = (newH - (long)H) / 2;
  long offsetX = (newW - (long)W) / 2;

  long Y = y + offsetY; // position in the resized image
  long X = x + offsetX;

  if (Y < 0 || Y >= newH || X < 0 || X >= newW) {
    dst[base + (ulong)(y * (long)W + x)] = 0.0f;
    return;
  }

  // resized(Y,X) bilinearly samples src at (Y*H/newH, X*W/newW)
  float sy = (float)Y * (float)H / (float)newH;
  float sx = (float)X * (float)W / (float)newW;
  dst[base + (ulong)(y * (long)W + x)] = aug_sample_bilinear_black(src, base, (long)H, (long)W, sy, sx);
}

// Apply a displacement field (bilinear, edge clamp), scaled by alpha. field[s] is H*W dx and dy.
kernel void aug_elastic_apply(global const float* src, global float* dst, global const int* apply,
                              global const float* dxField, global const float* dyField, float alpha, ulong C, ulong H,
                              ulong W, ulong total)
{
  ulong gid = get_global_id(0);
  if (gid >= total)
    return;

  ulong chw = C * H * W;
  ulong s = gid / chw;
  ulong rem = gid % chw;
  ulong hw = H * W;
  ulong c = rem / hw;
  ulong p = rem % hw;
  long y = (long)(p / W);
  long x = (long)(p % W);
  ulong base = s * chw + c * hw;

  if (!apply[s]) {
    dst[gid] = src[gid];
    return;
  }

  ulong fieldBase = s * hw + (ulong)p;
  float sy = (float)y + alpha * dyField[fieldBase];
  float sx = (float)x + alpha * dxField[fieldBase];
  dst[base + (ulong)(y * (long)W + x)] = aug_sample_bilinear_clamp(src, base, (long)H, (long)W, sy, sx);
}

//===================================================================================================================//
//-- Colour / intensity transforms (in-place; one work-item per element) --//
//===================================================================================================================//

// Brightness: v += delta. nElements = N*C*H*W.
kernel void aug_brightness(global float* img, global const int* apply, global const float* deltas, ulong C, ulong H,
                           ulong W, ulong total)
{
  ulong gid = get_global_id(0);
  if (gid >= total)
    return;

  ulong chw = C * H * W;
  ulong s = gid / chw;
  if (!apply[s])
    return;

  img[gid] = aug_clamp01(img[gid] + deltas[s]);
}

// Per-(sample,channel) mean reduction -> means[s*C + c]. One work-group per (sample,channel).
kernel void aug_channel_mean(global const float* img, global float* means, ulong C, ulong H, ulong W)
{
  local float partials[256];

  ulong sc = get_group_id(0); // flattened (sample, channel)
  ulong lid = get_local_id(0);
  ulong localSize = get_local_size(0);

  ulong hw = H * W;
  ulong base = sc * hw;

  float sum = 0.0f;
  for (ulong i = lid; i < hw; i += localSize)
    sum += img[base + i];

  partials[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (ulong stride = localSize / 2; stride > 0; stride >>= 1) {
    if (lid < stride)
      partials[lid] += partials[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0)
    means[sc] = partials[0] / (float)hw;
}

// Contrast around the per-channel mean: v = mean + factor*(v - mean).
kernel void aug_contrast(global float* img, global const int* apply, global const float* factors,
                         global const float* means, ulong C, ulong H, ulong W, ulong total)
{
  ulong gid = get_global_id(0);
  if (gid >= total)
    return;

  ulong chw = C * H * W;
  ulong s = gid / chw;
  if (!apply[s])
    return;

  ulong hw = H * W;
  ulong c = (gid % chw) / hw;
  float mean = means[s * C + c];
  img[gid] = aug_clamp01(mean + factors[s] * (img[gid] - mean));
}

// Hue shift (RGB only, C == 3). shiftDeg per sample. One work-item per (sample, pixel).
kernel void aug_hue(global float* img, global const int* apply, global const float* shiftDeg, ulong H, ulong W,
                    ulong totalPixels)
{
  ulong gid = get_global_id(0); // over N*H*W
  if (gid >= totalPixels)
    return;

  ulong hw = H * W;
  ulong s = gid / hw;
  if (!apply[s])
    return;

  ulong p = gid % hw;
  ulong base = s * 3 * hw; // 3 channels
  ulong ri = base + 0 * hw + p;
  ulong gi = base + 1 * hw + p;
  ulong bi = base + 2 * hw + p;

  float r = img[ri];
  float g = img[gi];
  float b = img[bi];

  float maxC = fmax(r, fmax(g, b));
  float minC = fmin(r, fmin(g, b));
  float delta = maxC - minC;

  float sat = (maxC > 0.0f) ? delta / maxC : 0.0f;
  float val = maxC;
  float hue = 0.0f;

  if (delta > 0.0f) {
    if (maxC == r)
      hue = 60.0f * fmod(((g - b) / delta), 6.0f);
    else if (maxC == g)
      hue = 60.0f * (((b - r) / delta) + 2.0f);
    else
      hue = 60.0f * (((r - g) / delta) + 4.0f);
    if (hue < 0.0f)
      hue += 360.0f;
  }

  hue = fmod(hue + shiftDeg[s] + 360.0f, 360.0f);

  float chroma = val * sat;
  float hPrime = hue / 60.0f;
  float xVal = chroma * (1.0f - fabs(fmod(hPrime, 2.0f) - 1.0f));
  float m = val - chroma;

  float rp = 0.0f, gp = 0.0f, bp = 0.0f;
  if (hPrime < 1.0f) {
    rp = chroma;
    gp = xVal;
  } else if (hPrime < 2.0f) {
    rp = xVal;
    gp = chroma;
  } else if (hPrime < 3.0f) {
    gp = chroma;
    bp = xVal;
  } else if (hPrime < 4.0f) {
    gp = xVal;
    bp = chroma;
  } else if (hPrime < 5.0f) {
    rp = xVal;
    bp = chroma;
  } else {
    rp = chroma;
    bp = xVal;
  }

  img[ri] = aug_clamp01(rp + m);
  img[gi] = aug_clamp01(gp + m);
  img[bi] = aug_clamp01(bp + m);
}

// Random erasing: zero a per-sample rectangle [x0, x0+ew) x [y0, y0+eh).
kernel void aug_erase(global float* img, global const int* apply, global const int* x0s, global const int* y0s,
                      global const int* ews, global const int* ehs, ulong C, ulong H, ulong W, ulong total)
{
  ulong gid = get_global_id(0);
  if (gid >= total)
    return;

  ulong chw = C * H * W;
  ulong s = gid / chw;
  if (!apply[s])
    return;

  ulong hw = H * W;
  ulong p = (gid % chw) % hw;
  long y = (long)(p / W);
  long x = (long)(p % W);

  if (x >= x0s[s] && x < x0s[s] + ews[s] && y >= y0s[s] && y < y0s[s] + ehs[s])
    img[gid] = 0.0f;
}

// Additive Gaussian noise via per-element PRNG (Box-Muller). stddev scalar, seed per batch.
kernel void aug_gaussian_noise(global float* img, global const int* apply, float stddev, uint seed, ulong C, ulong H,
                               ulong W, ulong total)
{
  ulong gid = get_global_id(0);
  if (gid >= total)
    return;

  ulong chw = C * H * W;
  ulong s = gid / chw;
  if (!apply[s])
    return;

  float u1 = fmax(aug_rand01(seed, gid * 2UL), 1e-7f);
  float u2 = aug_rand01(seed, gid * 2UL + 1UL);
  float n = sqrt(-2.0f * log(u1)) * cos(6.2831853f * u2); // standard normal
  img[gid] = aug_clamp01(img[gid] + n * stddev);
}

//===================================================================================================================//

#endif // NN_CLI_AUGMENT_CPP_CL
