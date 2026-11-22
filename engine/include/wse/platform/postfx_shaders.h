// Platform seam: R5b Vulkan fullscreen post-processing shaders and push-constant
// helpers (`docs/RECOMP_PLAN.md` "R5 — Graphics").
//
// Port infrastructure, not a reconstruction: mirrors no retail v1.1.5.6 layout
// and carries no PE/PDB citations. Standard library only; `<vulkan/vulkan.h>`
// stays out of portable engine headers.
//
// Shaders use GLSL 450 (Vulkan 1.0 / SPIR-V 1.0 compatible, fp32, no
// extensions). All fragment shaders share `kPostVertex` (`location = 0` `vec2 uv`)
// and bind the input colour image at `set = 0, binding = 0` (`sampler2D src`).

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace wse::platform {

// Fullscreen triangle vertex shader generated from `gl_VertexIndex` (draw 3
// vertices with no vertex buffer). Covers NDC [-1, 1]^2 and outputs
// top-left-origin `uv` in [0, 1]^2 across the viewport.
inline constexpr std::string_view kPostVertex = R"glsl(#version 450

layout(location = 0) out vec2 uv;

void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)glsl";

// Ported from AMD FidelityFX Super Resolution 1.0 (`ffx-fsr/ffx_fsr1.h` +
// `ffx-fsr/ffx_a.h`, GPUOpen-Effects/FidelityFX-FSR commit
// `a21ffb8f6c13233ba336352bdff293894c706575`) to plain fp32 GLSL 450.
//
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
inline constexpr std::string_view kFsrEasu = R"glsl(#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D src;

layout(push_constant) uniform Push {
    vec4 con0;
    vec4 con1;
    vec4 con2;
    vec4 con3;
} pc;

float fsrAprxLoRcp(float a) {
    return uintBitsToFloat(0x7ef07ebbu - floatBitsToUint(a));
}

float fsrAprxLoRsq(float a) {
    return uintBitsToFloat(0x5f347d74u - (floatBitsToUint(a) >> 1u));
}

void fsrEasuTap(
    inout vec3 aC,
    inout float aW,
    vec2 off,
    vec2 dir,
    vec2 len,
    float lob,
    float clp,
    vec3 c
) {
    vec2 v;
    v.x = (off.x * dir.x) + (off.y * dir.y);
    v.y = (off.x * (-dir.y)) + (off.y * dir.x);
    v *= len;
    float d2 = v.x * v.x + v.y * v.y;
    d2 = min(d2, clp);
    float wB = (2.0 / 5.0) * d2 - 1.0;
    float wA = lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = (25.0 / 16.0) * wB - (25.0 / 16.0 - 1.0);
    float w = wB * wA;
    aC += c * w;
    aW += w;
}

void fsrEasuSet(
    inout vec2 dir,
    inout float len,
    vec2 pp,
    bool biS,
    bool biT,
    bool biU,
    bool biV,
    float lA,
    float lB,
    float lC,
    float lD,
    float lE
) {
    float w = 0.0;
    if (biS) w = (1.0 - pp.x) * (1.0 - pp.y);
    if (biT) w = pp.x * (1.0 - pp.y);
    if (biU) w = (1.0 - pp.x) * pp.y;
    if (biV) w = pp.x * pp.y;

    float dc = lD - lC;
    float cb = lC - lB;
    float lenX = max(abs(dc), abs(cb));
    lenX = fsrAprxLoRcp(lenX);
    float dirX = lD - lB;
    dir.x += dirX * w;
    lenX = clamp(abs(dirX) * lenX, 0.0, 1.0);
    lenX *= lenX;
    len += lenX * w;

    float ec = lE - lC;
    float ca = lC - lA;
    float lenY = max(abs(ec), abs(ca));
    lenY = fsrAprxLoRcp(lenY);
    float dirY = lE - lA;
    dir.y += dirY * w;
    lenY = clamp(abs(dirY) * lenY, 0.0, 1.0);
    lenY *= lenY;
    len += lenY * w;
}

void main() {
    uvec2 ip = uvec2(gl_FragCoord.xy);
    vec2 pp = vec2(ip) * pc.con0.xy + pc.con0.zw;
    vec2 fp = floor(pp);
    pp -= fp;

    vec2 p0 = fp * pc.con1.xy + pc.con1.zw;
    vec2 p1 = p0 + pc.con2.xy;
    vec2 p2 = p0 + pc.con2.zw;
    vec2 p3 = p0 + pc.con3.xy;

    vec4 bczzR = textureGather(src, p0, 0);
    vec4 bczzG = textureGather(src, p0, 1);
    vec4 bczzB = textureGather(src, p0, 2);
    vec4 ijfeR = textureGather(src, p1, 0);
    vec4 ijfeG = textureGather(src, p1, 1);
    vec4 ijfeB = textureGather(src, p1, 2);
    vec4 klhgR = textureGather(src, p2, 0);
    vec4 klhgG = textureGather(src, p2, 1);
    vec4 klhgB = textureGather(src, p2, 2);
    vec4 zzonR = textureGather(src, p3, 0);
    vec4 zzonG = textureGather(src, p3, 1);
    vec4 zzonB = textureGather(src, p3, 2);

    vec4 bczzL = bczzB * 0.5 + (bczzR * 0.5 + bczzG);
    vec4 ijfeL = ijfeB * 0.5 + (ijfeR * 0.5 + ijfeG);
    vec4 klhgL = klhgB * 0.5 + (klhgR * 0.5 + klhgG);
    vec4 zzonL = zzonB * 0.5 + (zzonR * 0.5 + zzonG);

    float bL = bczzL.x;
    float cL = bczzL.y;
    float iL = ijfeL.x;
    float jL = ijfeL.y;
    float fL = ijfeL.z;
    float eL = ijfeL.w;
    float kL = klhgL.x;
    float lL = klhgL.y;
    float hL = klhgL.z;
    float gL = klhgL.w;
    float oL = zzonL.z;
    float nL = zzonL.w;

    vec2 dir = vec2(0.0);
    float len = 0.0;
    fsrEasuSet(dir, len, pp, true, false, false, false, bL, eL, fL, gL, jL);
    fsrEasuSet(dir, len, pp, false, true, false, false, cL, fL, gL, hL, kL);
    fsrEasuSet(dir, len, pp, false, false, true, false, fL, iL, jL, kL, nL);
    fsrEasuSet(dir, len, pp, false, false, false, true, gL, jL, kL, lL, oL);

    vec2 dir2 = dir * dir;
    float dirR = dir2.x + dir2.y;
    bool zro = dirR < (1.0 / 32768.0);
    dirR = fsrAprxLoRsq(dirR);
    dirR = zro ? 1.0 : dirR;
    dir.x = zro ? 1.0 : dir.x;
    dir *= dirR;

    len = len * 0.5;
    len *= len;

    float stretch = (dir.x * dir.x + dir.y * dir.y) * fsrAprxLoRcp(max(abs(dir.x), abs(dir.y)));
    vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    float clp = fsrAprxLoRcp(lob);

    vec3 tapF = vec3(ijfeR.z, ijfeG.z, ijfeB.z);
    vec3 tapG = vec3(klhgR.w, klhgG.w, klhgB.w);
    vec3 tapJ = vec3(ijfeR.y, ijfeG.y, ijfeB.y);
    vec3 tapK = vec3(klhgR.x, klhgG.x, klhgB.x);
    vec3 min4 = min(min(tapF, tapG), min(tapJ, tapK));
    vec3 max4 = max(max(tapF, tapG), max(tapJ, tapK));

    vec3 aC = vec3(0.0);
    float aW = 0.0;
    fsrEasuTap(aC, aW, vec2(0.0, -1.0) - pp, dir, len2, lob, clp, vec3(bczzR.x, bczzG.x, bczzB.x));
    fsrEasuTap(aC, aW, vec2(1.0, -1.0) - pp, dir, len2, lob, clp, vec3(bczzR.y, bczzG.y, bczzB.y));
    fsrEasuTap(aC, aW, vec2(-1.0, 1.0) - pp, dir, len2, lob, clp, vec3(ijfeR.x, ijfeG.x, ijfeB.x));
    fsrEasuTap(aC, aW, vec2(0.0, 1.0) - pp, dir, len2, lob, clp, tapJ);
    fsrEasuTap(aC, aW, vec2(0.0, 0.0) - pp, dir, len2, lob, clp, tapF);
    fsrEasuTap(aC, aW, vec2(-1.0, 0.0) - pp, dir, len2, lob, clp, vec3(ijfeR.w, ijfeG.w, ijfeB.w));
    fsrEasuTap(aC, aW, vec2(1.0, 1.0) - pp, dir, len2, lob, clp, tapK);
    fsrEasuTap(aC, aW, vec2(2.0, 1.0) - pp, dir, len2, lob, clp, vec3(klhgR.y, klhgG.y, klhgB.y));
    fsrEasuTap(aC, aW, vec2(2.0, 0.0) - pp, dir, len2, lob, clp, vec3(klhgR.z, klhgG.z, klhgB.z));
    fsrEasuTap(aC, aW, vec2(1.0, 0.0) - pp, dir, len2, lob, clp, tapG);
    fsrEasuTap(aC, aW, vec2(1.0, 2.0) - pp, dir, len2, lob, clp, vec3(zzonR.z, zzonG.z, zzonB.z));
    fsrEasuTap(aC, aW, vec2(0.0, 2.0) - pp, dir, len2, lob, clp, vec3(zzonR.w, zzonG.w, zzonB.w));

    vec3 pix = min(max4, max(min4, aC / aW));
    fragColor = vec4(pix, 1.0);
}
)glsl";

// Ported from AMD FidelityFX Super Resolution 1.0 RCAS (`ffx-fsr/ffx_fsr1.h` +
// `ffx-fsr/ffx_a.h`, GPUOpen-Effects/FidelityFX-FSR commit
// `a21ffb8f6c13233ba336352bdff293894c706575`) to plain fp32 GLSL 450.
// Push constant `con.x` is `exp2(-sharpness_stops)` from `rcas_constants()`.
//
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
inline constexpr std::string_view kFsrRcas = R"glsl(#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D src;

layout(push_constant) uniform Push {
    vec4 con;
} pc;

float fsrAprxMedRcp(float a) {
    float b = uintBitsToFloat(0x7ef19fffu - floatBitsToUint(a));
    return b * (-b * a + 2.0);
}

void main() {
    const float rcasLimit = 0.25 - (1.0 / 16.0);
    ivec2 sz = textureSize(src, 0);
    ivec2 mx = max(sz - ivec2(1), ivec2(0));
    ivec2 sp = ivec2(gl_FragCoord.xy);

    vec3 b = texelFetch(src, clamp(sp + ivec2(0, -1), ivec2(0), mx), 0).rgb;
    vec3 d = texelFetch(src, clamp(sp + ivec2(-1, 0), ivec2(0), mx), 0).rgb;
    vec4 ee = texelFetch(src, clamp(sp, ivec2(0), mx), 0);
    vec3 e = ee.rgb;
    vec3 f = texelFetch(src, clamp(sp + ivec2(1, 0), ivec2(0), mx), 0).rgb;
    vec3 h = texelFetch(src, clamp(sp + ivec2(0, 1), ivec2(0), mx), 0).rgb;

    vec3 mn4 = min(min(b, d), min(f, h));
    vec3 mx4 = max(max(b, d), max(f, h));

    // Guard 0/0 on flat black (mx4 == 0) or flat white (mn4 == 1) regions.
    vec3 hitMin = min(mn4, e) / max(4.0 * mx4, vec3(1e-6));
    vec3 hitMax = (1.0 - max(mx4, e)) / min(4.0 * mn4 - 4.0, vec3(-1e-6));
    vec3 lobeRGB = max(-hitMin, hitMax);
    float lobeMax = max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b));
    float lobe = max(-rcasLimit, min(lobeMax, 0.0)) * pc.con.x;

    float rcpL = fsrAprxMedRcp(4.0 * lobe + 1.0);
    vec3 pix = (lobe * b + lobe * d + lobe * h + lobe * f + e) * rcpL;
    fragColor = vec4(pix, ee.a);
}
)glsl";

// Ported from AMD FidelityFX CAS (`ffx-cas/ffx_cas.h` + `ffx-fsr/ffx_a.h`,
// GPUOpen-Effects/FidelityFX-CAS commit
// `9fabcc9a2c45f958aff55ddfda337e74ef894b7f`) sharpen-only path (`noScaling = true`)
// to plain fp32 GLSL 450. Push constant `sharpness` is in [0, 1].
//
// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
inline constexpr std::string_view kCas = R"glsl(#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D src;

layout(push_constant) uniform Push {
    float sharpness;
} pc;

float casAprxLoSqrt(float a) {
    return uintBitsToFloat((floatBitsToUint(a) >> 1u) + 0x1fbc4639u);
}

float casAprxLoRcp(float a) {
    return uintBitsToFloat(0x7ef07ebbu - floatBitsToUint(a));
}

float casAprxMedRcp(float a) {
    float b = uintBitsToFloat(0x7ef19fffu - floatBitsToUint(a));
    return b * (-b * a + 2.0);
}

void main() {
    ivec2 sz = textureSize(src, 0);
    ivec2 mxCoord = max(sz - ivec2(1), ivec2(0));
    ivec2 sp = ivec2(gl_FragCoord.xy);

    vec3 b = texelFetch(src, clamp(sp + ivec2(0, -1), ivec2(0), mxCoord), 0).rgb;
    vec3 d = texelFetch(src, clamp(sp + ivec2(-1, 0), ivec2(0), mxCoord), 0).rgb;
    vec4 ee = texelFetch(src, clamp(sp, ivec2(0), mxCoord), 0);
    vec3 e = ee.rgb;
    vec3 f = texelFetch(src, clamp(sp + ivec2(1, 0), ivec2(0), mxCoord), 0).rgb;
    vec3 h = texelFetch(src, clamp(sp + ivec2(0, 1), ivec2(0), mxCoord), 0).rgb;

    float mnG = min(min(min(d.g, e.g), f.g), min(b.g, h.g));
    float mxG = max(max(max(d.g, e.g), f.g), max(b.g, h.g));

    float rcpMG = casAprxLoRcp(mxG);
    float ampG = clamp(min(mnG, 1.0 - mxG) * rcpMG, 0.0, 1.0);
    ampG = casAprxLoSqrt(ampG);

    float peak = -1.0 / mix(8.0, 5.0, clamp(pc.sharpness, 0.0, 1.0));
    float wG = ampG * peak;
    float rcpWeight = casAprxMedRcp(1.0 + 4.0 * wG);

    vec3 pix = clamp((b * wG + d * wG + f * wG + h * wG + e) * rcpWeight, 0.0, 1.0);
    fragColor = vec4(pix, ee.a);
}
)glsl";

// Original lightweight luma-edge FXAA shader written for wse2-android.
// Computes Rec.601 luma, runs a local contrast check against
// `max(edge_threshold_min, luma_max * edge_threshold)`, determines horizontal vs
// vertical edge orientation from the 3x3 neighborhood, performs a 4-tap edge
// endpoint search, and blends along the perpendicular axis with subpixel
// fallback.
inline constexpr std::string_view kFxaa = R"glsl(#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D src;

layout(push_constant) uniform Push {
    float edge_threshold;
    float edge_threshold_min;
} pc;

float fxaaLuma(vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec2 invSize = 1.0 / vec2(max(textureSize(src, 0), ivec2(1)));
    vec4 centerSample = texture(src, uv);
    float lM = fxaaLuma(centerSample.rgb);
    float lN = fxaaLuma(texture(src, uv + vec2(0.0, -invSize.y)).rgb);
    float lS = fxaaLuma(texture(src, uv + vec2(0.0, invSize.y)).rgb);
    float lW = fxaaLuma(texture(src, uv + vec2(-invSize.x, 0.0)).rgb);
    float lE = fxaaLuma(texture(src, uv + vec2(invSize.x, 0.0)).rgb);

    float lMin = min(lM, min(min(lN, lS), min(lW, lE)));
    float lMax = max(lM, max(max(lN, lS), max(lW, lE)));
    float lRange = lMax - lMin;

    if (lRange < max(pc.edge_threshold_min, lMax * pc.edge_threshold)) {
        fragColor = centerSample;
        return;
    }

    float lNW = fxaaLuma(texture(src, uv + vec2(-invSize.x, -invSize.y)).rgb);
    float lNE = fxaaLuma(texture(src, uv + vec2(invSize.x, -invSize.y)).rgb);
    float lSW = fxaaLuma(texture(src, uv + vec2(-invSize.x, invSize.y)).rgb);
    float lSE = fxaaLuma(texture(src, uv + vec2(invSize.x, invSize.y)).rgb);

    float lAvg = (2.0 * (lN + lS + lW + lE) + (lNW + lNE + lSW + lSE)) * (1.0 / 12.0);
    float subpix = clamp(abs(lAvg - lM) / lRange, 0.0, 1.0);
    subpix = smoothstep(0.0, 1.0, subpix);
    subpix = subpix * subpix * 0.75;

    float edgeH = abs(lNW + lSW - 2.0 * lW) + 2.0 * abs(lN + lS - 2.0 * lM) +
                  abs(lNE + lSE - 2.0 * lE);
    float edgeV = abs(lNW + lNE - 2.0 * lN) + 2.0 * abs(lW + lE - 2.0 * lM) +
                  abs(lSW + lSE - 2.0 * lS);
    bool isHoriz = (edgeH >= edgeV);

    float lNeg = isHoriz ? lN : lW;
    float lPos = isHoriz ? lS : lE;
    float gradNeg = abs(lNeg - lM);
    float gradPos = abs(lPos - lM);
    bool negSteepest = (gradNeg >= gradPos);

    vec2 perpDir = isHoriz ? vec2(0.0, invSize.y) : vec2(invSize.x, 0.0);
    vec2 tangentStep = isHoriz ? vec2(invSize.x, 0.0) : vec2(0.0, invSize.y);
    float signedSign = negSteepest ? -1.0 : 1.0;
    float lEdge = 0.5 * (lM + (negSteepest ? lNeg : lPos));
    float gradLimit = 0.25 * max(gradNeg, gradPos);

    vec2 edgeBase = uv + perpDir * (0.5 * signedSign);
    vec2 uv1 = edgeBase - tangentStep;
    vec2 uv2 = edgeBase + tangentStep;
    float d1 = 1.0;
    float d2 = 1.0;
    float delta1 = fxaaLuma(texture(src, uv1).rgb) - lEdge;
    float delta2 = fxaaLuma(texture(src, uv2).rgb) - lEdge;
    bool done1 = abs(delta1) >= gradLimit;
    bool done2 = abs(delta2) >= gradLimit;

    const float strides[3] = float[3](1.5, 2.0, 4.0);
    for (int i = 0; i < 3; ++i) {
        if (done1 && done2) {
            break;
        }
        if (!done1) {
            uv1 -= tangentStep * strides[i];
            d1 += strides[i];
            delta1 = fxaaLuma(texture(src, uv1).rgb) - lEdge;
            done1 = abs(delta1) >= gradLimit;
        }
        if (!done2) {
            uv2 += tangentStep * strides[i];
            d2 += strides[i];
            delta2 = fxaaLuma(texture(src, uv2).rgb) - lEdge;
            done2 = abs(delta2) >= gradLimit;
        }
    }

    float dMin = min(d1, d2);
    float closestDelta = (d1 <= d2) ? delta1 : delta2;
    bool sameSide = ((lM - lEdge) < 0.0) == (closestDelta < 0.0);
    float edgeBlend = sameSide ? 0.0 : max(0.0, 0.5 - dMin / (d1 + d2));

    float blend = max(edgeBlend, subpix);
    vec2 sampleUv = uv + perpDir * (blend * signedSign);
    fragColor = vec4(texture(src, sampleUv).rgb, centerSample.a);
}
)glsl";

struct EasuConstants {
    float con[16]{};
};

// Computes FSR 1.0 EASU push constants matching AMD's `FsrEasuCon`:
// `con[0..3]` = `con0`, `con[4..7]` = `con1`, `con[8..11]` = `con2`,
// `con[12..15]` = `con3`.
[[nodiscard]] inline EasuConstants easu_constants(float in_w,
                                                  float in_h,
                                                  float in_view_w,
                                                  float in_view_h,
                                                  float out_w,
                                                  float out_h) noexcept {
    EasuConstants out{};
    if (in_w <= 0.0F || in_h <= 0.0F || out_w <= 0.0F || out_h <= 0.0F) {
        return out;
    }
    const float inv_in_w = 1.0F / in_w;
    const float inv_in_h = 1.0F / in_h;
    const float scale_x = in_view_w * (1.0F / out_w);
    const float scale_y = in_view_h * (1.0F / out_h);

    out.con[0] = scale_x;
    out.con[1] = scale_y;
    out.con[2] = 0.5F * scale_x - 0.5F;
    out.con[3] = 0.5F * scale_y - 0.5F;

    out.con[4] = inv_in_w;
    out.con[5] = inv_in_h;
    out.con[6] = 1.0F * inv_in_w;
    out.con[7] = -1.0F * inv_in_h;

    out.con[8] = -1.0F * inv_in_w;
    out.con[9] = 2.0F * inv_in_h;
    out.con[10] = 1.0F * inv_in_w;
    out.con[11] = 2.0F * inv_in_h;

    out.con[12] = 0.0F;
    out.con[13] = 4.0F * inv_in_h;
    out.con[14] = 0.0F;
    out.con[15] = 0.0F;
    return out;
}

namespace postfx_detail {

// Converts IEEE-754 binary32 to binary16 (round-to-nearest-even, clamping
// overflow/Inf/NaN to +-65504 matching AMD `AU1_AH1_AF1` safety behavior).
[[nodiscard]] inline std::uint16_t f32_to_f16(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
    const std::uint32_t exp = (bits >> 23U) & 0xFFU;
    const std::uint32_t mant = bits & 0x7FFFFFU;

    if (exp == 0xFFU || exp > 142U) {
        return static_cast<std::uint16_t>(sign | 0x7BFFU);
    }
    if (exp < 103U) {
        return sign;
    }
    if (exp < 113U) {
        const std::uint32_t full_mant = mant | 0x800000U;
        const std::uint32_t shift = 126U - exp;
        std::uint32_t sub = full_mant >> shift;
        const std::uint32_t rem_mask = (1U << shift) - 1U;
        const std::uint32_t half_way = 1U << (shift - 1U);
        const std::uint32_t rem = full_mant & rem_mask;
        if (rem > half_way || (rem == half_way && (sub & 1U) != 0U)) {
            ++sub;
        }
        return static_cast<std::uint16_t>(sign | sub);
    }
    const std::uint32_t half_exp = exp - 112U;
    std::uint32_t half_mant = mant >> 13U;
    const std::uint32_t round_bits = mant & 0x1FFFU;
    if (round_bits > 0x1000U || (round_bits == 0x1000U && (half_mant & 1U) != 0U)) {
        ++half_mant;
        if (half_mant == 0x400U) {
            half_mant = 0U;
            if (half_exp + 1U >= 31U) {
                return static_cast<std::uint16_t>(sign | 0x7BFFU);
            }
            return static_cast<std::uint16_t>(sign | ((half_exp + 1U) << 10U));
        }
    }
    return static_cast<std::uint16_t>(sign | (half_exp << 10U) | half_mant);
}

}  // namespace postfx_detail

struct RcasConstants {
    float con[4]{};
};

// Computes FSR 1.0 RCAS push constants matching AMD's `FsrRcasCon`:
// `con[0]` = `exp2(-sharpness_stops)`, `con[1]` = bit-cast packed fp16 pair
// `{sharpness, sharpness}`, `con[2] = con[3] = 0.0f`.
[[nodiscard]] inline RcasConstants rcas_constants(float sharpness_stops) noexcept {
    const float stops = std::max(0.0F, sharpness_stops);
    const float linear_sharp = std::exp2(-stops);
    const std::uint16_t half_sharp = postfx_detail::f32_to_f16(linear_sharp);
    const std::uint32_t packed_half2 =
        static_cast<std::uint32_t>(half_sharp) | (static_cast<std::uint32_t>(half_sharp) << 16U);

    RcasConstants out{};
    out.con[0] = linear_sharp;
    out.con[1] = std::bit_cast<float>(packed_half2);
    out.con[2] = 0.0F;
    out.con[3] = 0.0F;
    return out;
}

}  // namespace wse::platform
