#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <c10/cuda/CUDAGuard.h> // support multiple GPUs

#include <algorithm>
#include <stdexcept>

#include <cstdio>

#include "config.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_LONG(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x "must be an long tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

#ifndef M_PIf
#define M_PIf 3.141592653589793f
#endif

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

__device__ __forceinline__ float dot(const float3 & a, const float3 & b){
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 sub(const float3 & a, const float3 & b){
    float3 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;
    return c;
}

__device__ __forceinline__ float3 add(const float3 & a, const float3 & b){
    float3 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    return c;
}

__device__ __forceinline__ float3 multiply(const float3 & a, float k){
    float3 c;
    c.x = a.x * k;
    c.y = a.y * k;
    c.z = a.z * k;
    return c;
}

__device__ __forceinline__ float clamp(float v, float minv, float maxv){
    if(v < minv) v = minv;
    else if (v > maxv) v = maxv;
    return v;
}

template<int n_slot>
class Slot{
public:
    __device__ Slot(){
        slot_ptr = 0;
    }
    __device__ void Insert(int elem){
        for(int i = 0; i < slot_ptr; i++){
            if(slot[i] == elem){
                return;
            }
        }
        if (slot_ptr == n_slot){ // overflow
            //printf("overflow happen!\n");
            //return;
            __trap();
        }
        slot[slot_ptr++] = elem;
    }
public:
    int slot_ptr;
    int slot[n_slot];
};

/*
__device__ float3 closestPointOnTriangle(
    const float3 & x0,
    const float3 & x1,
    const float3 & x2,
    const float3 & sourcePosition,
    float * oS, float * oT
){
    float3 edge0 = sub(x1,x0);
    float3 edge1 = sub(x2,x0);
    float3 v0 = sub(x0,sourcePosition);

    float a = dot(edge0, edge0);
    float b = dot(edge0, edge1);
    float c = dot(edge1, edge1);
    float d = dot(edge0, v0);
    float e = dot(edge1, v0);

    float det = a * c - b * b;
    float s = b * e - c * d;
    float t = b * d - a * e;

    if (s + t < det)
    {
        if (s < 0.f)
        {
            if (t < 0.f)
            {
                if (d < 0.f)
                {
                    s = clamp(-d / a, 0.f, 1.f);
                    t = 0.f;
                }
                else
                {
                    s = 0.f;
                    t = clamp(-e / c, 0.f, 1.f);
                }
            }
            else
            {
                s = 0.f;
                t = clamp(-e / c, 0.f, 1.f);
            }
        }
        else if (t < 0.f)
        {
            s = clamp(-d / a, 0.f, 1.f);
            t = 0.f;
        }
        else
        {
            float invDet = 1.f / det;
            s *= invDet;
            t *= invDet;
        }
    }
    else
    {
        if (s < 0.f)
        {
            float tmp0 = b + d;
            float tmp1 = c + e;
            if (tmp1 > tmp0)
            {
                float numer = tmp1 - tmp0;
                float denom = a - 2 * b + c;
                s = clamp(numer / denom, 0.f, 1.f);
                t = 1 - s;
            }
            else
            {
                t = clamp(-e / c, 0.f, 1.f);
                s = 0.f;
            }
        }
        else if (t < 0.f)
        {
            if (a + d > b + e)
            {
                float numer = c + e - b - d;
                float denom = a - 2 * b + c;
                s = clamp(numer / denom, 0.f, 1.f);
                t = 1.f - s;
            }
            else
            {
                //s = clamp(-e / c, 0.f, 1.f); // original
                s = clamp(-d / a, 0.f, 1.f); // bugfix
                t = 0.f;
            }
        }
        else
        {
            float numer = c + e - b - d;
            float denom = a - 2 * b + c;
            s = clamp(numer / denom, 0.f, 1.f);
            t = 1.f - s;
        }
    }

    *oS = s;
    *oT = t;
    //return x0 + s * edge0 + t * edge1;
    return make_float3(
        x0.x + s * edge0.x + t * edge1.x,
        x0.y + s * edge0.y + t * edge1.y,
        x0.z + s * edge0.z + t * edge1.z
    );
}
*/

__device__ float3 closestPointOnTriangle_backward(
    const float3 & x0,
    const float3 & x1,
    const float3 & x2,
    const float3 & sourcePosition,
    float grad_s, float grad_t
){
    float3 edge0 = sub(x1,x0);
    float3 edge1 = sub(x2,x0);
    float3 v0 = sub(x0,sourcePosition);

    float a = dot(edge0, edge0);
    float b = dot(edge0, edge1);
    float c = dot(edge1, edge1);
    float d = dot(edge0, v0);
    float e = dot(edge1, v0);

    float det = a * c - b * b;
    float s = b * e - c * d;
    float t = b * d - a * e;

    int type = 0;
    if (s + t < det){
        if(s < 0.f){
            if (t < 0.f){
                if (d < 0.f){
                    type = 1; // s
                }
                else{
                    type = 2; // t
                }
            }
            else{
                type = 2; // t
            }
        }
        else if ( t < 0.f){
            type = 1; // s
        }
        //else{
        //    type = 0;
        //}
    }
    else{
        if (s < 0.f){
            if( c + e > b + d){
                type = 3; // st
            }
            else{
                type = 2; // t
            }
        }
        else if ( t < 0.f){
            if ( a + d > b + e){
                type = 3; // st
            }
            else{
                type = 1; // s
            }
        }
        else{
            type = 3; // st
        }
    }

    // compute according to type
    float3 grad_pos = make_float3(0.f,0.f,0.f);
    if (type == 0){
        float invDet = 1.f / det;
        grad_pos = add(
            multiply(edge0, (grad_s * c - grad_t * b) * invDet),
            multiply(edge1, (grad_t * a - grad_s * b) * invDet)
        );
    }
    else if (type == 1){
        float tmp = -d / a;
        if(tmp >= 0.f && tmp <= 1.f){
            grad_pos = multiply(edge0, grad_s / a);
        }
    }
    else if (type == 2){
        float tmp = -e / c;
        if(tmp >= 0.f && tmp <= 1.f){
            grad_pos = multiply(edge1, grad_t / c);
        }
    }
    else{ // type 3
        float numer = c + e - b - d;
        float denom = a - 2 * b + c;
        float tmp = numer / denom;
        if(tmp >= 0.f && tmp <= 1.f){
            grad_pos = multiply(sub(edge0, edge1), (grad_s - grad_t) / denom);
        }
    }
    return grad_pos;
}

__device__ float3 closestPointOnTriangle(
    const float3 & x0,
    const float3 & x1,
    const float3 & x2,
    const float3 & sourcePosition,
    float * oS, float * oT
){
    float3 edge0 = sub(x1,x0);
    float3 edge1 = sub(x2,x0);
    float3 v0 = sub(x0,sourcePosition);

    float a = dot(edge0, edge0);
    float b = dot(edge0, edge1);
    float c = dot(edge1, edge1);
    float d = dot(edge0, v0);
    float e = dot(edge1, v0);

    float det = a * c - b * b;
    float s = b * e - c * d;
    float t = b * d - a * e;

    int type = 0;
    if (s + t < det){
        if(s < 0.f){
            if (t < 0.f){
                if (d < 0.f){
                    type = 1; // s
                }
                else{
                    type = 2; // t
                }
            }
            else{
                type = 2; // t
            }
        }
        else if ( t < 0.f){
            type = 1; // s
        }
        //else{
        //    type = 0;
        //}
    }
    else{
        if (s < 0.f){
            if( c + e > b + d){
                type = 3; // st
            }
            else{
                type = 2; // t
            }
        }
        else if ( t < 0.f){
            if ( a + d > b + e){
                type = 3; // st
            }
            else{
                type = 1; // s
            }
        }
        else{
            type = 3; // st
        }
    }

    // compute according to type
    if(type == 0){
        float invDet = 1.f / det;
        s *= invDet;
        t *= invDet;
    }
    else if (type == 1){
        s = clamp(-d / a, 0.f, 1.f);
        t = 0.f;
    }
    else if (type == 2){
        s = 0.f;
        t = clamp(-e / c, 0.f, 1.f);
    }
    else{ // type == 3
        float numer = c + e - b - d;
        float denom = a - 2 * b + c;
        s = clamp(numer / denom, 0.f, 1.f);
        t = 1.f - s;
    }

    *oS = s;
    *oT = t;
    return make_float3(
        x0.x + s * edge0.x + t * edge1.x,
        x0.y + s * edge0.y + t * edge1.y,
        x0.z + s * edge0.z + t * edge1.z
    );
}

template<int B, int C>
__device__ __forceinline__ void sload(
    int64_t P, bool valid, const float * g_data, float * s_data, float * l_data
){
    #pragma unroll
    for(int i = 0; i < C; i++){
        int loc_offset = i * B + threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(B * C) + loc_offset;
        if(i_elem < P * C){
            s_data[loc_offset] = g_data[i_elem];
        }
    }
    __syncthreads();
    if(valid){
        #pragma unroll
        for(int i = 0; i < C; i++){
            l_data[i] = s_data[threadIdx.x * C + i];
        }
    }
}

template<int B, int C>
__device__ __forceinline__ void ssave(
    int64_t P, bool valid, float * g_data, float * s_data, const float * l_data
){
    if(valid){
        #pragma unroll
        for(int i = 0; i < C; i++){
            s_data[threadIdx.x * C + i] = l_data[i];
        }
    }
    __syncthreads();
    #pragma unroll
    for(int i = 0; i < C; i++){
        int loc_offset = i * B + threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(B * C) + loc_offset;
        if(i_elem < P * C){
            g_data[i_elem] = s_data[loc_offset];
        }
    }
}

template<typename T>
__device__ T length3(const T* x) {
    return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

template<typename T>
__device__ T dot3(const T* x,const T*y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

template<typename T>
__device__ void cross(const T* x, const T* y, T* z) {
    z[0] = x[1] * y[2] - x[2] * y[1];
    z[1] = x[2] * y[0] - x[0] * y[2];
    z[2] = x[0] * y[1] - x[1] * y[0];
}

////////////////////////////////////////////////////////

__global__ void GenerateCameraRayKernel_Forward(
    int H, int W, bool flipY, bool normalize, float z_sign,
    const float * __restrict__ proj, // 4x4, row major
    const float * __restrict__ c2w, // 4x4, row major
    float * __restrict__ rays // 3xHxW
){
    __shared__ float s_buff[32];
    if(c2w){
        //const float * ptr_src;
        if(threadIdx.x < 16) // warp 0
            s_buff[threadIdx.x] = proj[threadIdx.x];
        else if (threadIdx.x >= 32 && threadIdx.x < 32 + 16) // warp 1
            s_buff[threadIdx.x - 16] = c2w[threadIdx.x - 32];
    }
    else{
        if(threadIdx.x < 16){
            s_buff[threadIdx.x] = proj[threadIdx.x];
        }
    }
    __syncthreads();

    float out_dir[3];
    float l_mat[16];
    for(int i = 0; i < 16; i++){
        l_mat[i] = s_buff[i];
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= W * H) return;

    int iy = idx / W;
    int ix = idx % W;
    if(flipY) iy =  H - 1 - iy;
    float u = (ix + 0.5f) / W * 2.f - 1.f; // map to [-1,1]
    float v = (iy + 0.5f) / H * 2.f - 1.f;

    //float m = u - l_mat[2];
    //float n = v - l_mat[6];
    float m = u - z_sign * l_mat[2];
    float n = v - z_sign * l_mat[6];
    float denom = l_mat[0] * l_mat[5] - l_mat[1] * l_mat[4];
    float yrate = (n * l_mat[0] - m * l_mat[4]) / denom;
    float xrate = (m * l_mat[5] - n * l_mat[1]) / denom;
    //out_dir[0] = xrate * z_sign;
    //out_dir[1] = yrate * z_sign;
    out_dir[0] = xrate;
    out_dir[1] = yrate;
    out_dir[2] = z_sign;

    if(c2w){
        float p[3] = {out_dir[0], out_dir[1], out_dir[2]};
        for(int i = 0; i < 16; i++){
            l_mat[i] = s_buff[16 + i];
        }
        out_dir[0] = l_mat[0] * p[0] + l_mat[1] * p[1] + l_mat[2] * p[2];
        out_dir[1] = l_mat[4] * p[0] + l_mat[5] * p[1] + l_mat[6] * p[2];
        out_dir[2] = l_mat[8] * p[0] + l_mat[9] * p[1] + l_mat[10] * p[2];
    }
    if(normalize){
        float L = sqrt(out_dir[0] * out_dir[0] + out_dir[1] * out_dir[1] + out_dir[2] * out_dir[2]);
        out_dir[0] = out_dir[0] / L;
        out_dir[1] = out_dir[1] / L;
        out_dir[2] = out_dir[2] / L;
    }
    rays[0 * H * W + idx] = out_dir[0];
    rays[1 * H * W + idx] = out_dir[1];
    rays[2 * H * W + idx] = out_dir[2];
}

__device__ __inline__ void SampleEnvmap(
    float x, float y, float z,
    int H, int W, int C,
    int64_t stride, int64_t offset,
    const float * __restrict__ envmap,
    float * __restrict__ output
){
    float phi = atan2(z,x);
    if (phi < 0.f) phi += 2 * M_PIf;

    if(y < -1.f) y = -1.f;
    else if (y > 1.f) y = 1.f;
    float theta = asin(y);

    float u,v;
    u = phi / (2 * M_PIf);
    v = theta / M_PIf + 0.5f;
#ifdef LEFT_TOP_AS_ORIGIN
    v = 1.f - v;
#endif
    float p_u = u * W;
    float p_v = v * H;
    int ux_0 = int(floor(p_u - 0.5f));
    int uy_0 = int(floor(p_v - 0.5f));

    int ux_1 = ux_0 + 1;
    int uy_1 = uy_0 + 1;
    float kx = p_u - float(ux_0) - 0.5f;
    float ky = p_v - float(uy_0) - 0.5f;
#ifdef FIX_ENVMAP_SEAM
    if(ux_0 < 0) ux_0 = ux_0 + W;
    if(ux_0 >= W) ux_0 = ux_0 - W;
    if(ux_1 < 0) ux_1 = ux_1 + W;
    if(ux_1 >= W) ux_1 = ux_1 - W;
#else
    if(ux_0 < 0) ux_0 = 0;
    if(ux_0 >= W) ux_0 = W - 1;
    if(ux_1 < 0) ux_1 = 0;
    if(ux_1 >= W) ux_1 = W - 1;
#endif

    if(uy_0 < 0) uy_0 = 0;
    if(uy_0 >= H) uy_0 = H - 1;
    if(uy_1 < 0) uy_1 = 0;
    if(uy_1 >= H) uy_1 = H - 1;

    for(int iC = 0; iC < C; iC++){
        float v00 = envmap[(iC * H + uy_0) * W + ux_0];
        float v01 = envmap[(iC * H + uy_0) * W + ux_1];
        float v10 = envmap[(iC * H + uy_1) * W + ux_0];
        float v11 = envmap[(iC * H + uy_1) * W + ux_1];
        float o = (1-ky)*((1-kx) * v00 + kx * v01) + ky *((1-kx) * v10 + kx * v11);
        output[iC * stride + offset] = o;
    }
}

__device__ __inline__ void TrowbridgeReitzDistributionSample(
    const float * __restrict__ rnd2,
    float alpha,
    float * __restrict__ out3
){
    float cosTheta, phi;
    phi = (2 * M_PIf) * rnd2[1];
    float tanTheta2 = alpha * alpha * rnd2[0] / (1.0f - rnd2[0]);
    cosTheta = 1 / sqrt(1 + tanTheta2);
    float tmp = 1.f - cosTheta * cosTheta;
	if (tmp < 0.f) tmp = 0.f;
	float sinTheta = sqrt(tmp);
	float sin_phi = sin(phi);
	float cos_phi = cos(phi);
	out3[0] = sinTheta * cos_phi;
	out3[1] = sinTheta * sin_phi;
	out3[2] = cosTheta;
}

template<int C>
__global__ void BlurEnvmap_Forward(
    int H, int W, int tar_H, int tar_W, int N_samples, float alpha,
    const float * __restrict__ envmap, // CxHxW
    float * __restrict__ output // Cxtar_Hxtar_W
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int P = tar_H * tar_W;
    if(idx >= P) return;
    int ih = idx / tar_W;
    int iw = idx % tar_W;
    float v = (ih + 0.5f) / float(tar_H);
#ifndef LEFT_TOP_AS_ORIGIN
    v = 1.f - v;
#endif
    float theta = v * M_PIf; // [0,pi]
    //
    float u = (iw + 0.5f) / float(tar_W);
    float phi = u * 2 * M_PIf; // [0,2 * pi]
    float in3[3];
    float sin_theta, cos_theta;
    sincosf(theta, &sin_theta, &cos_theta);
    float sin_phi, cos_phi;
    sincosf(phi, &sin_phi, &cos_phi);
    //in3[0] = sin_theta * cos_phi;
    //in3[1] = sin_theta * sin_phi;
    //in3[2] = cos_theta;
    in3[0] = sin_theta * cos_phi;
    in3[1] = cos_theta;
    in3[2] = sin_theta * sin_phi;
    float L = length3(in3);
    in3[0] /= L; in3[1] /= L; in3[2] /= L;

    ///// Init
    float data[C];
    float data_add[C];
    for(int c = 0; c < C; c++) data[c] = 0.f;

    float local_coord[3];
    float sample_dir[3];
    float v_x[3], v_y[3];

    curandState state;
    const int seed = 0;
    curand_init(seed, idx, 0, &state);

    if (abs(in3[0]) > 0.9f){
        v_x[0] = in3[2]; v_x[1] = 0.f; v_x[2] = -in3[0];
    }else{
        v_x[0] = 0.f; v_x[1] = -in3[2]; v_x[2] = in3[1];
    }
    L = length3(v_x);
    v_x[0] /= L; v_x[1] /= L; v_x[2] /= L;
    cross(in3, v_x, v_y);
    L = length3(v_y);
    v_y[0] /= L; v_y[1] /= L; v_y[2] /= L;

    ///// Blur
    for (int i = 0; i < N_samples; i++){
        float rnd[2];
        rnd[0] = curand_uniform(&state);
        rnd[1] = curand_uniform(&state);
        TrowbridgeReitzDistributionSample(rnd, alpha, local_coord);
#pragma unroll
        for(int j = 0; j < 3; j++){
            sample_dir[j] = v_x[j] * local_coord[0] + v_y[j] * local_coord[1] + in3[j] * local_coord[2];
        }
        L = length3(sample_dir);
        sample_dir[0] /= L; sample_dir[1] /= L; sample_dir[2] /= L;
        SampleEnvmap(
            sample_dir[0], sample_dir[1], sample_dir[2],
            H, W, C, 1, 0,
            envmap, data_add
        );
#pragma unroll
        for(int c = 0; c < C; c++){
            data[c] += data_add[c];
        }
    }
#pragma unroll
    for(int c = 0; c < C; c++){
        output[c * P + idx] = data[c] / float(N_samples);
    }
}

__global__ void SampleEnvmap_Forward(
    int64_t P, int H, int W, int C,
    const float * __restrict__ query_point, //3xP
    const float * __restrict__ envmap, // CxHxW
    float * __restrict__ output // CxP
){
    int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= P) return;
    float x,y,z;
    x = query_point[0 * P + idx];
    y = query_point[1 * P + idx];
    z = query_point[2 * P + idx];
    float L = sqrt(x * x + y * y + z * z);
    x = x / L;
    y = y / L;
    z = z / L;
    //
    SampleEnvmap(x,y,z, H, W, C, P, idx, envmap, output);
}

__global__ void __launch_bounds__(256) GetUVofTriangle_Forward(
    int P,
    const float * __restrict__ triangles, // Px3(vertex)x3(xyz)
    const float * __restrict__ query_pos, // Px3
    float * __restrict__ barycentric_coord // Px3
){
    __shared__ float s_buff[9 * 256];
    float triangle[9];
    float pos[3];
    float coord[3];

    bool valid = (blockIdx.x * int64_t(256) + threadIdx.x  < P);

    sload<256,9>(P, valid, triangles, s_buff, triangle);
    __syncthreads();
    sload<256,3>(P, valid, query_pos, s_buff, pos);
    if(valid){
        float3 vertex0 = make_float3(triangle[0], triangle[1], triangle[2]);
        float3 vertex1 = make_float3(triangle[3], triangle[4], triangle[5]);
        float3 vertex2 = make_float3(triangle[6], triangle[7], triangle[8]);
        float3 l_query_pos = make_float3(pos[0], pos[1], pos[2]);
        float s, t;
        float3 close_p = closestPointOnTriangle(
            vertex0, vertex1, vertex2, l_query_pos,
            &s, &t
        );
        coord[0] = 1.f - s - t;
        coord[1] = s;
        coord[2] = t;
    }
    __syncthreads();
    ssave<256,3>(P, valid, barycentric_coord, s_buff, coord);
}

__global__ void __launch_bounds__(256) GetUVofTriangle_Backward(
    int P,
    const float * __restrict__ triangles, // Px3(vertex)x3(xyz)
    const float * __restrict__ query_pos, // Px3
    const float * __restrict__ grad_barycentric_coord, // Px3
    float * __restrict__ grad_query_pos // Px3
){
    __shared__ float s_buff[9 * 256];
    float triangle[9];
    float pos[3];
    float gcoord[3];
    float gpos[3];

    bool valid = (blockIdx.x * int64_t(256) + threadIdx.x  < P);

    sload<256,9>(P, valid, triangles, s_buff, triangle);
    __syncthreads();
    sload<256,3>(P, valid, query_pos, s_buff, pos);
    __syncthreads();
    sload<256,3>(P, valid, grad_barycentric_coord, s_buff, gcoord);
    if(valid){
        float3 vertex0 = make_float3(triangle[0], triangle[1], triangle[2]);
        float3 vertex1 = make_float3(triangle[3], triangle[4], triangle[5]);
        float3 vertex2 = make_float3(triangle[6], triangle[7], triangle[8]);
        float3 l_query_pos = make_float3(pos[0], pos[1], pos[2]);
        float grad_s = gcoord[1] - gcoord[0];
        float grad_t = gcoord[2] - gcoord[0];

        float3 out_grad = closestPointOnTriangle_backward(
            vertex0, vertex1, vertex2,
            l_query_pos, grad_s, grad_t
        );
        gpos[0] = out_grad.x; gpos[1] = out_grad.y; gpos[2] = out_grad.z;
    }
    __syncthreads();
    ssave<256,3>(P, valid, grad_query_pos, s_buff, gpos);
}

template<int N_slot>
__global__ void __launch_bounds__(256) GetNearestMeshPoints_Forward(
    int N_vertex, int N_face, int P, int K,
    const int * __restrict__ adjacency_head, // 2 x N_vertex
    const int * __restrict__ adjacency_list, // N_tbd
    const int * __restrict__ face_vertex_list, // 3 x N_face
    const float * __restrict__ vertex_pos, // 3 x N_vertex
    const float * __restrict__ query_pos, // 3xP
    const int64_t * __restrict__ idxs, // PxK
    int64_t * __restrict__ nearest_triangle_id, // P
    float * __restrict__ barycentric_coord // 3xP
){
    int index = blockIdx.x * int64_t(256) + threadIdx.x;
    if (index >= P) return;

    Slot<N_slot> slot;
    for(int k = 0; k < K; k++){
        int idx = (int)idxs[index * K + k]; // nearest vertex
        if(idx >= 0){
            // find triangle face adjacent to the vertex
            int n = adjacency_head[idx];
            int prefix_sum = adjacency_head[N_vertex + idx];
            int offset = prefix_sum - n;
            for(int j = 0; j < n; j++){
                int face_id = adjacency_list[offset + j];
                slot.Insert(face_id);
            }
        }
    }

    float3 l_query_pos = make_float3(
        query_pos[0 * P + index],
        query_pos[1 * P + index],
        query_pos[2 * P + index]
    );
    float closest_dist = FLT_MAX;
    int sel_face = -1;
    float sel_s, sel_t;
    for(int j=0; j < slot.slot_ptr; j++){
        int face_id = slot.slot[j];
        int vertex_id0 = face_vertex_list[0 * N_face + face_id];
        int vertex_id1 = face_vertex_list[1 * N_face + face_id];
        int vertex_id2 = face_vertex_list[2 * N_face + face_id];
        float3 vertex0 = make_float3(
            vertex_pos[0 * N_vertex + vertex_id0],
            vertex_pos[1 * N_vertex + vertex_id0],
            vertex_pos[2 * N_vertex + vertex_id0]
        );
        float3 vertex1 = make_float3(
            vertex_pos[0 * N_vertex + vertex_id1],
            vertex_pos[1 * N_vertex + vertex_id1],
            vertex_pos[2 * N_vertex + vertex_id1]
        );
        float3 vertex2 = make_float3(
            vertex_pos[0 * N_vertex + vertex_id2],
            vertex_pos[1 * N_vertex + vertex_id2],
            vertex_pos[2 * N_vertex + vertex_id2]
        );
        float s,t;
        float3 close_p = closestPointOnTriangle(
            vertex0, vertex1, vertex2, l_query_pos,
            &s, &t
        );
        float3 diff = sub(close_p, l_query_pos);
        float dist2 = dot(diff, diff);
        if (dist2 < closest_dist){
            sel_face = face_id;
            sel_s = s; sel_t = t;
            closest_dist = dist2;
        }
    }

    if (sel_face == -1){ // Not even one triangle found, we should report it !
        __trap();
        //printf("assertion fail at %d\n" , index);
    }

    nearest_triangle_id[index] = (int64_t)(sel_face);
    //barycentric_coord[0 * P + index] = sel_s;
    //barycentric_coord[1 * P + index] = sel_t;
    //barycentric_coord[2 * P + index] = 1.f - sel_s - sel_t;

    // BUGFIX
    barycentric_coord[0 * P + index] = 1.f - sel_s - sel_t;
    barycentric_coord[1 * P + index] = sel_s;
    barycentric_coord[2 * P + index] = sel_t;
}

__global__ void __launch_bounds__(256) GetNearestMeshPoints_Backward(
    int N_vertex, int N_face, int P,
    const int * __restrict__ face_vertex_list, // 3 x N_face
    const float * __restrict__ vertex_pos, // 3 x N_vertex
    const float * __restrict__ query_pos, // 3xP
    const int64_t * __restrict__ nearest_triangle_id,
    const float * __restrict__ grad_barycentric_coord,
    float * __restrict__ grad_query_pos // 3xP
){
    int index = blockIdx.x * int64_t(256) + threadIdx.x;
    if (index >= P) return;

    float3 l_query_pos = make_float3(
        query_pos[0 * P + index],
        query_pos[1 * P + index],
        query_pos[2 * P + index]
    );
    float3 l_grad_bc = make_float3(
        grad_query_pos[0 * P + index],
        grad_query_pos[1 * P + index],
        grad_query_pos[2 * P + index]
    );
    float grad_s = l_grad_bc.y - l_grad_bc.x;
    float grad_t = l_grad_bc.z - l_grad_bc.x;
    int face_id = (int)nearest_triangle_id[index];
    int vertex_id0 = face_vertex_list[0 * N_face + face_id];
    int vertex_id1 = face_vertex_list[1 * N_face + face_id];
    int vertex_id2 = face_vertex_list[2 * N_face + face_id];
    float3 vertex0 = make_float3(
        vertex_pos[0 * N_vertex + vertex_id0],
        vertex_pos[1 * N_vertex + vertex_id0],
        vertex_pos[2 * N_vertex + vertex_id0]
    );
    float3 vertex1 = make_float3(
        vertex_pos[0 * N_vertex + vertex_id1],
        vertex_pos[1 * N_vertex + vertex_id1],
        vertex_pos[2 * N_vertex + vertex_id1]
    );
    float3 vertex2 = make_float3(
        vertex_pos[0 * N_vertex + vertex_id2],
        vertex_pos[1 * N_vertex + vertex_id2],
        vertex_pos[2 * N_vertex + vertex_id2]
    );

    float3 out_grad = closestPointOnTriangle_backward(
        vertex0, vertex1, vertex2,
        l_query_pos, grad_s, grad_t
    );

    // write gradient of query_pos
    grad_query_pos[0 * P + index] = out_grad.x;
    grad_query_pos[1 * P + index] = out_grad.y;
    grad_query_pos[2 * P + index] = out_grad.z;
}

template<typename T>
__device__ void rotation_6d_to_matrix(
    const T * inputs, T * outputs
){
    T L = length3(inputs);
    T norm_x[3] = {inputs[0]/L, inputs[1]/L, inputs[2]/L};
    T dot_x_y = dot3(norm_x, inputs + 3);
    T norm_y[3] = {
        inputs[3] - dot_x_y * norm_x[0],
        inputs[4] - dot_x_y * norm_x[1],
        inputs[5] - dot_x_y * norm_x[2]
    };
    L = length3(norm_y);
    norm_y[0] /= L; norm_y[1] /= L; norm_y[2] /= L;
    T z[3];
    cross(norm_x, norm_y, z);
    outputs[0] = norm_x[0]; outputs[1] = norm_x[1]; outputs[2] = norm_x[2];
	outputs[3] = norm_y[0]; outputs[4] = norm_y[1]; outputs[5] = norm_y[2];
	outputs[6] = z[0]; outputs[7] = z[1]; outputs[8] = z[2];
}

template<typename T>
__device__ void log_so3(
    const T* inputs, T * outputs, T eps = 1e-4
) {
    T rot_trace = inputs[0] + inputs[4] + inputs[8];
    T phi_cos = (rot_trace - 1) * 0.5;
    if (phi_cos < -1) phi_cos = -1;
    if (phi_cos > 1) phi_cos = 1;
    T phi = acos(phi_cos);
    T phi_sin = sin(phi);
    T phi_factor;
    if (abs(phi_sin) > 0.5 * eps) {
        phi_factor = phi / (2 * phi_sin);
    }
    else { // avoid div tiny number
        phi_factor = 0.5 + (phi * phi) / 12;
    }
    outputs[0] = phi_factor * (inputs[7] - inputs[5]);
    outputs[1] = phi_factor * (inputs[2] - inputs[6]);
    outputs[2] = phi_factor * (inputs[3] - inputs[1]);
}

__global__ void __launch_bounds__(64) MTFormatConversion_Forward(
    int N,
    const float * __restrict__ jaw, // Nx6
    const float * __restrict__ eyes, // Nx12
    float * __restrict__ output // Nx15
){
    const int M = 3;
    __shared__ float sslot[54]; // (6+12) * 3
    bool flag = false;
    const float * ptr;
    if(threadIdx.x < M * 6){
        int offset = blockIdx.x * M * 6 + threadIdx.x;
        ptr = jaw + offset;
        if(offset < N * 6) flag = true;
    }
    else if (threadIdx.x < M * 18){
        int offset = blockIdx.x * M * 12 + (threadIdx.x - M * 6);
        ptr = eyes + offset;
        if(offset < N * 12) flag = true;
    }
    if(flag){
        sslot[threadIdx.x] = *ptr;
    }

    float output_[3];
    if(threadIdx.x < M * 3){
        float input[6];
        float mid[9];
        for(int j = 0; j < 6; j++){
            input[j] = sslot[threadIdx.x * 6 + j];
        }
        rotation_6d_to_matrix(input,mid);
        log_so3(mid,output_);
    }
    __syncthreads();
    if(threadIdx.x < M * 3){
        sslot[threadIdx.x * 3 + 0] = output_[0];
        sslot[threadIdx.x * 3 + 1] = output_[1];
        sslot[threadIdx.x * 3 + 2] = output_[2];
    }
    __syncthreads();
    int out_idx = blockIdx.x * M * 15 + threadIdx.x;
    if(threadIdx.x < M * 15 && out_idx < N * 15){
        float out_elem = 0.f;
        int i = threadIdx.x / 15;
        int j = threadIdx.x % 15;
        if(j >= 6){
            int offset;
            if(j >= 9){ offset = M * 3 + 6 * i + (j - 9); }
            else { offset = 3 * i + (j - 6);}
            out_elem = sslot[offset];
        }
        output[out_idx] = out_elem;
    }
}

__global__ void ExtractBitfield_Forward(
    int N, int N_bit, int i_start, int i_end,
    const unsigned int * __restrict__ bitfield, // N_slot x N
    float * __restrict__ output // B x N_bit
){
    int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int i_batch = (int)(idx / N_bit);
    int i_bit = (int)(idx % N_bit);
    int i_slot = i_bit / 32;
    int i_loc_bit = i_bit % 32;
    int i = i_start + i_batch;
    if(i >= i_end) return;
    unsigned int p0 = bitfield[i_slot * N + i];
    float tmp;
    if(p0 & (0x01 << i_loc_bit))
        tmp = 1.f;
    else
        tmp = 0.f;
    output[i_batch * N_bit + i_bit] = tmp;
}

//----------------------------------------

torch::Tensor extract_bitfield_forward(
    const torch::Tensor & bitfield,
    int N_bit, int i_start, int i_end
){
    CHECK_CUDA(bitfield); CHECK_CONTIGUOUS(bitfield); CHECK_IS_INT(bitfield);

    //const int N_slot = bitfield.size(0);
    const int N = bitfield.size(1);
    const int n_batch = i_end - i_start;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(bitfield));
    auto device = bitfield.device();

    at::TensorOptions opt(at::kFloat); opt = opt.device(device); opt = opt.requires_grad(false);
    torch::Tensor outputs = torch::empty({n_batch, N_bit}, opt);

    const uint32_t THREADS = 128;
    const uint32_t BLOCKS = (uint32_t)div_round_up((int64_t)n_batch * N_bit, (int64_t)THREADS);

    if(BLOCKS){
        ExtractBitfield_Forward<<<BLOCKS,THREADS>>>(
            N, N_bit, i_start, i_end,
            (unsigned int *)bitfield.data_ptr<int>(),
            outputs.data_ptr<float>()
        );
    }
    return outputs;
}

torch::Tensor mt_format_conversion_forward(
    const torch::Tensor & jaw,
    const torch::Tensor & eyes
){
    CHECK_CUDA(jaw); CHECK_CONTIGUOUS(jaw); CHECK_IS_FLOATING(jaw);
    CHECK_CUDA(eyes); CHECK_CONTIGUOUS(eyes); CHECK_IS_FLOATING(eyes);

    // TODO: shape check
    const int M = 3;
    int64_t N = jaw.size(0);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(jaw));
    auto device = jaw.device();

    at::TensorOptions opt(jaw.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor outputs = torch::empty({N,15}, opt);

    const uint32_t THREADS = 64;
    const uint32_t BLOCKS = (uint32_t)div_round_up(N, (int64_t)M);

    if(BLOCKS){
        MTFormatConversion_Forward<<<BLOCKS,THREADS>>>(
            N, jaw.data_ptr<float>(), eyes.data_ptr<float>(),
            outputs.data_ptr<float>()
        );
    }
    return outputs;
}

torch::Tensor blur_envmap_forward(
    const torch::Tensor & envmap,
    int tar_H, int tar_W, int N_samples, float alpha
){
    CHECK_CUDA(envmap); CHECK_CONTIGUOUS(envmap); CHECK_IS_FLOATING(envmap);

    int64_t C = envmap.size(0);
    int64_t H = envmap.size(1);
    int64_t W = envmap.size(2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(envmap));
    auto device = envmap.device();

    at::TensorOptions opt(envmap.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor sampled_envmap = torch::empty({C,tar_H, tar_W}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up((int64_t)tar_H * tar_W, (int64_t)256);

    if (BLOCKS){
        if (C == 3){
            BlurEnvmap_Forward<3><<<BLOCKS, THREADS>>>(
                H, W, tar_H, tar_W, N_samples, alpha,
                envmap.data_ptr<float>(),
                sampled_envmap.data_ptr<float>()
            );
        }
        else if (C == 4){
            BlurEnvmap_Forward<4><<<BLOCKS, THREADS>>>(
                H, W, tar_H, tar_W, N_samples, alpha,
                envmap.data_ptr<float>(),
                sampled_envmap.data_ptr<float>()
            );
        }
        else{
            throw std::runtime_error("Channel number not supported!");
        }
    }
    return sampled_envmap;
}

torch::Tensor sample_envmap_forward(
    const torch::Tensor & query_point,
    const torch::Tensor & envmap
){
    CHECK_CUDA(query_point); CHECK_CONTIGUOUS(query_point); CHECK_IS_FLOATING(query_point);
    CHECK_CUDA(envmap); CHECK_CONTIGUOUS(envmap); CHECK_IS_FLOATING(envmap);

    // TODO: shape check

    int64_t P = query_point.size(1);
    int64_t C = envmap.size(0);
    int64_t H = envmap.size(1);
    int64_t W = envmap.size(2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(envmap));
    auto device = envmap.device();

    at::TensorOptions opt(envmap.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor features = torch::empty({3,P}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);

    if(BLOCKS){
        SampleEnvmap_Forward<<<BLOCKS,THREADS>>>(
            P, H, W, C,
            query_point.data_ptr<float>(),
            envmap.data_ptr<float>(),
            features.data_ptr<float>()
        );
    }
    return features;
}

torch::Tensor generate_camera_ray_forward(
    const torch::Tensor & proj,
    at::optional<at::Tensor> & c2w,
    int H, int W, bool flipY, bool normalize, float z_sign
){
    CHECK_CUDA(proj); CHECK_CONTIGUOUS(proj); CHECK_IS_FLOATING(proj);
    const float * ptr_c2w = nullptr;
    bool use_c2w = c2w.has_value();

    if(use_c2w){
        at::Tensor tmp = c2w.value();
        CHECK_CUDA(tmp); CHECK_CONTIGUOUS(tmp); CHECK_IS_FLOATING(tmp);
        ptr_c2w = tmp.data_ptr<float>();
    }

    // TODO: shape check

    const at::cuda::OptionalCUDAGuard device_guard(device_of(proj));
    auto device = proj.device();

    at::TensorOptions opt(proj.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor rays = torch::empty({3,H,W}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(H * W, 256);

    if(BLOCKS){
        GenerateCameraRayKernel_Forward<<<BLOCKS,THREADS>>>(
            H, W, flipY, normalize, z_sign,
            proj.data_ptr<float>(),
            ptr_c2w,
            rays.data_ptr<float>()
        );
    }
    return rays;
}

torch::Tensor get_uv_of_triangle_forward(
    const torch::Tensor & triangles, // Px3x3
    const torch::Tensor & query_pos // Px3
){
    CHECK_CUDA(triangles); CHECK_CONTIGUOUS(triangles); CHECK_IS_FLOATING(triangles);
    CHECK_CUDA(query_pos); CHECK_CONTIGUOUS(query_pos); CHECK_IS_FLOATING(query_pos);

    int64_t P = query_pos.size(0);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query_pos));
    auto device = query_pos.device();

    at::TensorOptions opt(query_pos.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);
    torch::Tensor barycentric_coord = torch::empty({P,3}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);
    if (BLOCKS){
        GetUVofTriangle_Forward<<<BLOCKS,THREADS>>>(
            P,
            triangles.data_ptr<float>(),
            query_pos.data_ptr<float>(),
            barycentric_coord.data_ptr<float>()
        );
    }
    return barycentric_coord;
}

torch::Tensor get_uv_of_triangle_backward(
    const torch::Tensor & triangles, // Px3x3
    const torch::Tensor & query_pos, // Px3
    const torch::Tensor & grad_barycentric_coord // Px3
){
    CHECK_CUDA(triangles); CHECK_CONTIGUOUS(triangles); CHECK_IS_FLOATING(triangles);
    CHECK_CUDA(query_pos); CHECK_CONTIGUOUS(query_pos); CHECK_IS_FLOATING(query_pos);
    CHECK_CUDA(grad_barycentric_coord); CHECK_CONTIGUOUS(grad_barycentric_coord); CHECK_IS_FLOATING(grad_barycentric_coord);

    int64_t P = query_pos.size(0);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query_pos));
    auto device = query_pos.device();

    at::TensorOptions opt(query_pos.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);
    torch::Tensor grad_query_pos = torch::empty({P,3}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);
    if (BLOCKS){
        GetUVofTriangle_Backward<<<BLOCKS,THREADS>>>(
            P,
            triangles.data_ptr<float>(),
            query_pos.data_ptr<float>(),
            grad_barycentric_coord.data_ptr<float>(),
            grad_query_pos.data_ptr<float>()
        );
    }
    return grad_query_pos;
}

std::vector<torch::Tensor> get_nearest_mesh_points_forward(
    const torch::Tensor & adjacency_head, // 2 x N_vertex
    const torch::Tensor & adjacency_list, // N_tbd
    const torch::Tensor & face_vertex_list, // 3 x N_face
    const torch::Tensor & vertex_pos, // 3 x N_vertex
    const torch::Tensor & query_pos, // 3 x P
    const torch::Tensor & idxs // P x K
){
    CHECK_CUDA(adjacency_head); CHECK_CONTIGUOUS(adjacency_head); CHECK_IS_INT(adjacency_head);
    CHECK_CUDA(adjacency_list); CHECK_CONTIGUOUS(adjacency_list); CHECK_IS_INT(adjacency_list);
    CHECK_CUDA(face_vertex_list); CHECK_CONTIGUOUS(face_vertex_list); CHECK_IS_INT(face_vertex_list);

    CHECK_CUDA(vertex_pos); CHECK_CONTIGUOUS(vertex_pos); CHECK_IS_FLOATING(vertex_pos);
    CHECK_CUDA(query_pos); CHECK_CONTIGUOUS(query_pos); CHECK_IS_FLOATING(query_pos);
    CHECK_CUDA(idxs); CHECK_CONTIGUOUS(idxs); CHECK_IS_LONG(idxs);

    int64_t N_vertex = adjacency_head.size(1);
    int64_t N_face = face_vertex_list.size(1);
    int64_t P = query_pos.size(1);
    int64_t K = idxs.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query_pos));
    auto device = query_pos.device();

    at::TensorOptions opt(query_pos.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);
    at::TensorOptions opt_l(idxs.dtype()); opt_l = opt_l.device(device); opt_l = opt_l.requires_grad(false);

    torch::Tensor nearest_triangle_id = torch::empty({P}, opt_l);
    torch::Tensor barycentric_coord = torch::empty({3,P}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);
    //const int N_SLOT = 45;
    //const int N_SLOT = 64;
    const int N_SLOT = 128;
    if (BLOCKS){
        GetNearestMeshPoints_Forward<N_SLOT><<<BLOCKS, THREADS>>>(
            N_vertex, N_face, P, K,
            adjacency_head.data_ptr<int>(),
            adjacency_list.data_ptr<int>(),
            face_vertex_list.data_ptr<int>(),
            vertex_pos.data_ptr<float>(),
            query_pos.data_ptr<float>(),
            idxs.data_ptr<int64_t>(),
            nearest_triangle_id.data_ptr<int64_t>(),
            barycentric_coord.data_ptr<float>()
        );
    }
    return {nearest_triangle_id, barycentric_coord};
}

torch::Tensor get_nearest_mesh_points_backward(
    const torch::Tensor & face_vertex_list, // 3 x N_face
    const torch::Tensor & vertex_pos, // 3 x N_vertex
    const torch::Tensor & query_pos, // 3 x P,
    const torch::Tensor & nearest_triangle_id, // P
    const torch::Tensor & grad_barycentric_coord // 3 x P
){
    CHECK_CUDA(face_vertex_list); CHECK_CONTIGUOUS(face_vertex_list); CHECK_IS_INT(face_vertex_list);
    CHECK_CUDA(vertex_pos); CHECK_CONTIGUOUS(vertex_pos); CHECK_IS_FLOATING(vertex_pos);
    CHECK_CUDA(query_pos); CHECK_CONTIGUOUS(query_pos); CHECK_IS_FLOATING(query_pos);
    CHECK_CUDA(nearest_triangle_id); CHECK_CONTIGUOUS(nearest_triangle_id); CHECK_IS_LONG(nearest_triangle_id);

    int64_t N_vertex = vertex_pos.size(1);
    int64_t N_face = face_vertex_list.size(1);
    int64_t P = query_pos.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query_pos));
    auto device = query_pos.device();

    at::TensorOptions opt(query_pos.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor grad_query_pos = torch::empty({3, P}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);
    if(BLOCKS){
        GetNearestMeshPoints_Backward<<<BLOCKS, THREADS>>>(
            N_vertex, N_face, P,
            face_vertex_list.data_ptr<int>(),
            vertex_pos.data_ptr<float>(),
            query_pos.data_ptr<float>(),
            nearest_triangle_id.data_ptr<int64_t>(),
            grad_barycentric_coord.data_ptr<float>(),
            grad_query_pos.data_ptr<float>()
        );
    }
    return grad_query_pos;
}