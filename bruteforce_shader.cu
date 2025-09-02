#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

__device__ static constexpr float F0 = 0.04f;
__device__ static constexpr float thres = 1e-9f;

// struct float3{
//     float x,y,z;
// };

__device__ __forceinline__ float dot(const float3 & a, const float3 & b){
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 add(const float3 & a, const float3 & b){
    float3 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    return c;
}

__device__ __forceinline__ float grad_saturate(float x) {
	return (x >= 0.f && x <= 1.f) ? 1.f : 0.f;
}

__device__ __forceinline__ float3 normalize(const float3 & a){
    float L = sqrtf(dot(a,a));
    float3 out;
    out.x = a.x / L;
    out.y = a.y / L;
    out.z = a.z / L;
    return out;
}

__forceinline__ __device__ float3 grad_normalize(const float3 & v, const float3 & grad_output)
{
	float v2 = dot(v, v);
	float v1_5 = sqrtf(v2) * v2 ; // (x^2+y^2+z^2)^(+3/2)
	float gx = grad_output.x * (v.y * v.y + v.z * v.z) - grad_output.y * (v.x * v.y) - grad_output.z * (v.x * v.z);
	float gy = -grad_output.x * (v.x * v.y) + grad_output.y * (v.x * v.x + v.z * v.z) - grad_output.z * (v.y * v.z);
	float gz = -grad_output.x * (v.x * v.z) - grad_output.y * (v.y * v.z) + grad_output.z * (v.x * v.x + v.y * v.y);
	//return make_float3(gx,gy,gz) / v1_5;
	return make_float3(gx/v1_5, gy/v1_5, gz/v1_5);
}

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
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

//////////////////////


__device__ float getDeltaOmega(int iy, int W, int H){
    float delta_phi = (2 * M_PIf) / W;
    float delta_theta = M_PIf / H;
    float sin_theta1 = __sinf(delta_theta * (iy + 1) - M_PIf * 0.5f);
    float sin_theta0 = __sinf(delta_theta * iy - M_PIf * 0.5f);
    float delta_omega = (sin_theta1 - sin_theta0) * delta_phi;
    return delta_omega;
}

__device__ float3 getLightDir(int iy, int ix, int W, int H){
    float v = (float(iy) + 0.5f) / H;
    float u = (float(ix) + 0.5f) / W;
    float phi = u * M_PIf * 2;
    float theta = (v - 0.5f) * M_PIf;
    float sin_theta, cos_theta;
    float sin_phi, cos_phi;
    sincosf(theta, &sin_theta, &cos_theta);
    sincosf(phi, &sin_phi, &cos_phi);
    float3 c;
    c.y = sin_theta;
    c.x = cos_theta * cos_phi;
    c.z = cos_theta * sin_phi;
    return c;
}

__device__ __forceinline__ float FresnelApproximate(float R0, float cosTheta){
    float a = 1.f - __saturatef(cosTheta);
    float a2 = a * a;
    return (a2 * a2 * a) * (1 - R0) + R0;
}

__device__ __forceinline__ float grad_FresnelApproximate(float R0, float cosTheta, float grad){
    float a = 1.f - __saturatef(cosTheta);
	float a2 = a * a;
	return 5.f * (R0 - 1.f) * grad * (a2 * a2) * grad_saturate(cosTheta);
}

__device__ float G1_robust(
    const float3 & l, const float3 & v,
    const float3 & n, float alpha
){
    float dot_v_n = dot(v,n);
    float dot_l_n = dot(l,n);
    dot_v_n = max(dot_v_n, 0.f);
    dot_l_n = max(dot_l_n, 0.f);
    float a = dot_l_n / (dot_l_n * (1.f-alpha) + alpha);
    return a / (dot_v_n * (1.f-alpha) + alpha);
}

__device__ void grad_G1_robust(
    const float3 & l, const float3 & v,
    const float3 & n, float alpha, float grad,
    float3 & grad_l, float3 & grad_v, float3 & grad_n, float & grad_alpha
){
    float dot_v_n = dot(v,n);
    float dot_l_n = dot(l,n);
    dot_v_n = max(dot_v_n, 0.f);
    dot_l_n = max(dot_l_n, 0.f);
    float a_det = dot_l_n * (1.f-alpha) + alpha;
    float a = dot_l_n / a_det;
    float b = dot_v_n * (1.f - alpha) + alpha;
    float grad_a = grad / b;
    float grad_b = - grad * a / (b * b);
    // grad for a side
    float grad_dot_l_n = grad_a * alpha / (a_det * a_det);
    if( dot_l_n > 0.f){
        grad_l.x += grad_dot_l_n * n.x; grad_l.y += grad_dot_l_n * n.y; grad_l.z += grad_dot_l_n * n.z;
        grad_n.x += grad_dot_l_n * l.x; grad_n.y += grad_dot_l_n * l.y; grad_n.z += grad_dot_l_n * l.z;
    }
    grad_alpha += grad_a * dot_l_n  * (dot_l_n - 1) / (a_det * a_det);
    // grad for b side
    float grad_dot_v_n = grad_b * (1.f - alpha);
    if( dot_v_n > 0.f){
        grad_v.x += grad_dot_v_n * n.x; grad_v.y += grad_dot_v_n * n.y; grad_v.z += grad_dot_v_n * n.z;
        grad_n.x += grad_dot_v_n * v.x; grad_n.y += grad_dot_v_n * v.y; grad_n.z += grad_dot_v_n * v.z;
    }
    grad_alpha += grad_b * (1.f - dot_v_n);
}

__device__ float NDF1(
    const float3 & n, const float3 & h, float alpha2
){
    float dot_n_h = dot(n,h);
    float tmp = dot_n_h * dot_n_h * (alpha2 - 1.f) + 1.f;
    return alpha2 / max(M_PIf * tmp * tmp, thres);
    //return alpha2 / (M_PIf * tmp * tmp);
}

__device__ void grad_NDF1(
    const float3 & n, const float3 & h, float alpha2, float grad,
    float3 & grad_n, float3 & grad_h, float & grad_alpha2
){
    float dot_n_h = dot(n,h);
    float tmp = dot_n_h * dot_n_h * (alpha2 - 1.f) + 1.f;
    float tmp2 = M_PIf * tmp * tmp;
    grad_alpha2 += grad / max(tmp2, thres);
    if(tmp2 > thres){
        float grad_tmp = (-2.f * alpha2 * grad) / (tmp2 * tmp);
        grad_alpha2 += dot_n_h * dot_n_h * grad_tmp;
        float grad_dot_n_h = 2.f * grad_tmp * (alpha2 - 1.f) * dot_n_h;
        grad_n.x += grad_dot_n_h * h.x; grad_n.y += grad_dot_n_h * h.y; grad_n.z += grad_dot_n_h * h.z;
        grad_h.x += grad_dot_n_h * n.x; grad_h.y += grad_dot_n_h * n.y; grad_h.z += grad_dot_n_h * n.z;
    }
}

////////////////////////////////////

template<int C>
__global__ void __launch_bounds__(256) BruteForceDiffuseShaderKernel_Forward(
    int64_t P, int H, int W,
    const float * __restrict__ normal, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ envmap, // CxHxW
    const float * __restrict__ light2obj_rotmat, // 3x3
    float * __restrict__ shading // PxC
){
    __shared__ float s_buff[(C < 3 ? 3 : C) * 256]; // 3k
    float l_mat[9];
    float3 l_normal;
    float l_albedo[C];
    float l_shading[C];

    bool valid = (blockIdx.x * int64_t(256) + threadIdx.x  < P);

    /////////////// load mat
    if(threadIdx.x < 9){
        s_buff[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    if(valid){
        #pragma unroll
        for(int i = 0; i < 9; i++){
            l_mat[i] = s_buff[i];
        }
    }
    __syncthreads();
    ////////////// load normal
    {
        float tmp[3];
        sload<256,3>(P, valid, normal, s_buff, tmp);
        l_normal.x = tmp[0]; l_normal.y = tmp[1]; l_normal.z = tmp[2];
    }
    __syncthreads();
    ////////////// load albedo
    sload<256,C>(P, valid, albedo, s_buff, l_albedo);
    //__syncthreads();
    ////////////// clear shade
    if(valid){
        #pragma unroll
        for(int i = 0; i < C; i++)
            l_shading[i] = 0;
    }
    ////////////// load light
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    for(int i_loop = 0; i_loop < n_loop; i_loop++){ // process 256 lights each time
        __syncthreads();
        // read global to shared
        {
            int i_texel = i_loop * 256 + threadIdx.x;
            if(i_texel < N_texel){
                #pragma unroll
                for(int i_channel = 0; i_channel < C; i_channel++){
                    s_buff[i_channel * 256 + threadIdx.x] = envmap[i_channel * N_texel + i_texel];
                }
            }
        }
        __syncthreads();
        // shared to local
        if(valid){
            for(int j = 0; j < 256; j++){ // go through each light
                int i_texel = i_loop * 256 + j;
                if(i_texel < N_texel){
                    int ix = i_texel % W;
                    int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
                    iy = H - 1 - iy;
#endif
                    float3 l_dir = getLightDir(iy, ix, W, H);
                    float delta_omega = getDeltaOmega(iy, W, H);
                    float3 rot_l_dir;
                    rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
                    rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
                    rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
                    float cos_shade = dot(rot_l_dir, l_normal);
                    //if(cos_shade < 0) cos_shade = 0;
                    if(cos_shade > 0){
                        #pragma unroll
                        for(int i_channel = 0; i_channel < C; i_channel++){
                            l_shading[i_channel] += s_buff[i_channel * 256 + j] * l_albedo[i_channel] * (cos_shade/ M_PIf) * delta_omega;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    ////////////// store shading
    ssave<256,C>(P, valid, shading, s_buff, l_shading);
}

template<int C>
__global__ void __launch_bounds__(256) BruteForceDiffuseShaderKernel_Backward(
    int64_t P, int H, int W,
    const float * __restrict__ grad_shading, // PxC
    const float * __restrict__ normal, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ envmap, // CxHxW
    const float * __restrict__ light2obj_rotmat, // 3x3
    float * __restrict__ grad_normal, // Px3
    float * __restrict__ grad_albedo
){
    __shared__ float s_buff[(C < 3 ? 3 : C) * 256]; // 3k
    float l_mat[9];
    float3 l_normal;
    float l_albedo[C];
    float l_grad_shading[C];
    float l_grad_normal[3];
    float l_grad_albedo[C];

    bool valid = (blockIdx.x * int64_t(256) + threadIdx.x  < P);

    /////////////// load mat
    if(threadIdx.x < 9){
        s_buff[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    if(valid){
        #pragma unroll
        for(int i = 0; i < 9; i++){
            l_mat[i] = s_buff[i];
        }
    }
    __syncthreads();
    ////////////// load normal
    {
        float tmp[3];
        sload<256,3>(P,valid,normal,s_buff,tmp);
        l_normal.x = tmp[0]; l_normal.y = tmp[1]; l_normal.z = tmp[2];
    }
    __syncthreads();
    ////////////// load albedo
    sload<256,C>(P, valid, albedo, s_buff, l_albedo);
    __syncthreads();
    ////////////// load grad_shading
    sload<256,C>(P, valid, grad_shading, s_buff, l_grad_shading);
    //__syncthreads();
    ////////////// clear shade
    if(valid){
        l_grad_normal[0] = 0;
        l_grad_normal[1] = 0;
        l_grad_normal[2] = 0;
        #pragma unroll
        for(int i = 0; i < C ; i++){
            l_grad_albedo[i] = 0;
        }
    }
    ////////////// load light
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    for(int i_loop = 0; i_loop < n_loop; i_loop++){ // process 256 lights each time
        __syncthreads();
        // read global to shared
        {
            int i_texel = i_loop * 256 + threadIdx.x;
            if(i_texel < N_texel){
                #pragma unroll
                for(int i_channel = 0; i_channel < C; i_channel++){
                    s_buff[i_channel * 256 + threadIdx.x] = envmap[i_channel * N_texel + i_texel];
                }
            }
        }
        __syncthreads();
        // shared to local
        if(valid){
            for(int j = 0; j < 256; j++){ // go through each light
                int i_texel = i_loop * 256 + j;
                if(i_texel < N_texel){
                    int ix = i_texel % W;
                    int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
                    iy = H - 1 - iy;
#endif
                    float3 l_dir = getLightDir(iy, ix, W, H);
                    float delta_omega = getDeltaOmega(iy, W, H);
                    float3 rot_l_dir;
                    rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
                    rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
                    rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
                    float cos_shade = dot(rot_l_dir, l_normal);
                    float grad_dotn = 0.f;
                    //if(cos_shade < 0) cos_shade = 0;
                    if(cos_shade > 0){
                        #pragma unroll
                        for(int i_channel = 0; i_channel < C; i_channel++){
                            float l_intensity = s_buff[i_channel * 256 + j];
                            l_grad_albedo[i_channel] += l_grad_shading[i_channel] * l_intensity * (cos_shade / M_PIf) * delta_omega;
                            grad_dotn += l_grad_shading[i_channel] * l_intensity *  l_albedo[i_channel] * delta_omega / M_PIf;
                        }
                        l_grad_normal[0] += grad_dotn * rot_l_dir.x;
                        l_grad_normal[1] += grad_dotn * rot_l_dir.y;
                        l_grad_normal[2] += grad_dotn * rot_l_dir.z;
                    }
                }
            }
        }
    }
    __syncthreads();
    ////////////// store grad_normal
    ssave<256,3>(P, valid, grad_normal, s_buff, l_grad_normal);
    __syncthreads();
    ////////////// store grad_albedo
    ssave<256,C>(P, valid, grad_albedo, s_buff, l_grad_albedo);
}

template<int C>
__global__ void __launch_bounds__(256) BruteForceDiffuseShaderKernel_BackwardEnvmap(
    int64_t P, int H, int W,
    const float * __restrict__ grad_shading, // PxC
    const float * __restrict__ normal, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ light2obj_rotmat, // 3x3
    float * __restrict__ grad_envmap // CxHxW
){
    __shared__ float s_normal[3 * 256]; // 9k
    __shared__ float s_buff[C * 256];
    __shared__ float s_grad[C * 256];
    float l_mat[9];
    float3 l_normal;
    float l_grad_envmap[C];

    /////////////// load mat
    if(threadIdx.x < 9){
        s_buff[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    #pragma unroll
    for(int i = 0; i < 9; i++){
        l_mat[i] = s_buff[i];
    }
    __syncthreads();
    ////////////// load normal & albedo & gradin
    #pragma unroll
    for(int i = 0; i < 3; i++){
        int loc_offset = i * 256 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256 * 3) + loc_offset;
        if(i_elem < P * 3){
            s_normal[loc_offset] = normal[i_elem];
        }
    }
    #pragma unroll
    for(int i = 0; i < C; i++){
        int loc_offset = i * 256 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256 * C) + loc_offset;
        if(i_elem < P * C){
            s_buff[loc_offset] = albedo[i_elem];
            s_grad[loc_offset] = grad_shading[i_elem];
        }
    }
    __syncthreads();
    ////////////// acc grad of each texel
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    for(int i_loop = 0; i_loop < n_loop; i_loop ++){
        int i_texel = i_loop * 256 + threadIdx.x;
        if(i_texel < N_texel){ // check valid texel
            int ix = i_texel % W;
            int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
            iy = H - 1 - iy;
#endif
            float3 l_dir = getLightDir(iy, ix, W, H);
            float delta_omega = getDeltaOmega(iy, W, H);
            float3 rot_l_dir;
            rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
            rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
            rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
            // clear grad
            #pragma unroll
            for(int i = 0; i < C; i++){
                l_grad_envmap[i] = 0;
            }
            for(int i = 0; i < 256; i++){
                if( blockIdx.x * int64_t(256) + i < P){ // check valid fragment
                    l_normal.x = s_normal[i * 3 + 0];
                    l_normal.y = s_normal[i * 3 + 1];
                    l_normal.z = s_normal[i * 3 + 2];
                    float cos_shade = dot(rot_l_dir, l_normal);
                    //if(cos_shade < 0) cos_shade = 0;
                    if(cos_shade > 0){
                        #pragma unroll
                        for(int i_channel = 0; i_channel < C; i_channel++){
                            float albedo_ = s_buff[i * C + i_channel];
                            float grad_ = s_grad[i * C + i_channel];
                            l_grad_envmap[i_channel] += grad_ * albedo_ * (cos_shade / M_PIf) * delta_omega;
                        }
                    }
                }
            }
            // global write
            #pragma unroll
            for(int i_channel = 0; i_channel < C; i_channel++){
                atomicAdd(grad_envmap + (i_channel * N_texel + i_texel), l_grad_envmap[i_channel]);
            }
        }
    }
}

template<int C>
__global__ void __launch_bounds__(256) BruteForceSpecularShaderKernel_Forward(
    int64_t P, int H, int W,
    const float * __restrict__ normal, // Px3
    const float * __restrict__ view_dir, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ specular_albedo, // PxC
    const float * __restrict__ roughness, // Px1
    const float * __restrict__ envmap, // CxHxW
    const float * __restrict__ light2obj_rotmat, // 3x3
    float * __restrict__ shading,
    bool enable_diffuse, bool enable_specular
){
    __shared__ float s_buff[(C < 3 ? 3 : C) * 256]; // 3k
    float l_mat[9];
    float3 l_normal;
    float3 l_viewdir;
    float l_albedo[C];
    float l_specular_albedo[C];
    float l_roughness, alpha0, alpha1;
    float l_shading[C];

    bool valid = (blockIdx.x * int64_t(256) + threadIdx.x  < P);

    /////////////// load mat
    if(threadIdx.x < 9){
        s_buff[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    if(valid){
        #pragma unroll
        for(int i = 0; i < 9; i++){
            l_mat[i] = s_buff[i];
        }
    }
    __syncthreads();
    ////////////// load normal
    {
        float tmp[3];
        sload<256,3>(P, valid, normal, s_buff, tmp);
        l_normal.x = tmp[0]; l_normal.y = tmp[1]; l_normal.z = tmp[2];
    }
    __syncthreads();
    ////////////// load albedo
    if (enable_diffuse){
        sload<256,C>(P, valid, albedo, s_buff, l_albedo);
        __syncthreads();
    }
    ////////////// load specular_albedo & roughness
    if (enable_specular){
        sload<256,C>(P, valid, specular_albedo, s_buff, l_specular_albedo);
        __syncthreads();
        sload<256,1>(P, valid, roughness, s_buff, &l_roughness);
        alpha0 = l_roughness * l_roughness; // alpha = roughness ^ 2
        alpha0 = alpha0 * alpha0; // alpha2 = alpha * alpha
        alpha1 = l_roughness + 1.f;
        alpha1 = (alpha1 * alpha1) / 8.f;
        __syncthreads();
        float tmp[3];
        sload<256,3>(P, valid, view_dir, s_buff, tmp);
        l_viewdir.x = tmp[0]; l_viewdir.y = tmp[1]; l_viewdir.z = tmp[2];
        __syncthreads();
    }
    ////////////// clear shade
    if(valid){
        #pragma unroll
        for(int i = 0; i < C; i++)
            l_shading[i] = 0;
    }
    ////////////// load light
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    for(int i_loop = 0; i_loop < n_loop; i_loop++) { // process 256 lights each time
        // read global to shared
        {
            int i_texel = i_loop * 256 + threadIdx.x;
            if(i_texel < N_texel){
                #pragma unroll
                for(int i_channel = 0; i_channel < C; i_channel++){
                    s_buff[i_channel * 256 + threadIdx.x] = envmap[i_channel * N_texel + i_texel];
                }
            }
        }
        __syncthreads();
        // shared to local
        if (valid){
            for(int j = 0; j < 256; j++){ // go through each light
                int i_texel = i_loop * 256 + j;
                if(i_texel < N_texel){
                    int ix = i_texel % W;
                    int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
                    iy = H - 1 - iy;
#endif
                    float3 l_dir = getLightDir(iy, ix, W, H);
                    float delta_omega = getDeltaOmega(iy, W, H);
                    float3 rot_l_dir;
                    rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
                    rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
                    rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
                    // load light color
                    float l_envmap[C];
                    #pragma unroll
                    for(int i_channel = 0; i_channel < C; i_channel++){
                        l_envmap[i_channel] = s_buff[i_channel * 256 + j];
                    }
                    if (enable_diffuse){
                        float cos_shade = dot(rot_l_dir, l_normal);
                        //if(cos_shade < 0) cos_shade = 0;
                        if(cos_shade > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                l_shading[i_channel] += l_envmap[i_channel] * l_albedo[i_channel] * (cos_shade/ M_PIf) * delta_omega;
                            }
                        }
                    }
                    if (enable_specular){
                        float3 h = add(rot_l_dir, l_viewdir);
                        h = normalize(h);
                        float cosh = dot(rot_l_dir, h);
                        float F = FresnelApproximate(F0, cosh);
                        float G = G1_robust(rot_l_dir, l_viewdir, l_normal, alpha1);
                        float D = NDF1(l_normal, h, alpha0);
                        float tmp = (F * G * D) / 4.f ;
                        //tmp = max(tmp, 0.);
                        if(tmp > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                l_shading[i_channel] += l_envmap[i_channel] * l_specular_albedo[i_channel] * (tmp * delta_omega);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    //__syncthreads();
    ////////////// store shading
    ssave<256,C>(P, valid, shading, s_buff, l_shading);
}

template<int C>
__global__ void __launch_bounds__(256) BruteForceSpecularShaderKernel_Backward(
    int64_t P, int H, int W,
    const float * __restrict__ grad_shading, // PxC
    const float * __restrict__ normal, // Px3
    const float * __restrict__ view_dir, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ specular_albedo, // PxC
    const float * __restrict__ roughness, // Px1
    const float * __restrict__ envmap, // CxHxW
    const float * __restrict__ light2obj_rotmat, // 3x3
    float * __restrict__ grad_normal, // Px3
    float * __restrict__ grad_view_dir, // Px3
    float * __restrict__ grad_albedo, // PxC
    float * __restrict__ grad_specular_albedo, // PxC
    float * __restrict__ grad_roughness, // Px1
    bool enable_diffuse, bool enable_specular
){
    __shared__ float s_buff[(C < 3 ? 3 : C) * 256]; // 3k
    float l_mat[9];
    float3 l_normal;
    float3 l_viewdir;
    float l_albedo[C];
    float l_specular_albedo[C];
    float l_roughness, alpha0, alpha1;
    //
    float l_grad_shading[C];
    float l_grad_normal[3];
    float l_grad_viewdir[3];
    float l_grad_albedo[C];
    float l_grad_specular_albedo[C];
    float grad_alpha0, grad_alpha1;

    bool valid = (blockIdx.x * int64_t(256) + threadIdx.x  < P);

    /////////////// load mat
    if(threadIdx.x < 9){
        s_buff[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    if(valid){
        #pragma unroll
        for(int i = 0; i < 9; i++){
            l_mat[i] = s_buff[i];
        }
    }
    __syncthreads();
    ////////////// load normal
    {
        float tmp[3];
        sload<256,3>(P,valid,normal,s_buff,tmp);
        l_normal.x = tmp[0]; l_normal.y = tmp[1]; l_normal.z = tmp[2];
    }
    __syncthreads();
    ////////////// load albedo
    if (enable_diffuse){
        sload<256,C>(P, valid, albedo, s_buff, l_albedo);
        __syncthreads();
    }
    ////////////// load specular_albedo & roughness
    if (enable_specular){
        sload<256,C>(P, valid, specular_albedo, s_buff, l_specular_albedo);
        __syncthreads();
        sload<256,1>(P, valid, roughness, s_buff, &l_roughness);
        alpha0 = l_roughness * l_roughness; // alpha = roughness ^ 2
        alpha0 = alpha0 * alpha0; // alpha2 = alpha * alpha
        alpha1 = l_roughness + 1.f;
        alpha1 = (alpha1 * alpha1) / 8.f;
        __syncthreads();
        float tmp[3];
        sload<256,3>(P, valid, view_dir, s_buff, tmp);
        l_viewdir.x = tmp[0]; l_viewdir.y = tmp[1]; l_viewdir.z = tmp[2];
        __syncthreads();
    }
    ////////////// load grad_shading
    sload<256,C>(P, valid, grad_shading, s_buff, l_grad_shading);
    //__syncthreads();
    ////////////// clear shade
    if(valid){
        l_grad_normal[0] = 0; l_grad_normal[1] = 0; l_grad_normal[2] = 0;
        if (enable_diffuse){
            #pragma unroll
            for(int i = 0; i < C ; i++){
                l_grad_albedo[i] = 0;
            }
        }
        if (enable_specular){
            l_grad_viewdir[0] = 0; l_grad_viewdir[1] = 0; l_grad_viewdir[2] = 0;
            grad_alpha0 = 0; grad_alpha1 = 0;
            #pragma unroll
            for(int i = 0; i < C ; i++){
                l_grad_specular_albedo[i] = 0;
            }
        }
    }
    ////////////// load light
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    for(int i_loop = 0; i_loop < n_loop; i_loop++){ // process 256 lights each time
        __syncthreads();
        // read global to shared
        {
            int i_texel = i_loop * 256 + threadIdx.x;
            if(i_texel < N_texel){
                #pragma unroll
                for(int i_channel = 0; i_channel < C; i_channel++){
                    s_buff[i_channel * 256 + threadIdx.x] = envmap[i_channel * N_texel + i_texel];
                }
            }
        }
        __syncthreads();
        // shared to local
        if (valid){
            for(int j = 0; j < 256; j++){ // go through each light
                int i_texel = i_loop * 256 + j;
                if(i_texel < N_texel){
                    int ix = i_texel % W;
                    int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
                    iy = H - 1 - iy;
#endif
                    float3 l_dir = getLightDir(iy, ix, W, H);
                    float delta_omega = getDeltaOmega(iy, W, H);
                    float3 rot_l_dir;
                    rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
                    rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
                    rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
                    // load light color
                    float l_envmap[C];
                    #pragma unroll
                    for(int i_channel = 0; i_channel < C; i_channel++){
                        l_envmap[i_channel] = s_buff[i_channel * 256 + j];
                    }
                    ///// gradient for diffuse
                    if (enable_diffuse){
                        float cos_shade = dot(rot_l_dir, l_normal);
                        float grad_dotn = 0.f;
                        if(cos_shade > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float l_intensity = l_envmap[i_channel];
                                l_grad_albedo[i_channel] += l_grad_shading[i_channel] * l_intensity * (cos_shade / M_PIf) * delta_omega;
                                grad_dotn += l_grad_shading[i_channel] * l_intensity *  l_albedo[i_channel] * delta_omega / M_PIf;
                            }
                            l_grad_normal[0] += grad_dotn * rot_l_dir.x;
                            l_grad_normal[1] += grad_dotn * rot_l_dir.y;
                            l_grad_normal[2] += grad_dotn * rot_l_dir.z;
                        }
                    }
                    ///// gradient for specular
                    if (enable_specular){
                        float3 h_before = add(rot_l_dir, l_viewdir);
                        float3 h = normalize(h_before);
                        float cosh = dot(rot_l_dir, h);
                        float F = FresnelApproximate(F0, cosh);
                        float G = G1_robust(rot_l_dir, l_viewdir, l_normal, alpha1);
                        float D = NDF1(l_normal, h, alpha0);
                        float tmp = (F * G * D) / 4.f ;
                        float grad_tmp = 0.f;
                        if (tmp > 0){
                            // grad_shading -> grad_specular_albedo, grad_tmp
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float l_intensity = l_envmap[i_channel];
                                l_grad_specular_albedo[i_channel] += l_grad_shading[i_channel] * l_intensity * (tmp * delta_omega);
                                grad_tmp += l_grad_shading[i_channel] * l_intensity * l_specular_albedo[i_channel] * delta_omega;
                            }
                            //// grad_tmp -> grad_F, grad_D, grad_G
                            float grad_F = grad_tmp * (G * D) / 4.f;
                            float grad_G = grad_tmp * (F * D) / 4.f;
                            float grad_D = grad_tmp * (F * G) / 4.f;
                            //// grad_F -> cosh
                            float grad_cosh = grad_FresnelApproximate(F0, cosh, grad_F);
                            float3 grad_h;
                            float3 g_l_dir; // NOTE: Currently not used
                            grad_h.x = grad_cosh * rot_l_dir.x; grad_h.y = grad_cosh * rot_l_dir.y; grad_h.z = grad_cosh * rot_l_dir.z;
                            g_l_dir.x = grad_cosh * h.x; g_l_dir.y = grad_cosh * h.y; g_l_dir.z = grad_cosh * h.z;
                            //// grad_G -> n,l,v,alpha1
                            float3 g_viewdir = make_float3(0.f,0.f,0.f);
                            float3 g_normal = make_float3(0.f,0.f,0.f);
                            grad_G1_robust(
                                rot_l_dir, l_viewdir, l_normal, alpha1, grad_G,
                                g_l_dir, g_viewdir, g_normal, grad_alpha1
                            );
                            //// grad_D -> n,h,alpha0
                            grad_NDF1(
                                l_normal, h, alpha0, grad_D,
                                g_normal, grad_h, grad_alpha0
                            );
                            // h -> h_before
                            grad_h = grad_normalize(h_before, grad_h);
                            // h_before -> v,l
                            g_viewdir = add(g_viewdir, grad_h);
                            g_l_dir = add(g_l_dir, grad_h);
                            // Merge
                            l_grad_viewdir[0] += g_viewdir.x; l_grad_viewdir[1] += g_viewdir.y; l_grad_viewdir[2] += g_viewdir.z;
                            l_grad_normal[0] += g_normal.x; l_grad_normal[1] += g_normal.y; l_grad_normal[2] += g_normal.z;
                            // ? <- g_l_dir
                        }
                    }
                }
            }
        }
    }
    ////////////// store grad_normal
    __syncthreads();
    ssave<256,3>(P, valid, grad_normal, s_buff, l_grad_normal);
    if(enable_diffuse){
        ////////////// store grad_albedo
        __syncthreads();
        ssave<256,C>(P, valid, grad_albedo, s_buff, l_grad_albedo);
    }
    if(enable_specular){
        ////////////// store view_dir
        __syncthreads();
        ssave<256,3>(P, valid, grad_view_dir, s_buff, l_grad_viewdir);
        ////////////// store grad_specular_albedo
        __syncthreads();
        ssave<256,C>(P, valid, grad_specular_albedo, s_buff, l_grad_specular_albedo);
        ////////////// store albedo
        float l_grad_roughness = grad_alpha0 * 4.f * l_roughness * l_roughness * l_roughness;
        l_grad_roughness += grad_alpha1 * (1.f + l_roughness) / 4.f;
        __syncthreads();
        ssave<256,1>(P, valid, grad_roughness, s_buff, &l_grad_roughness);
    }
}

template<int C>
__global__ void __launch_bounds__(256) BruteForceSpecularShaderKernel_BackwardEnvmap(
    int64_t P, int H, int W,
    const float * __restrict__ grad_shading, // PxC
    const float * __restrict__ normal, // Px3
    const float * __restrict__ view_dir, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ specular_albedo, // PxC
    const float * __restrict__ roughness, // Px1
    const float * __restrict__ light2obj_rotmat, // 3x3
    float * __restrict__ grad_envmap, // CxHxW
    bool enable_diffuse, bool enable_specular
){
    // 16k
    __shared__ float s_normal[3 * 256]; // 3k
    __shared__ float s_viewdir[3 * 256]; // 3k
    __shared__ float s_albedo[C * 256]; // 3k
    __shared__ float s_specular_albedo[C * 256]; // 3k
    __shared__ float s_roughness[1 * 256]; // 1k
    __shared__ float s_grad[C * 256]; // 3k
    float l_mat[9];
    float3 l_normal;
    float l_grad_envmap[C];
    /////////////// load mat
    if(threadIdx.x < 9){
        s_normal[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    #pragma unroll
    for(int i = 0; i < 9; i++){
        l_mat[i] = s_normal[i];
    }
    __syncthreads();
    ////////////// load normal & viewdir & albedo & specular_albedo & roughness & gradin
    #pragma unroll
    for(int i = 0; i < 3; i++){
        int loc_offset = i * 256 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256 * 3) + loc_offset;
        if(i_elem < P * 3){
            s_normal[loc_offset] = normal[i_elem];
            if (enable_specular)
                s_viewdir[loc_offset] = view_dir[i_elem];
        }
    }
    {
        int loc_offset = threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256) + loc_offset;
        if(i_elem < P){
            s_roughness[loc_offset] = roughness[i_elem];
        }
    }
    #pragma unroll
    for(int i = 0; i < C; i++){
        int loc_offset = i * 256 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256 * C) + loc_offset;
        if(i_elem < P * C){
            if (enable_diffuse)
                s_albedo[loc_offset] = albedo[i_elem];
            if (enable_specular)
                s_specular_albedo[loc_offset] = specular_albedo[i_elem];
            s_grad[loc_offset] = grad_shading[i_elem];
        }
    }
    __syncthreads();
    ////////////// acc grad of each texel
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    for(int i_loop = 0; i_loop < n_loop; i_loop ++){
        int i_texel = i_loop * 256 + threadIdx.x;
        if(i_texel < N_texel){ // check valid texel
            int ix = i_texel % W;
            int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
            iy = H - 1 - iy;
#endif
            float3 l_dir = getLightDir(iy, ix, W, H);
            float delta_omega = getDeltaOmega(iy, W, H);
            float3 rot_l_dir;
            rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
            rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
            rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
            // clear grad
            #pragma unroll
            for(int i = 0; i < C; i++){
                l_grad_envmap[i] = 0;
            }
            for(int i = 0; i < 256; i++){
                if( blockIdx.x * int64_t(256) + i < P){ // check valid fragment
                    l_normal.x = s_normal[i * 3 + 0];
                    l_normal.y = s_normal[i * 3 + 1];
                    l_normal.z = s_normal[i * 3 + 2];
                    if(enable_diffuse){
                        float cos_shade = dot(rot_l_dir, l_normal);
                        if(cos_shade > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float albedo_ = s_albedo[i * C + i_channel];
                                float grad_ = s_grad[i * C + i_channel];
                                l_grad_envmap[i_channel] += grad_ * albedo_ * (cos_shade / M_PIf) * delta_omega;
                            }
                        }
                    }
                    if (enable_specular){
                        float3 l_viewdir;
                        l_viewdir.x = s_viewdir[i * 3 + 0];
                        l_viewdir.y = s_viewdir[i * 3 + 1];
                        l_viewdir.z = s_viewdir[i * 3 + 2];
                        float l_roughness = s_roughness[i];
                        float alpha0 = l_roughness * l_roughness;
                        alpha0 = alpha0 * alpha0;
                        float alpha1 = l_roughness + 1.f;
                        alpha1 = (alpha1 * alpha1) / 8.f;
                        float3 h = add(rot_l_dir, l_viewdir);
                        h = normalize(h);
                        float cosh = dot(rot_l_dir, h);
                        float F = FresnelApproximate(F0, cosh);
                        float G = G1_robust(rot_l_dir, l_viewdir, l_normal, alpha1);
                        float D = NDF1(l_normal, h, alpha0);
                        float tmp = (F * G * D) / 4.f ;
                        if(tmp > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float albedo_ = s_specular_albedo[i * C + i_channel];
                                float grad_ = s_grad[i * C + i_channel];
                                l_grad_envmap[i_channel] += grad_ * albedo_ * tmp * delta_omega;
                            }
                        }
                    }
                }
            }
            // global write
            #pragma unroll
            for(int i_channel = 0; i_channel < C; i_channel++){
                atomicAdd(grad_envmap + (i_channel * N_texel + i_texel), l_grad_envmap[i_channel]);
            }
        }
    }
}

template<int C>
__global__ void __launch_bounds__(256) BruteForceSpecularShaderClampKernel_Forward(
    int64_t P, int H, int W,
    const float * __restrict__ normal, // Px3
    const float * __restrict__ view_dir, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ specular_albedo, // PxC
    const float * __restrict__ roughness, // Px1
    const float * __restrict__ envmap, // CxHxW
    const float * __restrict__ light2obj_rotmat, // 3x3
    float * __restrict__ shading,
    bool enable_diffuse, bool enable_specular
){
    __shared__ float s_buff[(C < 3 ? 3 : C) * 256]; // 3k
    float l_mat[9];
    float3 l_normal;
    float3 l_viewdir;
    float l_albedo[C];
    float l_specular_albedo[C];
    float l_roughness, alpha0, alpha1;
    float l_shading[C];

    bool valid = (blockIdx.x * int64_t(256) + threadIdx.x  < P);

    /////////////// load mat
    if(threadIdx.x < 9){
        s_buff[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    if(valid){
        #pragma unroll
        for(int i = 0; i < 9; i++){
            l_mat[i] = s_buff[i];
        }
    }
    __syncthreads();
    ////////////// load normal
    {
        float tmp[3];
        sload<256,3>(P, valid, normal, s_buff, tmp);
        l_normal.x = tmp[0]; l_normal.y = tmp[1]; l_normal.z = tmp[2];
        __syncthreads();
        sload<256,3>(P, valid, view_dir, s_buff, tmp);
        l_viewdir.x = tmp[0]; l_viewdir.y = tmp[1]; l_viewdir.z = tmp[2];
    }
    __syncthreads();
    ////////////// load albedo
    if (enable_diffuse){
        sload<256,C>(P, valid, albedo, s_buff, l_albedo);
        __syncthreads();
    }
    ////////////// load specular_albedo & roughness
    if (enable_specular){
        sload<256,C>(P, valid, specular_albedo, s_buff, l_specular_albedo);
        __syncthreads();
        sload<256,1>(P, valid, roughness, s_buff, &l_roughness);
        alpha0 = l_roughness * l_roughness; // alpha = roughness ^ 2
        alpha0 = alpha0 * alpha0; // alpha2 = alpha * alpha
        alpha1 = l_roughness + 1.f;
        alpha1 = (alpha1 * alpha1) / 8.f;
        __syncthreads();
    }
    ////////////// clear shade
    if(valid){
        #pragma unroll
        for(int i = 0; i < C; i++)
            l_shading[i] = 0;
    }
    ////////////// load light
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    float dot_view = dot(l_normal, l_viewdir);
    for(int i_loop = 0; i_loop < n_loop; i_loop++) { // process 256 lights each time
        // read global to shared
        {
            int i_texel = i_loop * 256 + threadIdx.x;
            if(i_texel < N_texel){
                #pragma unroll
                for(int i_channel = 0; i_channel < C; i_channel++){
                    s_buff[i_channel * 256 + threadIdx.x] = envmap[i_channel * N_texel + i_texel];
                }
            }
        }
        __syncthreads();
        // shared to local
        if(valid && dot_view > 0.){
            for(int j = 0; j < 256; j++){ // go through each light
                int i_texel = i_loop * 256 + j;
                if(i_texel < N_texel){
                    int ix = i_texel % W;
                    int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
                    iy = H - 1 - iy;
#endif
                    float3 l_dir = getLightDir(iy, ix, W, H);
                    float delta_omega = getDeltaOmega(iy, W, H);
                    float3 rot_l_dir;
                    rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
                    rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
                    rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
                    // load light color
                    float l_envmap[C];
                    #pragma unroll
                    for(int i_channel = 0; i_channel < C; i_channel++){
                        l_envmap[i_channel] = s_buff[i_channel * 256 + j];
                    }
                    if (enable_diffuse){
                        float cos_shade = dot(rot_l_dir, l_normal);
                        //if(cos_shade < 0) cos_shade = 0;
                        if(cos_shade > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                l_shading[i_channel] += l_envmap[i_channel] * l_albedo[i_channel] * (cos_shade/ M_PIf) * delta_omega;
                            }
                        }
                    }
                    if (enable_specular){
                        float3 h = add(rot_l_dir, l_viewdir);
                        h = normalize(h);
                        float cosh = dot(rot_l_dir, h);
                        float F = FresnelApproximate(F0, cosh);
                        float G = G1_robust(rot_l_dir, l_viewdir, l_normal, alpha1);
                        float D = NDF1(l_normal, h, alpha0);
                        float tmp = (F * G * D) / 4.f ;
                        //tmp = max(tmp, 0.);
                        if(tmp > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                l_shading[i_channel] += l_envmap[i_channel] * l_specular_albedo[i_channel] * (tmp * delta_omega);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    //__syncthreads();
    ////////////// store shading
    ssave<256,C>(P, valid, shading, s_buff, l_shading);
}

template<int C>
__global__ void __launch_bounds__(256) BruteForceSpecularShaderClampKernel_Backward(
    int64_t P, int H, int W,
    const float * __restrict__ grad_shading, // PxC
    const float * __restrict__ normal, // Px3
    const float * __restrict__ view_dir, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ specular_albedo, // PxC
    const float * __restrict__ roughness, // Px1
    const float * __restrict__ envmap, // CxHxW
    const float * __restrict__ light2obj_rotmat, // 3x3
    float * __restrict__ grad_normal, // Px3
    float * __restrict__ grad_view_dir, // Px3
    float * __restrict__ grad_albedo, // PxC
    float * __restrict__ grad_specular_albedo, // PxC
    float * __restrict__ grad_roughness, // Px1
    bool enable_diffuse, bool enable_specular
){
    __shared__ float s_buff[(C < 3 ? 3 : C) * 256]; // 3k
    float l_mat[9];
    float3 l_normal;
    float3 l_viewdir;
    float l_albedo[C];
    float l_specular_albedo[C];
    float l_roughness, alpha0, alpha1;
    //
    float l_grad_shading[C];
    float l_grad_normal[3];
    float l_grad_viewdir[3];
    float l_grad_albedo[C];
    float l_grad_specular_albedo[C];
    float grad_alpha0, grad_alpha1;

    bool valid = (blockIdx.x * int64_t(256) + threadIdx.x  < P);

    /////////////// load mat
    if(threadIdx.x < 9){
        s_buff[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    if(valid){
        #pragma unroll
        for(int i = 0; i < 9; i++){
            l_mat[i] = s_buff[i];
        }
    }
    __syncthreads();
    ////////////// load normal
    {
        float tmp[3];
        sload<256,3>(P,valid,normal,s_buff,tmp);
        l_normal.x = tmp[0]; l_normal.y = tmp[1]; l_normal.z = tmp[2];
        __syncthreads();
        sload<256,3>(P, valid, view_dir, s_buff, tmp);
        l_viewdir.x = tmp[0]; l_viewdir.y = tmp[1]; l_viewdir.z = tmp[2];
    }
    __syncthreads();
    ////////////// load albedo
    if (enable_diffuse){
        sload<256,C>(P, valid, albedo, s_buff, l_albedo);
        __syncthreads();
    }
    ////////////// load specular_albedo & roughness
    if (enable_specular){
        sload<256,C>(P, valid, specular_albedo, s_buff, l_specular_albedo);
        __syncthreads();
        sload<256,1>(P, valid, roughness, s_buff, &l_roughness);
        alpha0 = l_roughness * l_roughness; // alpha = roughness ^ 2
        alpha0 = alpha0 * alpha0; // alpha2 = alpha * alpha
        alpha1 = l_roughness + 1.f;
        alpha1 = (alpha1 * alpha1) / 8.f;
        __syncthreads();
    }
    ////////////// load grad_shading
    sload<256,C>(P, valid, grad_shading, s_buff, l_grad_shading);
    //__syncthreads();
    ////////////// clear shade
    if(valid){
        l_grad_normal[0] = 0; l_grad_normal[1] = 0; l_grad_normal[2] = 0;
        if (enable_diffuse){
            #pragma unroll
            for(int i = 0; i < C ; i++){
                l_grad_albedo[i] = 0;
            }
        }
        if (enable_specular){
            l_grad_viewdir[0] = 0; l_grad_viewdir[1] = 0; l_grad_viewdir[2] = 0;
            grad_alpha0 = 0; grad_alpha1 = 0;
            #pragma unroll
            for(int i = 0; i < C ; i++){
                l_grad_specular_albedo[i] = 0;
            }
        }
    }
    ////////////// load light
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    float dot_view = dot(l_normal, l_viewdir);
    for(int i_loop = 0; i_loop < n_loop; i_loop++){ // process 256 lights each time
        __syncthreads();
        // read global to shared
        {
            int i_texel = i_loop * 256 + threadIdx.x;
            if(i_texel < N_texel){
                #pragma unroll
                for(int i_channel = 0; i_channel < C; i_channel++){
                    s_buff[i_channel * 256 + threadIdx.x] = envmap[i_channel * N_texel + i_texel];
                }
            }
        }
        __syncthreads();
        // shared to local
        if (valid && dot_view > 0.){
            for(int j = 0; j < 256; j++){ // go through each light
                int i_texel = i_loop * 256 + j;
                if(i_texel < N_texel){
                    int ix = i_texel % W;
                    int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
                    iy = H - 1 - iy;
#endif
                    float3 l_dir = getLightDir(iy, ix, W, H);
                    float delta_omega = getDeltaOmega(iy, W, H);
                    float3 rot_l_dir;
                    rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
                    rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
                    rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
                    // load light color
                    float l_envmap[C];
                    #pragma unroll
                    for(int i_channel = 0; i_channel < C; i_channel++){
                        l_envmap[i_channel] = s_buff[i_channel * 256 + j];
                    }
                    ///// gradient for diffuse
                    if (enable_diffuse){
                        float cos_shade = dot(rot_l_dir, l_normal);
                        float grad_dotn = 0.f;
                        if(cos_shade > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float l_intensity = l_envmap[i_channel];
                                l_grad_albedo[i_channel] += l_grad_shading[i_channel] * l_intensity * (cos_shade / M_PIf) * delta_omega;
                                grad_dotn += l_grad_shading[i_channel] * l_intensity *  l_albedo[i_channel] * delta_omega / M_PIf;
                            }
                            l_grad_normal[0] += grad_dotn * rot_l_dir.x;
                            l_grad_normal[1] += grad_dotn * rot_l_dir.y;
                            l_grad_normal[2] += grad_dotn * rot_l_dir.z;
                        }
                    }
                    ///// gradient for specular
                    if (enable_specular){
                        float3 h_before = add(rot_l_dir, l_viewdir);
                        float3 h = normalize(h_before);
                        float cosh = dot(rot_l_dir, h);
                        float F = FresnelApproximate(F0, cosh);
                        float G = G1_robust(rot_l_dir, l_viewdir, l_normal, alpha1);
                        float D = NDF1(l_normal, h, alpha0);
                        float tmp = (F * G * D) / 4.f ;
                        float grad_tmp = 0.f;
                        if (tmp > 0){
                            // grad_shading -> grad_specular_albedo, grad_tmp
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float l_intensity = l_envmap[i_channel];
                                l_grad_specular_albedo[i_channel] += l_grad_shading[i_channel] * l_intensity * (tmp * delta_omega);
                                grad_tmp += l_grad_shading[i_channel] * l_intensity * l_specular_albedo[i_channel] * delta_omega;
                            }
                            //// grad_tmp -> grad_F, grad_D, grad_G
                            float grad_F = grad_tmp * (G * D) / 4.f;
                            float grad_G = grad_tmp * (F * D) / 4.f;
                            float grad_D = grad_tmp * (F * G) / 4.f;
                            //// grad_F -> cosh
                            float grad_cosh = grad_FresnelApproximate(F0, cosh, grad_F);
                            float3 grad_h;
                            float3 g_l_dir; // NOTE: Currently not used
                            grad_h.x = grad_cosh * rot_l_dir.x; grad_h.y = grad_cosh * rot_l_dir.y; grad_h.z = grad_cosh * rot_l_dir.z;
                            g_l_dir.x = grad_cosh * h.x; g_l_dir.y = grad_cosh * h.y; g_l_dir.z = grad_cosh * h.z;
                            //// grad_G -> n,l,v,alpha1
                            float3 g_viewdir = make_float3(0.f,0.f,0.f);
                            float3 g_normal = make_float3(0.f,0.f,0.f);
                            grad_G1_robust(
                                rot_l_dir, l_viewdir, l_normal, alpha1, grad_G,
                                g_l_dir, g_viewdir, g_normal, grad_alpha1
                            );
                            //// grad_D -> n,h,alpha0
                            grad_NDF1(
                                l_normal, h, alpha0, grad_D,
                                g_normal, grad_h, grad_alpha0
                            );
                            // h -> h_before
                            grad_h = grad_normalize(h_before, grad_h);
                            // h_before -> v,l
                            g_viewdir = add(g_viewdir, grad_h);
                            g_l_dir = add(g_l_dir, grad_h);
                            // Merge
                            l_grad_viewdir[0] += g_viewdir.x; l_grad_viewdir[1] += g_viewdir.y; l_grad_viewdir[2] += g_viewdir.z;
                            l_grad_normal[0] += g_normal.x; l_grad_normal[1] += g_normal.y; l_grad_normal[2] += g_normal.z;
                            // ? <- g_l_dir
                        }
                    }
                }
            }
        }
    }
    ////////////// store grad_normal
    __syncthreads();
    ssave<256,3>(P, valid, grad_normal, s_buff, l_grad_normal);
    if(enable_diffuse){
        ////////////// store grad_albedo
        __syncthreads();
        ssave<256,C>(P, valid, grad_albedo, s_buff, l_grad_albedo);
    }
    if(enable_specular){
        ////////////// store view_dir
        __syncthreads();
        ssave<256,3>(P, valid, grad_view_dir, s_buff, l_grad_viewdir);
        ////////////// store grad_specular_albedo
        __syncthreads();
        ssave<256,C>(P, valid, grad_specular_albedo, s_buff, l_grad_specular_albedo);
        ////////////// store albedo
        float l_grad_roughness = grad_alpha0 * 4.f * l_roughness * l_roughness * l_roughness;
        l_grad_roughness += grad_alpha1 * (1.f + l_roughness) / 4.f;
        __syncthreads();
        ssave<256,1>(P, valid, grad_roughness, s_buff, &l_grad_roughness);
    }
}

template<int C>
__global__ void __launch_bounds__(256) BruteForceSpecularShaderClampKernel_BackwardEnvmap(
    int64_t P, int H, int W,
    const float * __restrict__ grad_shading, // PxC
    const float * __restrict__ normal, // Px3
    const float * __restrict__ view_dir, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ specular_albedo, // PxC
    const float * __restrict__ roughness, // Px1
    const float * __restrict__ light2obj_rotmat, // 3x3
    float * __restrict__ grad_envmap, // CxHxW
    bool enable_diffuse, bool enable_specular
){
    // 16k
    __shared__ float s_normal[3 * 256]; // 3k
    __shared__ float s_viewdir[3 * 256]; // 3k
    __shared__ float s_albedo[C * 256]; // 3k
    __shared__ float s_specular_albedo[C * 256]; // 3k
    __shared__ float s_roughness[1 * 256]; // 1k
    __shared__ float s_grad[C * 256]; // 3k
    float l_mat[9];
    float3 l_normal; float3 l_viewdir;
    float l_grad_envmap[C];
    /////////////// load mat
    if(threadIdx.x < 9){
        s_normal[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    #pragma unroll
    for(int i = 0; i < 9; i++){
        l_mat[i] = s_normal[i];
    }
    __syncthreads();
    ////////////// load normal & viewdir & albedo & specular_albedo & roughness & gradin
    #pragma unroll
    for(int i = 0; i < 3; i++){
        int loc_offset = i * 256 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256 * 3) + loc_offset;
        if(i_elem < P * 3){
            s_normal[loc_offset] = normal[i_elem];
            s_viewdir[loc_offset] = view_dir[i_elem];
        }
    }
    {
        int loc_offset = threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256) + loc_offset;
        if(i_elem < P){
            s_roughness[loc_offset] = roughness[i_elem];
        }
    }
    #pragma unroll
    for(int i = 0; i < C; i++){
        int loc_offset = i * 256 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256 * C) + loc_offset;
        if(i_elem < P * C){
            if (enable_diffuse)
                s_albedo[loc_offset] = albedo[i_elem];
            if (enable_specular)
                s_specular_albedo[loc_offset] = specular_albedo[i_elem];
            s_grad[loc_offset] = grad_shading[i_elem];
        }
    }
    __syncthreads();
    ////////////// acc grad of each texel
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    for(int i_loop = 0; i_loop < n_loop; i_loop ++){
        int i_texel = i_loop * 256 + threadIdx.x;
        if(i_texel < N_texel){ // check valid texel
            int ix = i_texel % W;
            int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
            iy = H - 1 - iy;
#endif
            float3 l_dir = getLightDir(iy, ix, W, H);
            float delta_omega = getDeltaOmega(iy, W, H);
            float3 rot_l_dir;
            rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
            rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
            rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
            // clear grad
            #pragma unroll
            for(int i = 0; i < C; i++){
                l_grad_envmap[i] = 0;
            }
            for(int i = 0; i < 256; i++){
                if( blockIdx.x * int64_t(256) + i < P){ // check valid fragment
                    l_normal.x = s_normal[i * 3 + 0];
                    l_normal.y = s_normal[i * 3 + 1];
                    l_normal.z = s_normal[i * 3 + 2];

                    l_viewdir.x = s_viewdir[i * 3 + 0];
                    l_viewdir.y = s_viewdir[i * 3 + 1];
                    l_viewdir.z = s_viewdir[i * 3 + 2];

                    float dot_view = dot(l_normal, l_viewdir);

                    if(enable_diffuse && dot_view > 0.f){
                        float cos_shade = dot(rot_l_dir, l_normal);
                        if(cos_shade > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float albedo_ = s_albedo[i * C + i_channel];
                                float grad_ = s_grad[i * C + i_channel];
                                l_grad_envmap[i_channel] += grad_ * albedo_ * (cos_shade / M_PIf) * delta_omega;
                            }
                        }
                    }
                    if (enable_specular && dot_view > 0.f){
                        //float3 l_viewdir;
                        //l_viewdir.x = s_viewdir[i * 3 + 0];
                        //l_viewdir.y = s_viewdir[i * 3 + 1];
                        //l_viewdir.z = s_viewdir[i * 3 + 2];
                        float l_roughness = s_roughness[i];
                        float alpha0 = l_roughness * l_roughness;
                        alpha0 = alpha0 * alpha0;
                        float alpha1 = l_roughness + 1.f;
                        alpha1 = (alpha1 * alpha1) / 8.f;
                        float3 h = add(rot_l_dir, l_viewdir);
                        h = normalize(h);
                        float cosh = dot(rot_l_dir, h);
                        float F = FresnelApproximate(F0, cosh);
                        float G = G1_robust(rot_l_dir, l_viewdir, l_normal, alpha1);
                        float D = NDF1(l_normal, h, alpha0);
                        float tmp = (F * G * D) / 4.f ;
                        if(tmp > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float albedo_ = s_specular_albedo[i * C + i_channel];
                                float grad_ = s_grad[i * C + i_channel];
                                l_grad_envmap[i_channel] += grad_ * albedo_ * tmp * delta_omega;
                            }
                        }
                    }
                }
            }
            // global write
            #pragma unroll
            for(int i_channel = 0; i_channel < C; i_channel++){
                atomicAdd(grad_envmap + (i_channel * N_texel + i_texel), l_grad_envmap[i_channel]);
            }
        }
    }
}

template<int C>
__global__ void __launch_bounds__(256) BruteForceSpecularShaderKernel_Forward2(
    int64_t P, int H, int W,
    const float * __restrict__ normal, // Px3
    const float * __restrict__ view_dir, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ specular_albedo, // PxC
    const float * __restrict__ roughness, // Px1
    const float * __restrict__ envmap, // CxHxW
    const float * __restrict__ light2obj_rotmat, // 3x3
    float * __restrict__ shading,
    float * __restrict__ diffuse_shading,
    bool enable_specular
){
    __shared__ float s_buff[(C < 3 ? 3 : C) * 256]; // 3k
    float l_mat[9];
    float3 l_normal;
    float3 l_viewdir;
    float l_albedo[C];
    float l_specular_albedo[C];
    float l_roughness, alpha0, alpha1;
    float l_shading[C];
    float l_diffuse_shading[C];
    bool valid = (blockIdx.x * int64_t(256) + threadIdx.x  < P);

    /////////////// load mat
    if(threadIdx.x < 9){
        s_buff[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    if(valid){
        #pragma unroll
        for(int i = 0; i < 9; i++){
            l_mat[i] = s_buff[i];
        }
    }
    __syncthreads();
    ////////////// load normal
    {
        float tmp[3];
        sload<256,3>(P, valid, normal, s_buff, tmp);
        l_normal.x = tmp[0]; l_normal.y = tmp[1]; l_normal.z = tmp[2];
    }
    __syncthreads();
    ////////////// load albedo
    sload<256,C>(P, valid, albedo, s_buff, l_albedo);
    __syncthreads();
    ////////////// load specular_albedo & roughness
    if (enable_specular){
        sload<256,C>(P, valid, specular_albedo, s_buff, l_specular_albedo);
        __syncthreads();
        sload<256,1>(P, valid, roughness, s_buff, &l_roughness);
        alpha0 = l_roughness * l_roughness; // alpha = roughness ^ 2
        alpha0 = alpha0 * alpha0; // alpha2 = alpha * alpha
        alpha1 = l_roughness + 1.f;
        alpha1 = (alpha1 * alpha1) / 8.f;
        __syncthreads();
        float tmp[3];
        sload<256,3>(P, valid, view_dir, s_buff, tmp);
        l_viewdir.x = tmp[0]; l_viewdir.y = tmp[1]; l_viewdir.z = tmp[2];
        __syncthreads();
    }
    ////////////// clear shade
    if(valid){
        #pragma unroll
        for(int i = 0; i < C; i++){
            l_shading[i] = 0; l_diffuse_shading[i] = 0;
        }
    }
    ////////////// load light
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    for(int i_loop = 0; i_loop < n_loop; i_loop++) { // process 256 lights each time
        // read global to shared
        {
            int i_texel = i_loop * 256 + threadIdx.x;
            if(i_texel < N_texel){
                #pragma unroll
                for(int i_channel = 0; i_channel < C; i_channel++){
                    s_buff[i_channel * 256 + threadIdx.x] = envmap[i_channel * N_texel + i_texel];
                }
            }
        }
        __syncthreads();
        // shared to local
        if (valid){
            for(int j = 0; j < 256; j++){ // go through each light
                int i_texel = i_loop * 256 + j;
                if(i_texel < N_texel){
                    int ix = i_texel % W;
                    int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
                    iy = H - 1 - iy;
#endif
                    float3 l_dir = getLightDir(iy, ix, W, H);
                    float delta_omega = getDeltaOmega(iy, W, H);
                    float3 rot_l_dir;
                    rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
                    rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
                    rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
                    // load light color
                    float l_envmap[C];
                    #pragma unroll
                    for(int i_channel = 0; i_channel < C; i_channel++){
                        l_envmap[i_channel] = s_buff[i_channel * 256 + j];
                    }
                    { // diffuse
                        float cos_shade = dot(rot_l_dir, l_normal);
                        //if(cos_shade < 0) cos_shade = 0;
                        if(cos_shade > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float tmp = l_envmap[i_channel] * (cos_shade/ M_PIf) * delta_omega;
                                //l_shading[i_channel] += l_envmap[i_channel] * l_albedo[i_channel] * (cos_shade/ M_PIf) * delta_omega;
                                l_diffuse_shading[i_channel] += tmp;
                                l_shading[i_channel] += tmp * l_albedo[i_channel];
                            }
                        }
                    }
                    if (enable_specular){
                        float3 h = add(rot_l_dir, l_viewdir);
                        h = normalize(h);
                        float cosh = dot(rot_l_dir, h);
                        float F = FresnelApproximate(F0, cosh);
                        float G = G1_robust(rot_l_dir, l_viewdir, l_normal, alpha1);
                        float D = NDF1(l_normal, h, alpha0);
                        float tmp = (F * G * D) / 4.f ;
                        //tmp = max(tmp, 0.);
                        if(tmp > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                l_shading[i_channel] += l_envmap[i_channel] * l_specular_albedo[i_channel] * (tmp * delta_omega);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    //__syncthreads();
    ////////////// store shading
    ssave<256,C>(P, valid, shading, s_buff, l_shading);
    __syncthreads();
    ssave<256,C>(P, valid, diffuse_shading, s_buff, l_diffuse_shading);
}

template<int C>
__global__ void __launch_bounds__(256) BruteForceSpecularShaderKernel_Backward2(
    int64_t P, int H, int W,
    const float * __restrict__ grad_shading, // PxC
    const float * __restrict__ grad_diffuse_shading, // PxC
    const float * __restrict__ normal, // Px3
    const float * __restrict__ view_dir, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ specular_albedo, // PxC
    const float * __restrict__ roughness, // Px1
    const float * __restrict__ envmap, // CxHxW
    const float * __restrict__ light2obj_rotmat, // 3x3
    float * __restrict__ grad_normal, // Px3
    float * __restrict__ grad_view_dir, // Px3
    float * __restrict__ grad_albedo, // PxC
    float * __restrict__ grad_specular_albedo, // PxC
    float * __restrict__ grad_roughness, // Px1
    bool enable_specular
){
    __shared__ float s_buff[(C < 3 ? 3 : C) * 256]; // 3k
    float l_mat[9];
    float3 l_normal;
    float3 l_viewdir;
    float l_albedo[C];
    float l_specular_albedo[C];
    float l_roughness, alpha0, alpha1;
    //
    float l_grad_shading[C]; float l_grad_diffuse_shading[C];
    float l_grad_normal[3];
    float l_grad_viewdir[3];
    float l_grad_albedo[C];
    float l_grad_specular_albedo[C];
    float grad_alpha0, grad_alpha1;

    bool valid = (blockIdx.x * int64_t(256) + threadIdx.x  < P);

    /////////////// load mat
    if(threadIdx.x < 9){
        s_buff[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    if(valid){
        #pragma unroll
        for(int i = 0; i < 9; i++){
            l_mat[i] = s_buff[i];
        }
    }
    __syncthreads();
    ////////////// load normal
    {
        float tmp[3];
        sload<256,3>(P,valid,normal,s_buff,tmp);
        l_normal.x = tmp[0]; l_normal.y = tmp[1]; l_normal.z = tmp[2];
    }
    __syncthreads();
    ////////////// load albedo
    sload<256,C>(P, valid, albedo, s_buff, l_albedo);
    __syncthreads();
    ////////////// load specular_albedo & roughness
    if (enable_specular){
        sload<256,C>(P, valid, specular_albedo, s_buff, l_specular_albedo);
        __syncthreads();
        sload<256,1>(P, valid, roughness, s_buff, &l_roughness);
        alpha0 = l_roughness * l_roughness; // alpha = roughness ^ 2
        alpha0 = alpha0 * alpha0; // alpha2 = alpha * alpha
        alpha1 = l_roughness + 1.f;
        alpha1 = (alpha1 * alpha1) / 8.f;
        __syncthreads();
        float tmp[3];
        sload<256,3>(P, valid, view_dir, s_buff, tmp);
        l_viewdir.x = tmp[0]; l_viewdir.y = tmp[1]; l_viewdir.z = tmp[2];
        __syncthreads();
    }
    ////////////// load grad_shading
    sload<256,C>(P, valid, grad_shading, s_buff, l_grad_shading);
    __syncthreads();
    sload<256,C>(P, valid, grad_diffuse_shading, s_buff, l_grad_diffuse_shading);
    //__syncthreads();
    ////////////// clear shade
    if(valid){
        l_grad_normal[0] = 0; l_grad_normal[1] = 0; l_grad_normal[2] = 0;
        { // diffuse
            #pragma unroll
            for(int i = 0; i < C ; i++){
                l_grad_albedo[i] = 0;
            }
        }
        if (enable_specular){
            l_grad_viewdir[0] = 0; l_grad_viewdir[1] = 0; l_grad_viewdir[2] = 0;
            grad_alpha0 = 0; grad_alpha1 = 0;
            #pragma unroll
            for(int i = 0; i < C ; i++){
                l_grad_specular_albedo[i] = 0;
            }
        }
    }
    ////////////// load light
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    for(int i_loop = 0; i_loop < n_loop; i_loop++){ // process 256 lights each time
        __syncthreads();
        // read global to shared
        {
            int i_texel = i_loop * 256 + threadIdx.x;
            if(i_texel < N_texel){
                #pragma unroll
                for(int i_channel = 0; i_channel < C; i_channel++){
                    s_buff[i_channel * 256 + threadIdx.x] = envmap[i_channel * N_texel + i_texel];
                }
            }
        }
        __syncthreads();
        // shared to local
        if (valid){
            for(int j = 0; j < 256; j++){ // go through each light
                int i_texel = i_loop * 256 + j;
                if(i_texel < N_texel){
                    int ix = i_texel % W;
                    int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
                    iy = H - 1 - iy;
#endif
                    float3 l_dir = getLightDir(iy, ix, W, H);
                    float delta_omega = getDeltaOmega(iy, W, H);
                    float3 rot_l_dir;
                    rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
                    rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
                    rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
                    // load light color
                    float l_envmap[C];
                    #pragma unroll
                    for(int i_channel = 0; i_channel < C; i_channel++){
                        l_envmap[i_channel] = s_buff[i_channel * 256 + j];
                    }
                    ///// gradient for diffuse
                    {
                        float cos_shade = dot(rot_l_dir, l_normal);
                        float grad_dotn = 0.f;
                        if(cos_shade > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float l_intensity = l_envmap[i_channel];
                                l_grad_albedo[i_channel] += l_grad_shading[i_channel] * l_intensity * (cos_shade / M_PIf) * delta_omega;
                                //grad_dotn += l_grad_shading[i_channel] * l_intensity *  l_albedo[i_channel] * delta_omega / M_PIf;
                                grad_dotn += (l_grad_shading[i_channel] * l_albedo[i_channel] + l_grad_diffuse_shading[i_channel]) * l_intensity * delta_omega / M_PIf;
                            }
                            l_grad_normal[0] += grad_dotn * rot_l_dir.x;
                            l_grad_normal[1] += grad_dotn * rot_l_dir.y;
                            l_grad_normal[2] += grad_dotn * rot_l_dir.z;
                        }
                    }
                    ///// gradient for specular
                    if (enable_specular){
                        float3 h_before = add(rot_l_dir, l_viewdir);
                        float3 h = normalize(h_before);
                        float cosh = dot(rot_l_dir, h);
                        float F = FresnelApproximate(F0, cosh);
                        float G = G1_robust(rot_l_dir, l_viewdir, l_normal, alpha1);
                        float D = NDF1(l_normal, h, alpha0);
                        float tmp = (F * G * D) / 4.f ;
                        float grad_tmp = 0.f;
                        if (tmp > 0){
                            // grad_shading -> grad_specular_albedo, grad_tmp
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float l_intensity = l_envmap[i_channel];
                                l_grad_specular_albedo[i_channel] += l_grad_shading[i_channel] * l_intensity * (tmp * delta_omega);
                                grad_tmp += l_grad_shading[i_channel] * l_intensity * l_specular_albedo[i_channel] * delta_omega;
                            }
                            //// grad_tmp -> grad_F, grad_D, grad_G
                            float grad_F = grad_tmp * (G * D) / 4.f;
                            float grad_G = grad_tmp * (F * D) / 4.f;
                            float grad_D = grad_tmp * (F * G) / 4.f;
                            //// grad_F -> cosh
                            float grad_cosh = grad_FresnelApproximate(F0, cosh, grad_F);
                            float3 grad_h;
                            float3 g_l_dir; // NOTE: Currently not used
                            grad_h.x = grad_cosh * rot_l_dir.x; grad_h.y = grad_cosh * rot_l_dir.y; grad_h.z = grad_cosh * rot_l_dir.z;
                            g_l_dir.x = grad_cosh * h.x; g_l_dir.y = grad_cosh * h.y; g_l_dir.z = grad_cosh * h.z;
                            //// grad_G -> n,l,v,alpha1
                            float3 g_viewdir = make_float3(0.f,0.f,0.f);
                            float3 g_normal = make_float3(0.f,0.f,0.f);
                            grad_G1_robust(
                                rot_l_dir, l_viewdir, l_normal, alpha1, grad_G,
                                g_l_dir, g_viewdir, g_normal, grad_alpha1
                            );
                            //// grad_D -> n,h,alpha0
                            grad_NDF1(
                                l_normal, h, alpha0, grad_D,
                                g_normal, grad_h, grad_alpha0
                            );
                            // h -> h_before
                            grad_h = grad_normalize(h_before, grad_h);
                            // h_before -> v,l
                            g_viewdir = add(g_viewdir, grad_h);
                            g_l_dir = add(g_l_dir, grad_h);
                            // Merge
                            l_grad_viewdir[0] += g_viewdir.x; l_grad_viewdir[1] += g_viewdir.y; l_grad_viewdir[2] += g_viewdir.z;
                            l_grad_normal[0] += g_normal.x; l_grad_normal[1] += g_normal.y; l_grad_normal[2] += g_normal.z;
                            // ? <- g_l_dir
                        }
                    }
                }
            }
        }
    }
    ////////////// store grad_normal
    __syncthreads();
    ssave<256,3>(P, valid, grad_normal, s_buff, l_grad_normal);
    ////////////// store grad_albedo
    __syncthreads();
    ssave<256,C>(P, valid, grad_albedo, s_buff, l_grad_albedo);
    if(enable_specular){
        ////////////// store view_dir
        __syncthreads();
        ssave<256,3>(P, valid, grad_view_dir, s_buff, l_grad_viewdir);
        ////////////// store grad_specular_albedo
        __syncthreads();
        ssave<256,C>(P, valid, grad_specular_albedo, s_buff, l_grad_specular_albedo);
        ////////////// store albedo
        float l_grad_roughness = grad_alpha0 * 4.f * l_roughness * l_roughness * l_roughness;
        l_grad_roughness += grad_alpha1 * (1.f + l_roughness) / 4.f;
        __syncthreads();
        ssave<256,1>(P, valid, grad_roughness, s_buff, &l_grad_roughness);
    }
}

template<int C>
__global__ void __launch_bounds__(256) BruteForceSpecularShaderKernel_BackwardEnvmap2(
    int64_t P, int H, int W,
    const float * __restrict__ grad_shading, // PxC
    const float * __restrict__ grad_diffuse_shading, // PxC
    const float * __restrict__ normal, // Px3
    const float * __restrict__ view_dir, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ specular_albedo, // PxC
    const float * __restrict__ roughness, // Px1
    const float * __restrict__ light2obj_rotmat, // 3x3
    float * __restrict__ grad_envmap, // CxHxW
    bool enable_specular
){
    // 13k
    __shared__ float s_normal[3 * 256]; // 3k
    __shared__ float s_viewdir[3 * 256]; // 3k
    __shared__ float s_diffuse_grad[C * 256]; // 3k
    __shared__ float s_specular_grad[C * 256]; // 3k
    __shared__ float s_roughness[1 * 256]; // 1k
    float l_mat[9];
    float3 l_normal;
    float l_grad_envmap[C];
    /////////////// load mat
    if(threadIdx.x < 9){
        s_normal[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    #pragma unroll
    for(int i = 0; i < 9; i++){
        l_mat[i] = s_normal[i];
    }
    __syncthreads();
    ////////////// load normal & viewdir & albedo & specular_albedo & roughness & gradin
    #pragma unroll
    for(int i = 0; i < 3; i++){
        int loc_offset = i * 256 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256 * 3) + loc_offset;
        if(i_elem < P * 3){
            s_normal[loc_offset] = normal[i_elem];
            if (enable_specular)
                s_viewdir[loc_offset] = view_dir[i_elem];
        }
    }
    {
        int loc_offset = threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256) + loc_offset;
        if(i_elem < P){
            s_roughness[loc_offset] = roughness[i_elem];
        }
    }
    #pragma unroll
    for(int i = 0; i < C; i++){
        int loc_offset = i * 256 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256 * C) + loc_offset;
        if(i_elem < P * C){
            float l_grad_shading = grad_shading[i_elem];
            float l_grad_diffuse_shading = grad_diffuse_shading[i_elem];
            float l_albedo = albedo[i_elem];
            s_diffuse_grad[loc_offset] = l_grad_shading * l_albedo + l_grad_diffuse_shading;
            if (enable_specular){
                float l_specular_albedo = specular_albedo[i_elem];
                s_specular_grad[loc_offset] = l_grad_shading * l_specular_albedo;
            }
        }
    }
    __syncthreads();
    ////////////// acc grad of each texel
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    for(int i_loop = 0; i_loop < n_loop; i_loop ++){
        int i_texel = i_loop * 256 + threadIdx.x;
        if(i_texel < N_texel){ // check valid texel
            int ix = i_texel % W;
            int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
            iy = H - 1 - iy;
#endif
            float3 l_dir = getLightDir(iy, ix, W, H);
            float delta_omega = getDeltaOmega(iy, W, H);
            float3 rot_l_dir;
            rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
            rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
            rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
            // clear grad
            #pragma unroll
            for(int i = 0; i < C; i++){
                l_grad_envmap[i] = 0;
            }
            for(int i = 0; i < 256; i++){
                if( blockIdx.x * int64_t(256) + i < P){ // check valid fragment
                    l_normal.x = s_normal[i * 3 + 0];
                    l_normal.y = s_normal[i * 3 + 1];
                    l_normal.z = s_normal[i * 3 + 2];
                    { // diffuse
                        float cos_shade = dot(rot_l_dir, l_normal);
                        if(cos_shade > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float diffuse_grad = s_diffuse_grad[i * C + i_channel];
                                l_grad_envmap[i_channel] += diffuse_grad * (cos_shade / M_PIf) * delta_omega;
                            }
                        }
                    }
                    if (enable_specular){
                        float3 l_viewdir;
                        l_viewdir.x = s_viewdir[i * 3 + 0];
                        l_viewdir.y = s_viewdir[i * 3 + 1];
                        l_viewdir.z = s_viewdir[i * 3 + 2];
                        float l_roughness = s_roughness[i];
                        float alpha0 = l_roughness * l_roughness;
                        alpha0 = alpha0 * alpha0;
                        float alpha1 = l_roughness + 1.f;
                        alpha1 = (alpha1 * alpha1) / 8.f;
                        float3 h = add(rot_l_dir, l_viewdir);
                        h = normalize(h);
                        float cosh = dot(rot_l_dir, h);
                        float F = FresnelApproximate(F0, cosh);
                        float G = G1_robust(rot_l_dir, l_viewdir, l_normal, alpha1);
                        float D = NDF1(l_normal, h, alpha0);
                        float tmp = (F * G * D) / 4.f ;
                        if(tmp > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float specular_grad = s_specular_grad[i * C + i_channel];
                                l_grad_envmap[i_channel] += specular_grad * tmp * delta_omega;
                            }
                        }
                    }
                }
            }
            // global write
            #pragma unroll
            for(int i_channel = 0; i_channel < C; i_channel++){
                atomicAdd(grad_envmap + (i_channel * N_texel + i_texel), l_grad_envmap[i_channel]);
            }
        }
    }
}

template<int C>
__global__ void __launch_bounds__(256) BruteForceSpecularVisibilityShaderKernel_Forward2(
    int64_t P, int H, int W, int N_face, int N_vertex,
    const float * __restrict__ normal, // Px3
    const float * __restrict__ view_dir, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ specular_albedo, // PxC
    const float * __restrict__ roughness, // Px1
    const float * __restrict__ envmap, // CxHxW
    const float * __restrict__ light2obj_rotmat, // 3x3
    //
    const int * __restrict__ face_vertex_list, // 3xN_face
    const int64_t * __restrict__ nearest_triangle_id, // P
    const float * __restrict__ barycentric_coord, // 3xP
    const uint32_t * __restrict__ visibility, // n_slot x N_vertex
    //
    float * __restrict__ shading,
    float * __restrict__ diffuse_shading,
    bool enable_specular
){
    __shared__ float s_buff[(C < 3 ? 3 : C) * 256]; // 3k
    float l_mat[9];
    float3 l_normal;
    float3 l_viewdir;
    float l_albedo[C];
    float l_specular_albedo[C];
    float l_roughness, alpha0, alpha1;
    float l_shading[C];
    float l_diffuse_shading[C];
    int64_t id = blockIdx.x * int64_t(256) + threadIdx.x;
    bool valid = (id < P);
    /////////////// load mat
    if(threadIdx.x < 9){
        s_buff[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    if(valid){
        #pragma unroll
        for(int i = 0; i < 9; i++){
            l_mat[i] = s_buff[i];
        }
    }
    __syncthreads();
    ////////////// load normal
    {
        float tmp[3];
        sload<256,3>(P, valid, normal, s_buff, tmp);
        l_normal.x = tmp[0]; l_normal.y = tmp[1]; l_normal.z = tmp[2];
    }
    __syncthreads();
    ////////////// load albedo
    sload<256,C>(P, valid, albedo, s_buff, l_albedo);
    __syncthreads();
    ////////////// load specular_albedo & roughness
    if (enable_specular){
        sload<256,C>(P, valid, specular_albedo, s_buff, l_specular_albedo);
        __syncthreads();
        sload<256,1>(P, valid, roughness, s_buff, &l_roughness);
        alpha0 = l_roughness * l_roughness; // alpha = roughness ^ 2
        alpha0 = alpha0 * alpha0; // alpha2 = alpha * alpha
        alpha1 = l_roughness + 1.f;
        alpha1 = (alpha1 * alpha1) / 8.f;
        __syncthreads();
        float tmp[3];
        sload<256,3>(P, valid, view_dir, s_buff, tmp);
        l_viewdir.x = tmp[0]; l_viewdir.y = tmp[1]; l_viewdir.z = tmp[2];
        __syncthreads();
    }
    ////////////// clear shade & load visibility
    int64_t nf_id; int v0, v1, v2;
    uint32_t p0, p1, p2;
    float c0, c1, c2;
    if(valid){
        #pragma unroll
        for(int i = 0; i < C; i++){
            l_shading[i] = 0; l_diffuse_shading[i] = 0;
        }
        nf_id = nearest_triangle_id[id];
        v0 = face_vertex_list[nf_id];
        v1 = face_vertex_list[nf_id + N_face];
        v2 = face_vertex_list[nf_id + N_face + N_face];
        c0 = barycentric_coord[id];
        c1 = barycentric_coord[id + P];
        c2 = barycentric_coord[id + P + P];
    }
    ////////////// load light
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    int i_slot = 0;
    for(int i_loop = 0; i_loop < n_loop; i_loop++) { // process 256 lights each time
        // read global to shared
        {
            int i_texel = i_loop * 256 + threadIdx.x;
            if(i_texel < N_texel){
                #pragma unroll
                for(int i_channel = 0; i_channel < C; i_channel++){
                    s_buff[i_channel * 256 + threadIdx.x] = envmap[i_channel * N_texel + i_texel];
                }
            }
        }
        __syncthreads();
        // shared to local
        if (valid){
            for(int j = 0; j < 256; j++){ // go through each light
                int i_texel = i_loop * 256 + j;
                if(i_texel < N_texel){
                    int ix = i_texel % W;
                    int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
                    iy = H - 1 - iy;
#endif
                    // compute visibility
                    if(i_texel % 32 == 0){
                        p0 = visibility[i_slot * N_vertex + v0];
                        p1 = visibility[i_slot * N_vertex + v1];
                        p2 = visibility[i_slot * N_vertex + v2]; 
                        i_slot++;
                    }
                    uint32_t bit_disp = (0x01 << (i_texel % 32));
                    float vis_0 = (p0 & bit_disp) ? 1.f : 0.f;
                    float vis_1 = (p1 & bit_disp) ? 1.f : 0.f;
                    float vis_2 = (p2 & bit_disp) ? 1.f : 0.f;
                    float vis_ = vis_0 * c0 + vis_1 * c1 + vis_2 * c2;
                    // 
                    float3 l_dir = getLightDir(iy, ix, W, H);
                    float delta_omega = getDeltaOmega(iy, W, H);
                    float3 rot_l_dir;
                    rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
                    rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
                    rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
                    // load light color
                    float l_envmap[C];
                    #pragma unroll
                    for(int i_channel = 0; i_channel < C; i_channel++){
                        l_envmap[i_channel] = s_buff[i_channel * 256 + j];
                    }
                    { // diffuse
                        float cos_shade = dot(rot_l_dir, l_normal);
                        //if(cos_shade < 0) cos_shade = 0;
                        if(cos_shade > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float tmp = l_envmap[i_channel] * (cos_shade/ M_PIf) * delta_omega;
                                tmp *= vis_;
                                //l_shading[i_channel] += l_envmap[i_channel] * l_albedo[i_channel] * (cos_shade/ M_PIf) * delta_omega;
                                l_diffuse_shading[i_channel] += tmp;
                                l_shading[i_channel] += tmp * l_albedo[i_channel];
                            }
                        }
                    }
                    if (enable_specular){
                        float3 h = add(rot_l_dir, l_viewdir);
                        h = normalize(h);
                        float cosh = dot(rot_l_dir, h);
                        float F = FresnelApproximate(F0, cosh);
                        float G = G1_robust(rot_l_dir, l_viewdir, l_normal, alpha1);
                        float D = NDF1(l_normal, h, alpha0);
                        float tmp = (F * G * D) / 4.f ;
                        //tmp = max(tmp, 0.);
                        if(tmp > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                //l_shading[i_channel] += l_envmap[i_channel] * l_specular_albedo[i_channel] * (tmp * delta_omega);
                                l_shading[i_channel] += l_envmap[i_channel] * l_specular_albedo[i_channel] * (tmp * delta_omega * vis_);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    //__syncthreads();
    ////////////// store shading
    ssave<256,C>(P, valid, shading, s_buff, l_shading);
    __syncthreads();
    ssave<256,C>(P, valid, diffuse_shading, s_buff, l_diffuse_shading);
}


template<int C>
__global__ void __launch_bounds__(256) BruteForceSpecularVisibilityShaderKernel_Backward2(
    int64_t P, int H, int W, int N_face, int N_vertex,   
    const float * __restrict__ grad_shading, // PxC
    const float * __restrict__ grad_diffuse_shading, // PxC
    const float * __restrict__ normal, // Px3
    const float * __restrict__ view_dir, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ specular_albedo, // PxC
    const float * __restrict__ roughness, // Px1
    const float * __restrict__ envmap, // CxHxW
    const float * __restrict__ light2obj_rotmat, // 3x3
    //
    const int * __restrict__ face_vertex_list, // 3xN_face
    const int64_t * __restrict__ nearest_triangle_id, // P
    const float * __restrict__ barycentric_coord, // 3xP
    const uint32_t * __restrict__ visibility, // n_slot x N_vertex
    //
    float * __restrict__ grad_normal, // Px3
    float * __restrict__ grad_view_dir, // Px3
    float * __restrict__ grad_albedo, // PxC
    float * __restrict__ grad_specular_albedo, // PxC
    float * __restrict__ grad_roughness, // Px1
    bool enable_specular
){
    __shared__ float s_buff[(C < 3 ? 3 : C) * 256]; // 3k
    float l_mat[9];
    float3 l_normal;
    float3 l_viewdir;
    float l_albedo[C];
    float l_specular_albedo[C];
    float l_roughness, alpha0, alpha1;
    //
    float l_grad_shading[C]; float l_grad_diffuse_shading[C];
    float l_grad_normal[3];
    float l_grad_viewdir[3];
    float l_grad_albedo[C];
    float l_grad_specular_albedo[C];
    float grad_alpha0, grad_alpha1;

    int64_t id = blockIdx.x * int64_t(256) + threadIdx.x;
    bool valid = (id < P);

    /////////////// load mat
    if(threadIdx.x < 9){
        s_buff[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    if(valid){
        #pragma unroll
        for(int i = 0; i < 9; i++){
            l_mat[i] = s_buff[i];
        }
    }
    __syncthreads();
    ////////////// load normal
    {
        float tmp[3];
        sload<256,3>(P,valid,normal,s_buff,tmp);
        l_normal.x = tmp[0]; l_normal.y = tmp[1]; l_normal.z = tmp[2];
    }
    __syncthreads();
    ////////////// load albedo
    sload<256,C>(P, valid, albedo, s_buff, l_albedo);
    __syncthreads();
    ////////////// load specular_albedo & roughness
    if (enable_specular){
        sload<256,C>(P, valid, specular_albedo, s_buff, l_specular_albedo);
        __syncthreads();
        sload<256,1>(P, valid, roughness, s_buff, &l_roughness);
        alpha0 = l_roughness * l_roughness; // alpha = roughness ^ 2
        alpha0 = alpha0 * alpha0; // alpha2 = alpha * alpha
        alpha1 = l_roughness + 1.f;
        alpha1 = (alpha1 * alpha1) / 8.f;
        __syncthreads();
        float tmp[3];
        sload<256,3>(P, valid, view_dir, s_buff, tmp);
        l_viewdir.x = tmp[0]; l_viewdir.y = tmp[1]; l_viewdir.z = tmp[2];
        __syncthreads();
    }
    ////////////// load grad_shading
    sload<256,C>(P, valid, grad_shading, s_buff, l_grad_shading);
    __syncthreads();
    sload<256,C>(P, valid, grad_diffuse_shading, s_buff, l_grad_diffuse_shading);
    //__syncthreads();
    ////////////// clear shade
    int64_t nf_id; int v0, v1, v2;
    uint32_t p0, p1, p2;
    float c0, c1, c2;
    if(valid){
        l_grad_normal[0] = 0; l_grad_normal[1] = 0; l_grad_normal[2] = 0;
        { // diffuse
            #pragma unroll
            for(int i = 0; i < C ; i++){
                l_grad_albedo[i] = 0;
            }
        }
        if (enable_specular){
            l_grad_viewdir[0] = 0; l_grad_viewdir[1] = 0; l_grad_viewdir[2] = 0;
            grad_alpha0 = 0; grad_alpha1 = 0;
            #pragma unroll
            for(int i = 0; i < C ; i++){
                l_grad_specular_albedo[i] = 0;
            }
        }
        nf_id = nearest_triangle_id[id];
        v0 = face_vertex_list[nf_id];
        v1 = face_vertex_list[nf_id + N_face];
        v2 = face_vertex_list[nf_id + N_face + N_face];
        c0 = barycentric_coord[id];
        c1 = barycentric_coord[id + P];
        c2 = barycentric_coord[id + P + P];
    }
    ////////////// load light
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    int i_slot = 0;
    for(int i_loop = 0; i_loop < n_loop; i_loop++){ // process 256 lights each time
        __syncthreads();
        // read global to shared
        {
            int i_texel = i_loop * 256 + threadIdx.x;
            if(i_texel < N_texel){
                #pragma unroll
                for(int i_channel = 0; i_channel < C; i_channel++){
                    s_buff[i_channel * 256 + threadIdx.x] = envmap[i_channel * N_texel + i_texel];
                }
            }
        }
        __syncthreads();
        // shared to local
        if (valid){
            for(int j = 0; j < 256; j++){ // go through each light
                int i_texel = i_loop * 256 + j;
                if(i_texel < N_texel){
                    int ix = i_texel % W;
                    int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
                    iy = H - 1 - iy;
#endif
                    // compute visibility
                    if(i_texel % 32 == 0){
                        p0 = visibility[i_slot * N_vertex + v0];
                        p1 = visibility[i_slot * N_vertex + v1];
                        p2 = visibility[i_slot * N_vertex + v2]; 
                        i_slot++;
                    }
                    uint32_t bit_disp = (0x01 << (i_texel % 32));
                    float vis_0 = (p0 & bit_disp) ? 1.f : 0.f;
                    float vis_1 = (p1 & bit_disp) ? 1.f : 0.f;
                    float vis_2 = (p2 & bit_disp) ? 1.f : 0.f;
                    float vis_ = vis_0 * c0 + vis_1 * c1 + vis_2 * c2;
                    // 
                    float3 l_dir = getLightDir(iy, ix, W, H);
                    float delta_omega = getDeltaOmega(iy, W, H);
                    float3 rot_l_dir;
                    rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
                    rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
                    rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
                    // load light color
                    float l_envmap[C];
                    #pragma unroll
                    for(int i_channel = 0; i_channel < C; i_channel++){
                        l_envmap[i_channel] = s_buff[i_channel * 256 + j];
                    }
                    ///// gradient for diffuse
                    {
                        float cos_shade = dot(rot_l_dir, l_normal);
                        float grad_dotn = 0.f;
                        if(cos_shade > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float l_intensity = l_envmap[i_channel];
                                l_intensity *= vis_;
                                l_grad_albedo[i_channel] += l_grad_shading[i_channel] * l_intensity * (cos_shade / M_PIf) * delta_omega;
                                //grad_dotn += l_grad_shading[i_channel] * l_intensity *  l_albedo[i_channel] * delta_omega / M_PIf;
                                grad_dotn += (l_grad_shading[i_channel] * l_albedo[i_channel] + l_grad_diffuse_shading[i_channel]) * l_intensity * delta_omega / M_PIf;
                            }
                            l_grad_normal[0] += grad_dotn * rot_l_dir.x;
                            l_grad_normal[1] += grad_dotn * rot_l_dir.y;
                            l_grad_normal[2] += grad_dotn * rot_l_dir.z;
                        }
                    }
                    ///// gradient for specular
                    if (enable_specular){
                        float3 h_before = add(rot_l_dir, l_viewdir);
                        float3 h = normalize(h_before);
                        float cosh = dot(rot_l_dir, h);
                        float F = FresnelApproximate(F0, cosh);
                        float G = G1_robust(rot_l_dir, l_viewdir, l_normal, alpha1);
                        float D = NDF1(l_normal, h, alpha0);
                        float tmp = (F * G * D) / 4.f ;
                        float grad_tmp = 0.f;
                        if (tmp > 0){
                            // grad_shading -> grad_specular_albedo, grad_tmp
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float l_intensity = l_envmap[i_channel];
                                l_intensity *= vis_;
                                l_grad_specular_albedo[i_channel] += l_grad_shading[i_channel] * l_intensity * (tmp * delta_omega);
                                grad_tmp += l_grad_shading[i_channel] * l_intensity * l_specular_albedo[i_channel] * delta_omega;
                            }
                            //// grad_tmp -> grad_F, grad_D, grad_G
                            float grad_F = grad_tmp * (G * D) / 4.f;
                            float grad_G = grad_tmp * (F * D) / 4.f;
                            float grad_D = grad_tmp * (F * G) / 4.f;
                            //// grad_F -> cosh
                            float grad_cosh = grad_FresnelApproximate(F0, cosh, grad_F);
                            float3 grad_h;
                            float3 g_l_dir; // NOTE: Currently not used
                            grad_h.x = grad_cosh * rot_l_dir.x; grad_h.y = grad_cosh * rot_l_dir.y; grad_h.z = grad_cosh * rot_l_dir.z;
                            g_l_dir.x = grad_cosh * h.x; g_l_dir.y = grad_cosh * h.y; g_l_dir.z = grad_cosh * h.z;
                            //// grad_G -> n,l,v,alpha1
                            float3 g_viewdir = make_float3(0.f,0.f,0.f);
                            float3 g_normal = make_float3(0.f,0.f,0.f);
                            grad_G1_robust(
                                rot_l_dir, l_viewdir, l_normal, alpha1, grad_G,
                                g_l_dir, g_viewdir, g_normal, grad_alpha1
                            );
                            //// grad_D -> n,h,alpha0
                            grad_NDF1(
                                l_normal, h, alpha0, grad_D,
                                g_normal, grad_h, grad_alpha0
                            );
                            // h -> h_before
                            grad_h = grad_normalize(h_before, grad_h);
                            // h_before -> v,l
                            g_viewdir = add(g_viewdir, grad_h);
                            g_l_dir = add(g_l_dir, grad_h);
                            // Merge
                            l_grad_viewdir[0] += g_viewdir.x; l_grad_viewdir[1] += g_viewdir.y; l_grad_viewdir[2] += g_viewdir.z;
                            l_grad_normal[0] += g_normal.x; l_grad_normal[1] += g_normal.y; l_grad_normal[2] += g_normal.z;
                            // ? <- g_l_dir
                        }
                    }
                }
            }
        }
    }
    ////////////// store grad_normal
    __syncthreads();
    ssave<256,3>(P, valid, grad_normal, s_buff, l_grad_normal);
    ////////////// store grad_albedo
    __syncthreads();
    ssave<256,C>(P, valid, grad_albedo, s_buff, l_grad_albedo);
    if(enable_specular){
        ////////////// store view_dir
        __syncthreads();
        ssave<256,3>(P, valid, grad_view_dir, s_buff, l_grad_viewdir);
        ////////////// store grad_specular_albedo
        __syncthreads();
        ssave<256,C>(P, valid, grad_specular_albedo, s_buff, l_grad_specular_albedo);
        ////////////// store albedo
        float l_grad_roughness = grad_alpha0 * 4.f * l_roughness * l_roughness * l_roughness;
        l_grad_roughness += grad_alpha1 * (1.f + l_roughness) / 4.f;
        __syncthreads();
        ssave<256,1>(P, valid, grad_roughness, s_buff, &l_grad_roughness);
    }
}

template<int C>
__global__ void __launch_bounds__(256) BruteForceSpecularVisibilityShaderKernel_BackwardEnvmap2(
    int64_t P, int H, int W, int N_face, int N_vertex,
    const float * __restrict__ grad_shading, // PxC
    const float * __restrict__ grad_diffuse_shading, // PxC
    const float * __restrict__ normal, // Px3
    const float * __restrict__ view_dir, // Px3
    const float * __restrict__ albedo, // PxC
    const float * __restrict__ specular_albedo, // PxC
    const float * __restrict__ roughness, // Px1
    const float * __restrict__ light2obj_rotmat, // 3x3
    //
    const int * __restrict__ face_vertex_list, // 3xN_face
    const int64_t * __restrict__ nearest_triangle_id, // P
    const float * __restrict__ barycentric_coord, // 3xP
    const uint32_t * __restrict__ visibility, // n_slot x N_vertex
    //
    float * __restrict__ grad_envmap, // CxHxW
    bool enable_specular
){
    // 19k
    __shared__ float s_normal[3 * 256]; // 3k
    __shared__ float s_viewdir[3 * 256]; // 3k
    __shared__ float s_diffuse_grad[C * 256]; // 3k
    __shared__ float s_specular_grad[C * 256]; // 3k
    __shared__ float s_roughness[1 * 256]; // 1k
    __shared__ float s_c[3 * 256]; // 3k
    __shared__ int s_v[3 * 256]; // 3k

    float l_mat[9];
    float3 l_normal;
    float l_grad_envmap[C];
    /////////////// load mat
    if(threadIdx.x < 9){
        s_normal[threadIdx.x] = light2obj_rotmat[threadIdx.x];
    }
    __syncthreads();
    #pragma unroll
    for(int i = 0; i < 9; i++){
        l_mat[i] = s_normal[i];
    }
    __syncthreads();
    ////////////// load normal & viewdir & albedo & specular_albedo & roughness & gradin
    ////////////// load visibility
    #pragma unroll
    for(int i = 0; i < 3; i++){
        int loc_offset = i * 256 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256 * 3) + loc_offset;
        if(i_elem < P * 3){
            s_normal[loc_offset] = normal[i_elem];
            if (enable_specular)
                s_viewdir[loc_offset] = view_dir[i_elem];
        }
    }
    {
        int loc_offset = threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256) + loc_offset;
        if(i_elem < P){
            s_roughness[loc_offset] = roughness[i_elem];
            int64_t nf_id = nearest_triangle_id[i_elem];
            s_v[loc_offset * 3 + 0] = face_vertex_list[nf_id];
            s_v[loc_offset * 3 + 1] = face_vertex_list[nf_id + N_face];
            s_v[loc_offset * 3 + 2] = face_vertex_list[nf_id + N_face + N_face];

            s_c[loc_offset * 3 + 0] = barycentric_coord[i_elem];
            s_c[loc_offset * 3 + 1] = barycentric_coord[i_elem + P];
            s_c[loc_offset * 3 + 2] = barycentric_coord[i_elem + P + P];
        }
    }
    #pragma unroll
    for(int i = 0; i < C; i++){
        int loc_offset = i * 256 + threadIdx.x;
        int64_t i_elem = blockIdx.x * (int64_t)(256 * C) + loc_offset;
        if(i_elem < P * C){
            float l_grad_shading = grad_shading[i_elem];
            float l_grad_diffuse_shading = grad_diffuse_shading[i_elem];
            float l_albedo = albedo[i_elem];
            s_diffuse_grad[loc_offset] = l_grad_shading * l_albedo + l_grad_diffuse_shading;
            if (enable_specular){
                float l_specular_albedo = specular_albedo[i_elem];
                s_specular_grad[loc_offset] = l_grad_shading * l_specular_albedo;
            }
        }
    }
    __syncthreads();
    ////////////// acc grad of each texel
    int N_texel = H * W;
    int n_loop = div_round_up(N_texel, 256);
    for(int i_loop = 0; i_loop < n_loop; i_loop ++){
        int i_texel = i_loop * 256 + threadIdx.x;
        if(i_texel < N_texel){ // check valid texel
            int ix = i_texel % W;
            int iy = i_texel / W;
#ifdef LEFT_TOP_AS_ORIGIN
            iy = H - 1 - iy;
#endif
            float3 l_dir = getLightDir(iy, ix, W, H);
            float delta_omega = getDeltaOmega(iy, W, H);
            float3 rot_l_dir;
            rot_l_dir.x = l_mat[0] * l_dir.x + l_mat[1] * l_dir.y + l_mat[2] * l_dir.z;
            rot_l_dir.y = l_mat[3] * l_dir.x + l_mat[4] * l_dir.y + l_mat[5] * l_dir.z;
            rot_l_dir.z = l_mat[6] * l_dir.x + l_mat[7] * l_dir.y + l_mat[8] * l_dir.z;
            // clear grad
            #pragma unroll
            for(int i = 0; i < C; i++){
                l_grad_envmap[i] = 0;
            }
            for(int i = 0; i < 256; i++){
                if( blockIdx.x * int64_t(256) + i < P){ // check valid fragment
                    // compute visibility
                    int i_slot = i_texel / 32;
                    int v0 = s_v[i * 3 + 0]; 
                    int v1 = s_v[i * 3 + 1];
                    int v2 = s_v[i * 3 + 2];
                    float c0 = s_c[i * 3 + 0];
                    float c1 = s_c[i * 3 + 1];
                    float c2 = s_c[i * 3 + 2];
                    uint32_t p0 = visibility[i_slot * N_vertex + v0];
                    uint32_t p1 = visibility[i_slot * N_vertex + v1];
                    uint32_t p2 = visibility[i_slot * N_vertex + v2];
                    uint32_t bit_disp = (0x01 << (i_texel % 32));
                    float vis_0 = (p0 & bit_disp) ? 1.f : 0.f;
                    float vis_1 = (p1 & bit_disp) ? 1.f : 0.f;
                    float vis_2 = (p2 & bit_disp) ? 1.f : 0.f;
                    float vis_ = vis_0 * c0 + vis_1 * c1 + vis_2 * c2;
                    //
                    l_normal.x = s_normal[i * 3 + 0];
                    l_normal.y = s_normal[i * 3 + 1];
                    l_normal.z = s_normal[i * 3 + 2];
                    { // diffuse
                        float cos_shade = dot(rot_l_dir, l_normal);
                        if(cos_shade > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float diffuse_grad = s_diffuse_grad[i * C + i_channel];
                                //l_grad_envmap[i_channel] += diffuse_grad * (cos_shade / M_PIf) * delta_omega;
                                l_grad_envmap[i_channel] += diffuse_grad * (cos_shade / M_PIf) * delta_omega * vis_;
                            }
                        }
                    }
                    if (enable_specular){
                        float3 l_viewdir;
                        l_viewdir.x = s_viewdir[i * 3 + 0];
                        l_viewdir.y = s_viewdir[i * 3 + 1];
                        l_viewdir.z = s_viewdir[i * 3 + 2];
                        float l_roughness = s_roughness[i];
                        float alpha0 = l_roughness * l_roughness;
                        alpha0 = alpha0 * alpha0;
                        float alpha1 = l_roughness + 1.f;
                        alpha1 = (alpha1 * alpha1) / 8.f;
                        float3 h = add(rot_l_dir, l_viewdir);
                        h = normalize(h);
                        float cosh = dot(rot_l_dir, h);
                        float F = FresnelApproximate(F0, cosh);
                        float G = G1_robust(rot_l_dir, l_viewdir, l_normal, alpha1);
                        float D = NDF1(l_normal, h, alpha0);
                        float tmp = (F * G * D) / 4.f ;
                        if(tmp > 0){
                            #pragma unroll
                            for(int i_channel = 0; i_channel < C; i_channel++){
                                float specular_grad = s_specular_grad[i * C + i_channel];
                                //l_grad_envmap[i_channel] += specular_grad * tmp * delta_omega;
                                l_grad_envmap[i_channel] += specular_grad * tmp * delta_omega * vis_;
                            }
                        }
                    }
                }                 
            }
            // global write
            #pragma unroll
            for(int i_channel = 0; i_channel < C; i_channel++){
                atomicAdd(grad_envmap + (i_channel * N_texel + i_texel), l_grad_envmap[i_channel]);
            }
        }
    }
}

/////////////////////////////

torch::Tensor bruteforce_diffuse_shader_forward(
    const torch::Tensor & normal, // Px3
    const torch::Tensor & albedo, // PxC
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat // 3x3
){
    CHECK_CUDA(normal); CHECK_CONTIGUOUS(normal); CHECK_IS_FLOATING(normal);
    CHECK_CUDA(albedo); CHECK_CONTIGUOUS(albedo); CHECK_IS_FLOATING(albedo);
    CHECK_CUDA(envmap); CHECK_CONTIGUOUS(envmap); CHECK_IS_FLOATING(envmap);
    CHECK_CUDA(light2obj_rotmat); CHECK_CONTIGUOUS(light2obj_rotmat); CHECK_IS_FLOATING(light2obj_rotmat);

    // TODO: shape check

    int64_t P = normal.size(0);
    int64_t C = albedo.size(1);
    int H = (int)envmap.size(1);
    int W = (int)envmap.size(2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(normal));
    auto device = normal.device();

    at::TensorOptions opt(normal.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor shading = torch::empty({P, C}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);

    if(BLOCKS){
        if(C == 3){
            BruteForceDiffuseShaderKernel_Forward<3><<<BLOCKS,THREADS>>>(
                P,H,W,
                normal.data_ptr<float>(),
                albedo.data_ptr<float>(),
                envmap.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                shading.data_ptr<float>()
            );
        }
        else{
            throw std::runtime_error("Channel number not supported!");
        }
    }
    return shading;
}

std::vector<torch::Tensor> bruteforce_diffuse_shader_backward(
    const torch::Tensor & grad_shading, // Px3
    const torch::Tensor & normal, // Px3
    const torch::Tensor & albedo, // Px3
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat // 3x3
){
    CHECK_CUDA(grad_shading); CHECK_CONTIGUOUS(grad_shading); CHECK_IS_FLOATING(grad_shading);
    CHECK_CUDA(normal); CHECK_CONTIGUOUS(normal); CHECK_IS_FLOATING(normal);
    CHECK_CUDA(albedo); CHECK_CONTIGUOUS(albedo); CHECK_IS_FLOATING(albedo);
    CHECK_CUDA(envmap); CHECK_CONTIGUOUS(envmap); CHECK_IS_FLOATING(envmap);
    CHECK_CUDA(light2obj_rotmat); CHECK_CONTIGUOUS(light2obj_rotmat); CHECK_IS_FLOATING(light2obj_rotmat);

    // TODO: shape check

    int64_t P = normal.size(0);
    int64_t C = albedo.size(1);
    int H = (int)envmap.size(1);
    int W = (int)envmap.size(2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(normal));
    auto device = normal.device();

    at::TensorOptions opt(normal.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor grad_albedo = torch::empty({P, C}, opt);
    torch::Tensor grad_normal = torch::empty({P, 3}, opt);
    torch::Tensor grad_envmap = torch::zeros({C, H, W}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);

    if(BLOCKS){
        if(C == 3){
            BruteForceDiffuseShaderKernel_Backward<3><<<BLOCKS, THREADS>>>(
                P,H,W,
                grad_shading.data_ptr<float>(),
                normal.data_ptr<float>(),
                albedo.data_ptr<float>(),
                envmap.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                grad_normal.data_ptr<float>(), // [out]
                grad_albedo.data_ptr<float>() // [out]
            );
            BruteForceDiffuseShaderKernel_BackwardEnvmap<3><<<BLOCKS, THREADS>>>(
                P,H,W,
                grad_shading.data_ptr<float>(),
                normal.data_ptr<float>(),
                albedo.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                grad_envmap.data_ptr<float>() // [out]
            );
        }
        else{
            throw std::runtime_error("Channel number not supported!");
        }
    }

    return {grad_normal, grad_albedo, grad_envmap };
}

//---------------------------------------------------------

torch::Tensor bruteforce_specular_shader_forward(
    const torch::Tensor & normal, // Px3
    const torch::Tensor & view_dir, // Px3
    const torch::Tensor & albedo, // Px3
    const torch::Tensor & specular_albedo, // Px3
    const torch::Tensor & roughness, // Px1
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat, // 3x3
    bool enable_diffuse, bool enable_specular
){
    CHECK_CUDA(normal); CHECK_CONTIGUOUS(normal); CHECK_IS_FLOATING(normal);
    CHECK_CUDA(view_dir); CHECK_CONTIGUOUS(view_dir); CHECK_IS_FLOATING(view_dir);
    CHECK_CUDA(albedo); CHECK_CONTIGUOUS(albedo); CHECK_IS_FLOATING(albedo);
    CHECK_CUDA(specular_albedo); CHECK_CONTIGUOUS(specular_albedo); CHECK_IS_FLOATING(specular_albedo);
    CHECK_CUDA(roughness); CHECK_CONTIGUOUS(roughness); CHECK_IS_FLOATING(roughness);
    CHECK_CUDA(envmap); CHECK_CONTIGUOUS(envmap); CHECK_IS_FLOATING(envmap);
    CHECK_CUDA(light2obj_rotmat); CHECK_CONTIGUOUS(light2obj_rotmat); CHECK_IS_FLOATING(light2obj_rotmat);

    // TODO: shape check

    int64_t P = normal.size(0);
    int64_t C = albedo.size(1);
    int H = (int)envmap.size(1);
    int W = (int)envmap.size(2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(normal));
    auto device = normal.device();

    at::TensorOptions opt(normal.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor shading = torch::empty({P, C}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);

    if (BLOCKS){
        if(C == 3){
            BruteForceSpecularShaderKernel_Forward<3><<<BLOCKS, THREADS>>>(
                P, H, W,
                normal.data_ptr<float>(),
                view_dir.data_ptr<float>(),
                albedo.data_ptr<float>(),
                specular_albedo.data_ptr<float>(),
                roughness.data_ptr<float>(),
                envmap.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                shading.data_ptr<float>(),
                enable_diffuse, enable_specular
            );
        }
        else{
            throw std::runtime_error("Channel number not supported!");
        }
    }
    return shading;
}

std::vector<torch::Tensor> bruteforce_specular_shader_backward(
    const torch::Tensor & grad_shading, // Px3
    const torch::Tensor & normal, // Px3
    const torch::Tensor & view_dir, // Px3
    const torch::Tensor & albedo, // PxC
    const torch::Tensor & specular_albedo, // PxC
    const torch::Tensor & roughness, // Px1
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat, // 3x3
    bool enable_diffuse, bool enable_specular
){
    CHECK_CUDA(grad_shading); CHECK_CONTIGUOUS(grad_shading); CHECK_IS_FLOATING(grad_shading);
    CHECK_CUDA(normal); CHECK_CONTIGUOUS(normal); CHECK_IS_FLOATING(normal);
    CHECK_CUDA(view_dir); CHECK_CONTIGUOUS(view_dir); CHECK_IS_FLOATING(view_dir);
    CHECK_CUDA(albedo); CHECK_CONTIGUOUS(albedo); CHECK_IS_FLOATING(albedo);
    CHECK_CUDA(specular_albedo); CHECK_CONTIGUOUS(specular_albedo); CHECK_IS_FLOATING(specular_albedo);
    CHECK_CUDA(roughness); CHECK_CONTIGUOUS(roughness); CHECK_IS_FLOATING(roughness);
    CHECK_CUDA(envmap); CHECK_CONTIGUOUS(envmap); CHECK_IS_FLOATING(envmap);
    CHECK_CUDA(light2obj_rotmat); CHECK_CONTIGUOUS(light2obj_rotmat); CHECK_IS_FLOATING(light2obj_rotmat);

    // TODO: shape check

    int64_t P = normal.size(0);
    int64_t C = albedo.size(1);
    int H = (int)envmap.size(1);
    int W = (int)envmap.size(2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(normal));
    auto device = normal.device();

    at::TensorOptions opt(normal.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness;
    torch::Tensor grad_normal = torch::empty({P, 3}, opt);
    torch::Tensor grad_envmap = torch::zeros({C, H, W}, opt);
    // diffuse relative
    if (enable_diffuse){
        grad_albedo = torch::empty({P, C}, opt);
    }
    else{
        grad_albedo = torch::zeros({P, C}, opt);
    }
    // specular relative
    if (enable_specular){
        grad_view_dir = torch::empty({P, 3}, opt);
        grad_roughness = torch::empty({P, 1}, opt);
        grad_specular_albedo = torch::empty({P, C}, opt);
    }
    else{
        grad_view_dir = torch::zeros({P, 3}, opt);
        grad_roughness = torch::zeros({P, 1}, opt);
        grad_specular_albedo = torch::zeros({P, C}, opt);
    }

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);

    if (BLOCKS){
        if (C == 3){
            BruteForceSpecularShaderKernel_Backward<3><<<BLOCKS,THREADS>>>(
                P,H,W,
                grad_shading.data_ptr<float>(),
                normal.data_ptr<float>(),
                view_dir.data_ptr<float>(),
                albedo.data_ptr<float>(),
                specular_albedo.data_ptr<float>(),
                roughness.data_ptr<float>(),
                envmap.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                // output
                grad_normal.data_ptr<float>(),
                grad_view_dir.data_ptr<float>(),
                grad_albedo.data_ptr<float>(),
                grad_specular_albedo.data_ptr<float>(),
                grad_roughness.data_ptr<float>(),
                enable_diffuse, enable_specular
            );
            BruteForceSpecularShaderKernel_BackwardEnvmap<3><<<BLOCKS, THREADS>>>(
                P,H,W,
                grad_shading.data_ptr<float>(),
                normal.data_ptr<float>(),
                view_dir.data_ptr<float>(),
                albedo.data_ptr<float>(),
                specular_albedo.data_ptr<float>(),
                roughness.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                grad_envmap.data_ptr<float>(),
                enable_diffuse, enable_specular
            );
        }
        else{
            throw std::runtime_error("Channel number not supported!");
        }
    }

    return {grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap};
}

//---------------------------------------------------------

torch::Tensor bruteforce_specular_shader_clamp_forward(
    const torch::Tensor & normal, // Px3
    const torch::Tensor & view_dir, // Px3
    const torch::Tensor & albedo, // Px3
    const torch::Tensor & specular_albedo, // Px3
    const torch::Tensor & roughness, // Px1
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat, // 3x3
    bool enable_diffuse, bool enable_specular
){
    CHECK_CUDA(normal); CHECK_CONTIGUOUS(normal); CHECK_IS_FLOATING(normal);
    CHECK_CUDA(view_dir); CHECK_CONTIGUOUS(view_dir); CHECK_IS_FLOATING(view_dir);
    CHECK_CUDA(albedo); CHECK_CONTIGUOUS(albedo); CHECK_IS_FLOATING(albedo);
    CHECK_CUDA(specular_albedo); CHECK_CONTIGUOUS(specular_albedo); CHECK_IS_FLOATING(specular_albedo);
    CHECK_CUDA(roughness); CHECK_CONTIGUOUS(roughness); CHECK_IS_FLOATING(roughness);
    CHECK_CUDA(envmap); CHECK_CONTIGUOUS(envmap); CHECK_IS_FLOATING(envmap);
    CHECK_CUDA(light2obj_rotmat); CHECK_CONTIGUOUS(light2obj_rotmat); CHECK_IS_FLOATING(light2obj_rotmat);

    // TODO: shape check

    int64_t P = normal.size(0);
    int64_t C = albedo.size(1);
    int H = (int)envmap.size(1);
    int W = (int)envmap.size(2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(normal));
    auto device = normal.device();

    at::TensorOptions opt(normal.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor shading = torch::empty({P, C}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);

    if (BLOCKS){
        if(C == 3){
            BruteForceSpecularShaderClampKernel_Forward<3><<<BLOCKS, THREADS>>>(
                P, H, W,
                normal.data_ptr<float>(),
                view_dir.data_ptr<float>(),
                albedo.data_ptr<float>(),
                specular_albedo.data_ptr<float>(),
                roughness.data_ptr<float>(),
                envmap.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                shading.data_ptr<float>(),
                enable_diffuse, enable_specular
            );
        }
        else{
            throw std::runtime_error("Channel number not supported!");
        }
    }
    return shading;
}

std::vector<torch::Tensor> bruteforce_specular_shader_clamp_backward(
    const torch::Tensor & grad_shading, // Px3
    const torch::Tensor & normal, // Px3
    const torch::Tensor & view_dir, // Px3
    const torch::Tensor & albedo, // PxC
    const torch::Tensor & specular_albedo, // PxC
    const torch::Tensor & roughness, // Px1
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat, // 3x3
    bool enable_diffuse, bool enable_specular
){
    CHECK_CUDA(grad_shading); CHECK_CONTIGUOUS(grad_shading); CHECK_IS_FLOATING(grad_shading);
    CHECK_CUDA(normal); CHECK_CONTIGUOUS(normal); CHECK_IS_FLOATING(normal);
    CHECK_CUDA(view_dir); CHECK_CONTIGUOUS(view_dir); CHECK_IS_FLOATING(view_dir);
    CHECK_CUDA(albedo); CHECK_CONTIGUOUS(albedo); CHECK_IS_FLOATING(albedo);
    CHECK_CUDA(specular_albedo); CHECK_CONTIGUOUS(specular_albedo); CHECK_IS_FLOATING(specular_albedo);
    CHECK_CUDA(roughness); CHECK_CONTIGUOUS(roughness); CHECK_IS_FLOATING(roughness);
    CHECK_CUDA(envmap); CHECK_CONTIGUOUS(envmap); CHECK_IS_FLOATING(envmap);
    CHECK_CUDA(light2obj_rotmat); CHECK_CONTIGUOUS(light2obj_rotmat); CHECK_IS_FLOATING(light2obj_rotmat);

    // TODO: shape check

    int64_t P = normal.size(0);
    int64_t C = albedo.size(1);
    int H = (int)envmap.size(1);
    int W = (int)envmap.size(2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(normal));
    auto device = normal.device();

    at::TensorOptions opt(normal.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness;
    torch::Tensor grad_normal = torch::empty({P, 3}, opt);
    torch::Tensor grad_envmap = torch::zeros({C, H, W}, opt);
    // diffuse relative
    if (enable_diffuse){
        grad_albedo = torch::empty({P, C}, opt);
    }
    else{
        grad_albedo = torch::zeros({P, C}, opt);
    }
    // specular relative
    if (enable_specular){
        grad_view_dir = torch::empty({P, 3}, opt);
        grad_roughness = torch::empty({P, 1}, opt);
        grad_specular_albedo = torch::empty({P, C}, opt);
    }
    else{
        grad_view_dir = torch::zeros({P, 3}, opt);
        grad_roughness = torch::zeros({P, 1}, opt);
        grad_specular_albedo = torch::zeros({P, C}, opt);
    }

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);

    if (BLOCKS){
        if (C == 3){
            BruteForceSpecularShaderClampKernel_Backward<3><<<BLOCKS,THREADS>>>(
                P,H,W,
                grad_shading.data_ptr<float>(),
                normal.data_ptr<float>(),
                view_dir.data_ptr<float>(),
                albedo.data_ptr<float>(),
                specular_albedo.data_ptr<float>(),
                roughness.data_ptr<float>(),
                envmap.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                // output
                grad_normal.data_ptr<float>(),
                grad_view_dir.data_ptr<float>(),
                grad_albedo.data_ptr<float>(),
                grad_specular_albedo.data_ptr<float>(),
                grad_roughness.data_ptr<float>(),
                enable_diffuse, enable_specular
            );
            BruteForceSpecularShaderClampKernel_BackwardEnvmap<3><<<BLOCKS, THREADS>>>(
                P,H,W,
                grad_shading.data_ptr<float>(),
                normal.data_ptr<float>(),
                view_dir.data_ptr<float>(),
                albedo.data_ptr<float>(),
                specular_albedo.data_ptr<float>(),
                roughness.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                grad_envmap.data_ptr<float>(),
                enable_diffuse, enable_specular
            );
        }
        else{
            throw std::runtime_error("Channel number not supported!");
        }
    }

    return {grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap};
}

std::vector<torch::Tensor> bruteforce_specular_shader_forward2(
    const torch::Tensor & normal, // Px3
    const torch::Tensor & view_dir, // Px3
    const torch::Tensor & albedo, // Px3
    const torch::Tensor & specular_albedo, // Px3
    const torch::Tensor & roughness, // Px1
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat, // 3x3
    bool enable_specular
){
    CHECK_CUDA(normal); CHECK_CONTIGUOUS(normal); CHECK_IS_FLOATING(normal);
    CHECK_CUDA(view_dir); CHECK_CONTIGUOUS(view_dir); CHECK_IS_FLOATING(view_dir);
    CHECK_CUDA(albedo); CHECK_CONTIGUOUS(albedo); CHECK_IS_FLOATING(albedo);
    CHECK_CUDA(specular_albedo); CHECK_CONTIGUOUS(specular_albedo); CHECK_IS_FLOATING(specular_albedo);
    CHECK_CUDA(roughness); CHECK_CONTIGUOUS(roughness); CHECK_IS_FLOATING(roughness);
    CHECK_CUDA(envmap); CHECK_CONTIGUOUS(envmap); CHECK_IS_FLOATING(envmap);
    CHECK_CUDA(light2obj_rotmat); CHECK_CONTIGUOUS(light2obj_rotmat); CHECK_IS_FLOATING(light2obj_rotmat);

    // TODO: shape check

    int64_t P = normal.size(0);
    int64_t C = albedo.size(1);
    int H = (int)envmap.size(1);
    int W = (int)envmap.size(2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(normal));
    auto device = normal.device();

    at::TensorOptions opt(normal.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor shading = torch::empty({P,C}, opt);
    torch::Tensor diffuse_shading = torch::empty({P,C}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);

    if (BLOCKS){
        if(C == 3){
            BruteForceSpecularShaderKernel_Forward2<3><<<BLOCKS, THREADS>>>(
                P, H, W,
                normal.data_ptr<float>(),
                view_dir.data_ptr<float>(),
                albedo.data_ptr<float>(),
                specular_albedo.data_ptr<float>(),
                roughness.data_ptr<float>(),
                envmap.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                shading.data_ptr<float>(),
                diffuse_shading.data_ptr<float>(),
                enable_specular
            );
        }
        else{
            throw std::runtime_error("Channel number not supported!");
        }
    }
    return {shading, diffuse_shading};
}

std::vector<torch::Tensor> bruteforce_specular_shader_backward2(
    const torch::Tensor & grad_shading, // Px3
    const torch::Tensor & grad_diffuse_shading, // Px3
    const torch::Tensor & normal, // Px3
    const torch::Tensor & view_dir, // Px3
    const torch::Tensor & albedo, // PxC
    const torch::Tensor & specular_albedo, // PxC
    const torch::Tensor & roughness, // Px1
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat, // 3x3
    bool enable_specular
){
    CHECK_CUDA(grad_shading); CHECK_CONTIGUOUS(grad_shading); CHECK_IS_FLOATING(grad_shading);
    CHECK_CUDA(grad_diffuse_shading); CHECK_CONTIGUOUS(grad_diffuse_shading); CHECK_IS_FLOATING(grad_diffuse_shading);
    CHECK_CUDA(normal); CHECK_CONTIGUOUS(normal); CHECK_IS_FLOATING(normal);
    CHECK_CUDA(view_dir); CHECK_CONTIGUOUS(view_dir); CHECK_IS_FLOATING(view_dir);
    CHECK_CUDA(albedo); CHECK_CONTIGUOUS(albedo); CHECK_IS_FLOATING(albedo);
    CHECK_CUDA(specular_albedo); CHECK_CONTIGUOUS(specular_albedo); CHECK_IS_FLOATING(specular_albedo);
    CHECK_CUDA(roughness); CHECK_CONTIGUOUS(roughness); CHECK_IS_FLOATING(roughness);
    CHECK_CUDA(envmap); CHECK_CONTIGUOUS(envmap); CHECK_IS_FLOATING(envmap);
    CHECK_CUDA(light2obj_rotmat); CHECK_CONTIGUOUS(light2obj_rotmat); CHECK_IS_FLOATING(light2obj_rotmat);

    // TODO: shape check

    int64_t P = normal.size(0);
    int64_t C = albedo.size(1);
    int H = (int)envmap.size(1);
    int W = (int)envmap.size(2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(normal));
    auto device = normal.device();

    at::TensorOptions opt(normal.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor grad_view_dir, grad_specular_albedo, grad_roughness;
    torch::Tensor grad_normal = torch::empty({P, 3}, opt);
    torch::Tensor grad_envmap = torch::zeros({C, H, W}, opt);
    torch::Tensor grad_albedo = torch::empty({P, C}, opt);
    if(enable_specular){
        grad_view_dir = torch::empty({P, 3}, opt);
        grad_roughness = torch::empty({P, 1}, opt);
        grad_specular_albedo = torch::empty({P, C}, opt);
    }
    else{
        grad_view_dir = torch::zeros({P, 3}, opt);
        grad_roughness = torch::zeros({P, 1}, opt);
        grad_specular_albedo = torch::zeros({P, C}, opt);
    }

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);

    if (BLOCKS){
        if (C == 3){
            BruteForceSpecularShaderKernel_Backward2<3><<<BLOCKS, THREADS>>>(
                P,H,W,
                grad_shading.data_ptr<float>(),
                grad_diffuse_shading.data_ptr<float>(),
                normal.data_ptr<float>(),
                view_dir.data_ptr<float>(),
                albedo.data_ptr<float>(),
                specular_albedo.data_ptr<float>(),
                roughness.data_ptr<float>(),
                envmap.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                // output
                grad_normal.data_ptr<float>(),
                grad_view_dir.data_ptr<float>(),
                grad_albedo.data_ptr<float>(),
                grad_specular_albedo.data_ptr<float>(),
                grad_roughness.data_ptr<float>(),
                enable_specular
            );
            BruteForceSpecularShaderKernel_BackwardEnvmap2<3><<<BLOCKS,THREADS>>>(
                P,H,W,
                grad_shading.data_ptr<float>(),
                grad_diffuse_shading.data_ptr<float>(),
                normal.data_ptr<float>(),
                view_dir.data_ptr<float>(),
                albedo.data_ptr<float>(),
                specular_albedo.data_ptr<float>(),
                roughness.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                grad_envmap.data_ptr<float>(),
                enable_specular
            );
        }
        else{
            throw std::runtime_error("Channel number not supported!");
        }
    }
    return {grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap};
}

std::vector<torch::Tensor> bruteforce_specularvisibility_shader_forward2(
    const torch::Tensor & normal, // Px3
    const torch::Tensor & view_dir, // Px3
    const torch::Tensor & albedo, // Px3
    const torch::Tensor & specular_albedo, // Px3
    const torch::Tensor & roughness, // Px1
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat, // 3x3
    //
    const torch::Tensor & face_vertex_list, // 3xN_face
    const torch::Tensor & nearest_triangle_id, // P
    const torch::Tensor & barycentric_coord, // 3xP
    const torch::Tensor & visibility, // N_slotxN_vertex
    bool enable_specular
){
    CHECK_CUDA(normal); CHECK_CONTIGUOUS(normal); CHECK_IS_FLOATING(normal);
    CHECK_CUDA(view_dir); CHECK_CONTIGUOUS(view_dir); CHECK_IS_FLOATING(view_dir);
    CHECK_CUDA(albedo); CHECK_CONTIGUOUS(albedo); CHECK_IS_FLOATING(albedo);
    CHECK_CUDA(specular_albedo); CHECK_CONTIGUOUS(specular_albedo); CHECK_IS_FLOATING(specular_albedo);
    CHECK_CUDA(roughness); CHECK_CONTIGUOUS(roughness); CHECK_IS_FLOATING(roughness);
    CHECK_CUDA(envmap); CHECK_CONTIGUOUS(envmap); CHECK_IS_FLOATING(envmap);
    CHECK_CUDA(light2obj_rotmat); CHECK_CONTIGUOUS(light2obj_rotmat); CHECK_IS_FLOATING(light2obj_rotmat);

    CHECK_CUDA(face_vertex_list); CHECK_CONTIGUOUS(face_vertex_list); CHECK_IS_INT(face_vertex_list);
    CHECK_CUDA(nearest_triangle_id); CHECK_CONTIGUOUS(nearest_triangle_id); CHECK_IS_LONG(nearest_triangle_id);
    CHECK_CUDA(barycentric_coord); CHECK_CONTIGUOUS(barycentric_coord); CHECK_IS_FLOATING(barycentric_coord);
    CHECK_CUDA(visibility); CHECK_CONTIGUOUS(visibility); CHECK_IS_INT(visibility);

    // TODO: shape check

    int64_t P = normal.size(0);
    int64_t C = albedo.size(1);
    int H = (int)envmap.size(1);
    int W = (int)envmap.size(2);
    int N_face = (int)face_vertex_list.size(1);
    int N_vertex = (int)visibility.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(normal));
    auto device = normal.device();

    at::TensorOptions opt(normal.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor shading = torch::empty({P,C}, opt);
    torch::Tensor diffuse_shading = torch::empty({P,C}, opt);

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);

    if (BLOCKS){
        if(C == 3){
            BruteForceSpecularVisibilityShaderKernel_Forward2<3><<<BLOCKS, THREADS>>>(
                P, H, W, N_face, N_vertex,
                normal.data_ptr<float>(),
                view_dir.data_ptr<float>(),
                albedo.data_ptr<float>(),
                specular_albedo.data_ptr<float>(),
                roughness.data_ptr<float>(),
                envmap.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                //
                face_vertex_list.data_ptr<int>(),
                nearest_triangle_id.data_ptr<int64_t>(),
                barycentric_coord.data_ptr<float>(),
                (uint32_t*)visibility.data_ptr<int>(),
                //
                shading.data_ptr<float>(),
                diffuse_shading.data_ptr<float>(),
                enable_specular
            );
        }
        else{
            throw std::runtime_error("Channel number not supported!");
        }
    }
    return {shading, diffuse_shading};
}

std::vector<torch::Tensor> bruteforce_specularvisibility_shader_backward2(
    const torch::Tensor & grad_shading, // Px3
    const torch::Tensor & grad_diffuse_shading, // Px3
    const torch::Tensor & normal, // Px3
    const torch::Tensor & view_dir, // Px3
    const torch::Tensor & albedo, // PxC
    const torch::Tensor & specular_albedo, // PxC
    const torch::Tensor & roughness, // Px1
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat, // 3x3
    //
    const torch::Tensor & face_vertex_list, // 3xN_face
    const torch::Tensor & nearest_triangle_id, // P
    const torch::Tensor & barycentric_coord, // 3xP
    const torch::Tensor & visibility, // N_slotxN_vertex
    bool enable_specular
){
    CHECK_CUDA(grad_shading); CHECK_CONTIGUOUS(grad_shading); CHECK_IS_FLOATING(grad_shading);
    CHECK_CUDA(grad_diffuse_shading); CHECK_CONTIGUOUS(grad_diffuse_shading); CHECK_IS_FLOATING(grad_diffuse_shading);
    CHECK_CUDA(normal); CHECK_CONTIGUOUS(normal); CHECK_IS_FLOATING(normal);
    CHECK_CUDA(view_dir); CHECK_CONTIGUOUS(view_dir); CHECK_IS_FLOATING(view_dir);
    CHECK_CUDA(albedo); CHECK_CONTIGUOUS(albedo); CHECK_IS_FLOATING(albedo);
    CHECK_CUDA(specular_albedo); CHECK_CONTIGUOUS(specular_albedo); CHECK_IS_FLOATING(specular_albedo);
    CHECK_CUDA(roughness); CHECK_CONTIGUOUS(roughness); CHECK_IS_FLOATING(roughness);
    CHECK_CUDA(envmap); CHECK_CONTIGUOUS(envmap); CHECK_IS_FLOATING(envmap);
    CHECK_CUDA(light2obj_rotmat); CHECK_CONTIGUOUS(light2obj_rotmat); CHECK_IS_FLOATING(light2obj_rotmat);

    CHECK_CUDA(face_vertex_list); CHECK_CONTIGUOUS(face_vertex_list); CHECK_IS_INT(face_vertex_list);
    CHECK_CUDA(nearest_triangle_id); CHECK_CONTIGUOUS(nearest_triangle_id); CHECK_IS_LONG(nearest_triangle_id);
    CHECK_CUDA(barycentric_coord); CHECK_CONTIGUOUS(barycentric_coord); CHECK_IS_FLOATING(barycentric_coord);
    CHECK_CUDA(visibility); CHECK_CONTIGUOUS(visibility); CHECK_IS_INT(visibility);

    // TODO: shape check

    int64_t P = normal.size(0);
    int64_t C = albedo.size(1);
    int H = (int)envmap.size(1);
    int W = (int)envmap.size(2);
    int N_face = (int)face_vertex_list.size(1);
    int N_vertex = (int)visibility.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(normal));
    auto device = normal.device();

    at::TensorOptions opt(normal.dtype()); opt = opt.device(device); opt = opt.requires_grad(false);

    torch::Tensor grad_view_dir, grad_specular_albedo, grad_roughness;
    torch::Tensor grad_normal = torch::empty({P, 3}, opt);
    torch::Tensor grad_envmap = torch::zeros({C, H, W}, opt);
    torch::Tensor grad_albedo = torch::empty({P, C}, opt);
    if(enable_specular){
        grad_view_dir = torch::empty({P, 3}, opt);
        grad_roughness = torch::empty({P, 1}, opt);
        grad_specular_albedo = torch::empty({P, C}, opt);
    }
    else{
        grad_view_dir = torch::zeros({P, 3}, opt);
        grad_roughness = torch::zeros({P, 1}, opt);
        grad_specular_albedo = torch::zeros({P, C}, opt);
    }

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS = (uint32_t)div_round_up(P, (int64_t)256);

    if (BLOCKS){
        if (C == 3){
            BruteForceSpecularVisibilityShaderKernel_Backward2<3><<<BLOCKS, THREADS>>>(
                P,H,W,N_face, N_vertex,
                grad_shading.data_ptr<float>(),
                grad_diffuse_shading.data_ptr<float>(),
                normal.data_ptr<float>(),
                view_dir.data_ptr<float>(),
                albedo.data_ptr<float>(),
                specular_albedo.data_ptr<float>(),
                roughness.data_ptr<float>(),
                envmap.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                //
                face_vertex_list.data_ptr<int>(),
                nearest_triangle_id.data_ptr<int64_t>(),
                barycentric_coord.data_ptr<float>(),
                (uint32_t*)visibility.data_ptr<int>(),
                // output
                grad_normal.data_ptr<float>(),
                grad_view_dir.data_ptr<float>(),
                grad_albedo.data_ptr<float>(),
                grad_specular_albedo.data_ptr<float>(),
                grad_roughness.data_ptr<float>(),
                enable_specular
            );
            BruteForceSpecularVisibilityShaderKernel_BackwardEnvmap2<3><<<BLOCKS,THREADS>>>(
                P,H,W,N_face,N_vertex,
                grad_shading.data_ptr<float>(),
                grad_diffuse_shading.data_ptr<float>(),
                normal.data_ptr<float>(),
                view_dir.data_ptr<float>(),
                albedo.data_ptr<float>(),
                specular_albedo.data_ptr<float>(),
                roughness.data_ptr<float>(),
                light2obj_rotmat.data_ptr<float>(),
                //
                face_vertex_list.data_ptr<int>(),
                nearest_triangle_id.data_ptr<int64_t>(),
                barycentric_coord.data_ptr<float>(),
                (uint32_t*)visibility.data_ptr<int>(),
                //
                grad_envmap.data_ptr<float>(),
                enable_specular
            );
        }
        else{
            throw std::runtime_error("Channel number not supported!");
        }
    }
    return {grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap};
}