#include <torch/extension.h>

#include "gsface_shader.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bruteforce_diffuse_shader_forward", &bruteforce_diffuse_shader_forward, "bruteforce diffuse shader forward (CUDA)");
    m.def("bruteforce_diffuse_shader_backward", &bruteforce_diffuse_shader_backward, "bruteforce diffuse shader backward (CUDA)");

    m.def("bruteforce_specular_shader_forward", &bruteforce_specular_shader_forward, "bruteforce specular shader forward (CUDA)");
    m.def("bruteforce_specular_shader_backward", &bruteforce_specular_shader_backward, "bruteforce specular shader backward (CUDA)");
    // backface culling, set invalid shading point -> 0
    m.def("bruteforce_specular_shader_clamp_forward", &bruteforce_specular_shader_clamp_forward, "bruteforce specular shader clamp forward (CUDA)");
    m.def("bruteforce_specular_shader_clamp_backward", &bruteforce_specular_shader_clamp_backward, "bruteforce specular shader clamp backward (CUDA)");
    // + output white diffuse shading
    m.def("bruteforce_specular_shader_forward2", &bruteforce_specular_shader_forward2, "bruteforce specular shader forward2 (CUDA)");
    m.def("bruteforce_specular_shader_backward2", &bruteforce_specular_shader_backward2, "bruteforce specular shader backward2 (CUDA)");
    // + shadow
    m.def("bruteforce_specularvisibility_shader_forward2", &bruteforce_specularvisibility_shader_forward2, "bruteforce specular visibility shader forward2 (CUDA)");
    m.def("bruteforce_specularvisibility_shader_backward2", &bruteforce_specularvisibility_shader_backward2, "bruteforce specular visibility shader backward2 (CUDA)");

    m.def("blur_envmap_forward", &blur_envmap_forward, "blur envmap forward (CUDA)");

    m.def("sample_envmap_forward", &sample_envmap_forward, "sample envmap forward (CUDA)");

    m.def("generate_camera_ray_forward", &generate_camera_ray_forward, "generate camera ray forward (CUDA");

    m.def("get_nearest_mesh_points_forward", &get_nearest_mesh_points_forward, "get nearest mesh points forward (CUDA)");
    m.def("get_nearest_mesh_points_backward", &get_nearest_mesh_points_backward, "get nearest mesh points backward (CUDA)");
    m.def("get_uv_of_triangle_forward", &get_uv_of_triangle_forward, "get uv of triangle forward (CUDA)");
    m.def("get_uv_of_triangle_backward", &get_uv_of_triangle_backward, "get uv of triangle backward (CUDA");

    m.def("mt_format_conversion_forward", &mt_format_conversion_forward, "transfer metrical tracker input joints to dim=15 vector forward (CUDA)");
    m.def("extract_bitfield_forward", &extract_bitfield_forward, "extract bitfield to float forward (CUDA)");
}