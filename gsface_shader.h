

torch::Tensor bruteforce_diffuse_shader_forward(
    const torch::Tensor & normal, // Px3
    const torch::Tensor & albedo, // PxC
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat // 3x3
);

std::vector<torch::Tensor> bruteforce_diffuse_shader_backward(
    const torch::Tensor & grad_shading, // Px3
    const torch::Tensor & normal, // Px3
    const torch::Tensor & albedo, // Px3
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat // 3x3
);

torch::Tensor bruteforce_specular_shader_forward(
    const torch::Tensor & normal, // Px3
    const torch::Tensor & view_dir, // Px3
    const torch::Tensor & albedo, // Px3
    const torch::Tensor & specular_albedo, // Px3
    const torch::Tensor & roughness, // Px1
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat, // 3x3
    bool enable_diffuse, bool enable_specular
);

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
);

torch::Tensor bruteforce_specular_shader_clamp_forward(
    const torch::Tensor & normal, // Px3
    const torch::Tensor & view_dir, // Px3
    const torch::Tensor & albedo, // Px3
    const torch::Tensor & specular_albedo, // Px3
    const torch::Tensor & roughness, // Px1
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat, // 3x3
    bool enable_diffuse, bool enable_specular
);

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
);

std::vector<torch::Tensor> bruteforce_specular_shader_forward2(
    const torch::Tensor & normal, // Px3
    const torch::Tensor & view_dir, // Px3
    const torch::Tensor & albedo, // Px3
    const torch::Tensor & specular_albedo, // Px3
    const torch::Tensor & roughness, // Px1
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat, // 3x3
    bool enable_specular
);

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
);

std::vector<torch::Tensor> bruteforce_specularvisibility_shader_forward2(
    const torch::Tensor & normal, // Px3
    const torch::Tensor & view_dir, // Px3
    const torch::Tensor & albedo, // Px3
    const torch::Tensor & specular_albedo, // Px3
    const torch::Tensor & roughness, // Px1
    const torch::Tensor & envmap, // CxHxW
    const torch::Tensor & light2obj_rotmat, // 3x3
    const torch::Tensor & face_vertex_list, // 3xN_face
    const torch::Tensor & nearest_triangle_id, // P
    const torch::Tensor & barycentric_coord, // 3xP
    const torch::Tensor & visibility, // N_slotxN_vertex
    bool enable_specular
);

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
    const torch::Tensor & face_vertex_list, // 3xN_face
    const torch::Tensor & nearest_triangle_id, // P
    const torch::Tensor & barycentric_coord, // 3xP
    const torch::Tensor & visibility, // N_slotxN_vertex
    bool enable_specular
);

////////////////////////////////

torch::Tensor get_uv_of_triangle_forward(
    const torch::Tensor & triangles, // Px3x3
    const torch::Tensor & query_pos // Px3
);

torch::Tensor get_uv_of_triangle_backward(
    const torch::Tensor & triangles, // Px3x3
    const torch::Tensor & query_pos, // Px3
    const torch::Tensor & grad_barycentric_coord // Px3
);

std::vector<torch::Tensor> get_nearest_mesh_points_forward(
    const torch::Tensor & adjacency_head, // 2 x N_vertex
    const torch::Tensor & adjacency_list, // N_tbd
    const torch::Tensor & face_vertex_list, // 3 x N_face
    const torch::Tensor & vertex_pos, // 3 x N_vertex
    const torch::Tensor & query_pos, // 3 x P
    const torch::Tensor & idxs // P x K
);

torch::Tensor get_nearest_mesh_points_backward(
    const torch::Tensor & face_vertex_list, // 3 x N_face
    const torch::Tensor & vertex_pos, // 3 x N_vertex
    const torch::Tensor & query_pos, // 3 x P,
    const torch::Tensor & nearest_triangle_id, // P
    const torch::Tensor & grad_barycentric_coord // 3 x P
);

torch::Tensor generate_camera_ray_forward(
    const torch::Tensor & proj,
    at::optional<at::Tensor> & c2w,
    int H, int W, bool flipY, bool normalize, float z_sign
);

torch::Tensor blur_envmap_forward(
    const torch::Tensor & envmap,
    int tar_H, int tar_W, int N_samples, float alpha
);

torch::Tensor sample_envmap_forward(
    const torch::Tensor & query_point,
    const torch::Tensor & envmap
);

torch::Tensor mt_format_conversion_forward(
    const torch::Tensor & jaw, // N x 6
    const torch::Tensor & eyes // N x 12
);

torch::Tensor extract_bitfield_forward(
    const torch::Tensor & bitfield, // N_slot x N
    int N_bit, int i_start, int i_end
); // return (i_end - i_start) x N_bit