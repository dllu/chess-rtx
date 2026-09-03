#version 460
#extension GL_EXT_ray_tracing : require

struct RayPayload {
    vec3 color;
    uint depth;
    uint seed;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main() {
    vec3 direction = normalize(gl_WorldRayDirectionEXT);
    float horizon = smoothstep(-0.2, 0.65, direction.y);
    vec3 lowSky = vec3(0.055, 0.075, 0.105);
    vec3 highSky = vec3(0.26, 0.39, 0.58);
    vec3 sky = mix(lowSky, highSky, horizon);
    vec3 sunDirection = normalize(vec3(-0.38, 0.82, 0.42));
    float sun = pow(max(dot(direction, sunDirection), 0.0), 420.0);
    payload.color = sky + vec3(7.5, 6.2, 4.6) * sun;
}
