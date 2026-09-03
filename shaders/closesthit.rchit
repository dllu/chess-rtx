#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT sceneTlas;

layout(std140, set = 0, binding = 2) uniform FrameUniform {
    vec4 cameraPosition;
    vec4 cameraForward;
    vec4 cameraRight;
    vec4 cameraUp;
    vec4 lightPosition;
    vec4 render;
    vec4 effects;
    vec4 resolution;
} frame;

struct Vertex {
    vec4 position;
    vec4 normal;
};

layout(std430, set = 0, binding = 3) readonly buffer VertexBuffer {
    Vertex vertices[];
};
layout(std430, set = 0, binding = 4) readonly buffer IndexBuffer {
    uint indices[];
};
layout(std430, set = 0, binding = 5) readonly buffer TriangleMaterialBuffer {
    uint triangleMaterials[];
};

struct Material {
    vec4 baseColor;
    vec4 surface;
    vec4 optics;
};
layout(std430, set = 0, binding = 6) readonly buffer MaterialBuffer {
    Material materials[];
};

struct RayPayload {
    vec3 color;
    uint depth;
    uint seed;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT uint visibleToLight;
hitAttributeEXT vec2 hitAttributes;

uint hash(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float random(inout uint state) {
    state = hash(state + 0x9e3779b9u);
    return float(state & 0x00ffffffu) / float(0x01000000u);
}

float valueNoise(vec3 p) {
    vec3 cell = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n = dot(cell, vec3(1.0, 57.0, 113.0));
    float a = fract(sin(n + 0.0) * 43758.5453);
    float b = fract(sin(n + 1.0) * 43758.5453);
    float c = fract(sin(n + 57.0) * 43758.5453);
    float d = fract(sin(n + 58.0) * 43758.5453);
    float e = fract(sin(n + 113.0) * 43758.5453);
    float f1 = fract(sin(n + 114.0) * 43758.5453);
    float g = fract(sin(n + 170.0) * 43758.5453);
    float h = fract(sin(n + 171.0) * 43758.5453);
    return mix(mix(mix(a, b, f.x), mix(c, d, f.x), f.y),
               mix(mix(e, f1, f.x), mix(g, h, f.x), f.y), f.z);
}

float fbm(vec3 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int octave = 0; octave < 4; ++octave) {
        value += valueNoise(p) * amplitude;
        p = p * 2.03 + vec3(1.7, 4.2, 2.3);
        amplitude *= 0.5;
    }
    return value;
}

vec3 marbleColor(vec3 baseColor, vec3 worldPosition) {
    float turbulence = fbm(worldPosition * 1.7);
    float vein = abs(sin(worldPosition.x * 4.5 + worldPosition.z * 2.3 + turbulence * 7.0));
    vein = smoothstep(0.90, 0.995, vein);
    vec3 veinColor = mix(baseColor * 0.38, vec3(0.85), step(0.45, dot(baseColor, vec3(0.333))));
    return mix(baseColor * (0.85 + turbulence * 0.22), veinColor, vein * 0.42);
}

vec3 perturbReflection(vec3 direction, vec3 normal, float roughness, float anisotropy, inout uint seed) {
    vec3 tangent = normalize(abs(normal.y) < 0.95 ? cross(normal, vec3(0.0, 1.0, 0.0))
                                                  : cross(normal, vec3(1.0, 0.0, 0.0)));
    vec3 bitangent = cross(normal, tangent);
    float angle = random(seed) * 6.2831853;
    float radius = sqrt(random(seed)) * roughness;
    float tangentSpread = radius * (1.0 + anisotropy * 0.82);
    float bitangentSpread = radius * (1.0 - anisotropy * 0.82);
    return normalize(direction + tangent * cos(angle) * tangentSpread +
                     bitangent * sin(angle) * bitangentSpread);
}

void main() {
    uint triangle = gl_PrimitiveID;
    uint i0 = indices[triangle * 3u + 0u];
    uint i1 = indices[triangle * 3u + 1u];
    uint i2 = indices[triangle * 3u + 2u];
    vec3 barycentric = vec3(1.0 - hitAttributes.x - hitAttributes.y,
                            hitAttributes.x, hitAttributes.y);
    vec3 objectNormal = normalize(vertices[i0].normal.xyz * barycentric.x +
                                  vertices[i1].normal.xyz * barycentric.y +
                                  vertices[i2].normal.xyz * barycentric.z);
    vec3 normal = normalize(vec3(gl_ObjectToWorldEXT * vec4(objectNormal, 0.0)));
    vec3 rayDirection = normalize(gl_WorldRayDirectionEXT);
    if (dot(normal, rayDirection) > 0.0) {
        normal = -normal;
    }
    vec3 worldPosition = gl_WorldRayOriginEXT + rayDirection * gl_HitTEXT;

    uint materialIndex = triangleMaterials[triangle];
    Material material = materials[materialIndex];
    vec3 albedo = material.baseColor.rgb;
    if ((materialIndex == 2u || materialIndex == 3u) && frame.resolution.w < 0.5) {
        albedo = marbleColor(albedo, worldPosition);
    }

    float roughness = clamp(material.surface.x, 0.015, 1.0);
    float metallic = clamp(material.surface.y, 0.0, 1.0);
    float transmission = clamp(material.surface.z, 0.0, 1.0) * frame.effects.y;
    float ior = max(material.surface.w, 1.001);
    float anisotropy = clamp(material.optics.x, -0.95, 0.95);
    float causticStrength = material.optics.y * frame.effects.z;
    float clearCoat = material.optics.z;

    vec3 toLight = frame.lightPosition.xyz - worldPosition;
    float lightDistance = length(toLight);
    vec3 lightDirection = toLight / lightDistance;
    if (frame.effects.w > 0.5) {
        vec3 lightTangent = normalize(cross(lightDirection, vec3(0.13, 1.0, 0.27)));
        vec3 lightBitangent = cross(lightDirection, lightTangent);
        float angle = random(payload.seed) * 6.2831853;
        float radius = sqrt(random(payload.seed)) * frame.render.y;
        vec3 sampledLight = frame.lightPosition.xyz +
            (lightTangent * cos(angle) + lightBitangent * sin(angle)) * radius;
        toLight = sampledLight - worldPosition;
        lightDistance = length(toLight);
        lightDirection = toLight / lightDistance;
    }

    visibleToLight = 0u;
    traceRayEXT(
        sceneTlas,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT |
            gl_RayFlagsSkipClosestHitShaderEXT,
        0xff,
        0,
        0,
        1,
        worldPosition + normal * 0.004,
        0.002,
        lightDirection,
        max(lightDistance - 0.02, 0.003),
        1);

    float nDotL = max(dot(normal, lightDirection), 0.0);
    float visibility = visibleToLight != 0u ? 1.0 : 0.08;
    float attenuation = frame.lightPosition.w / max(lightDistance * lightDistance, 1.0);
    vec3 viewDirection = -rayDirection;
    vec3 halfVector = normalize(lightDirection + viewDirection);
    float specularPower = mix(12.0, 900.0, 1.0 - roughness);
    float specular = pow(max(dot(normal, halfVector), 0.0), specularPower);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 direct = albedo * nDotL * (1.0 - metallic) * 1.65;
    direct += f0 * specular * mix(1.0, 5.0, metallic);
    direct *= attenuation * visibility;
    vec3 color = albedo * vec3(0.055, 0.065, 0.082) + direct;

    uint savedDepth = payload.depth;
    if (savedDepth < uint(frame.render.w + 0.5)) {
        float cosTheta = max(dot(viewDirection, normal), 0.0);
        float dielectricF0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
        float fresnel = dielectricF0 + (1.0 - dielectricF0) * pow(1.0 - cosTheta, 5.0);
        vec3 reflectedDirection = perturbReflection(
            reflect(rayDirection, normal), normal, roughness, anisotropy, payload.seed);

        if (transmission > 0.001) {
            payload.depth = savedDepth + 1u;
            payload.color = vec3(0.0);
            traceRayEXT(sceneTlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0,
                        worldPosition + normal * 0.004, 0.002,
                        reflectedDirection, 1000.0, 0);
            vec3 reflected = payload.color;

            float eta = 1.0 / ior;
            vec3 refractionNormal = normal;
            if (dot(rayDirection, objectNormal) > 0.0) {
                eta = ior;
                refractionNormal = -normal;
            }
            vec3 refractedDirection = refract(rayDirection, refractionNormal, eta);
            vec3 refracted = reflected;
            if (dot(refractedDirection, refractedDirection) > 0.0001) {
                refractedDirection = perturbReflection(
                    normalize(refractedDirection), -refractionNormal,
                    roughness * 0.72, anisotropy, payload.seed);
                payload.depth = savedDepth + 1u;
                payload.color = vec3(0.0);
                traceRayEXT(sceneTlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0,
                            worldPosition - normal * 0.006, 0.003,
                            refractedDirection, 1000.0, 0);
                refracted = payload.color * mix(vec3(1.0), albedo, 0.32);
            }
            float focus = pow(max(dot(normalize(refractedDirection), -lightDirection), 0.0), 48.0);
            vec3 caustic = albedo * focus * causticStrength * 2.5;
            vec3 glass = mix(refracted, reflected, fresnel) + caustic;
            color = mix(color, glass, transmission);
        } else if (frame.effects.x > 0.5 && (metallic > 0.02 || clearCoat > 0.02)) {
            payload.depth = savedDepth + 1u;
            payload.color = vec3(0.0);
            traceRayEXT(sceneTlas, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0,
                        worldPosition + normal * 0.004, 0.002,
                        reflectedDirection, 1000.0, 0);
            vec3 reflected = payload.color;
            float reflectAmount = clamp(metallic * (1.0 - roughness * 0.45) + clearCoat * fresnel, 0.0, 0.96);
            vec3 tintedReflection = mix(reflected, reflected * albedo, metallic * 0.72);
            color = mix(color, tintedReflection, reflectAmount);
        }
    }

    payload.depth = savedDepth;
    payload.color = max(color, vec3(0.0));
}
