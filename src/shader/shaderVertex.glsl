#version 460 core
// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition;// at model space
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in vec3 vertexNormal;

// Output data ; will be interpolated for each fragment.
layout(location = 0) out vec3 fragColor;  // fragment color
layout(location = 1) out vec3 fragPos;    // fragment position
layout(location = 2) out vec3 fragNormal; // fragment normal
layout(location = 3) out vec4 fragPosLightSpace; //fragment position in light space

// Values that stay constant for the whole mesh.
// uniform mat4 MVP;       // model view projection matrix
uniform mat4 lightSpaceMatrix; // light space matrix
uniform mat4 model; // model matrix
uniform mat4 view; // view matrix
uniform mat4 projection; // projection matrix


void main() {
    // REF: https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/2.lighting/2.2.basic_lighting_specular/2.2.basic_lighting.vs
    // https://learnopengl.com/Lighting/Basic-Lighting
    // https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/5.advanced_lighting/3.1.3.shadow_mapping/3.1.3.shadow_mapping.vs
    fragPos = vec3(model * vec4(vertexPosition, 1.0));
    fragColor = vertexColor;
    fragNormal = mat3(transpose(inverse(model))) * vertexNormal;

    // Output position of the vertex, in clip space : MVP * position
    gl_Position = projection * view * vec4(fragPos, 1.0);

    fragPosLightSpace = lightSpaceMatrix * vec4(fragPos,1.0);
}