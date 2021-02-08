#version 460 core
// Interpolated values from the vertex shaders
in vec3 fragColor;
in vec3 fragPos;// fragment position
in vec3 fragNormal; // fragment normal

// Ouput data
out vec3 color;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;
uniform vec3 viewPos;//view position

struct DirectionLight { 
    vec3 direction; // light direction, pointing from light
    vec3 color;     // light color
    float ambient;  // ambient intensity
    float diffuse;  // diffuse intensity
    float specular; // specular intensity
};
uniform DirectionLight light;

// float near = 0.01; 
// float far  = 100.0;
// float LinearizeDepth(float depth) 
// {
//     float z = depth * 2.0 - 1.0; // back to NDC 
//     return (2.0 * near * far) / (far + near - z * (far - near));	
// }


void main() {
    // REF: https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/2.lighting/2.2.basic_lighting_specular/2.2.basic_lighting.fs
    // The color of each vertex will be interpolated
    // to produce the color of each fragment
    if(fragNormal.x!=0.0 && fragNormal.y!=0.0 && fragNormal.z!=0.0){
        vec3 normal = normalize(fragNormal);
        // diffuse
        float diffuse = light.diffuse*max(dot(normal,-light.direction),0.0);
        
        // specular blinn-phong
        // ref: https://learnopengl.com/Advanced-Lighting/Advanced-Lighting
        vec3 viewDir = normalize(viewPos - fragPos);

        vec3 halfwayDir = normalize(-light.direction + viewDir);  
        float specular = light.specular *pow(max(dot(normal, halfwayDir), 0.0), 16.0);
        // vec3 reflectDir = reflect(light.direction, normal);
        // float specular = light.specular * pow(max(dot(viewDir, reflectDir), 0.0), 32);

        // float ratio = clamp(diffuse+ambient+specular,0,1);
        float ratio = diffuse+light.ambient+specular;
        
        //Output color = color specified in the vertex shader,
        //interpolated between all 3 surrounding vertices
        color = light.color*fragColor*ratio;
    }else{
        color = light.color*fragColor;
    }

    // float depth = LinearizeDepth(gl_FragCoord.z) / far; // divide by far for demonstration
    // color = vec3(depth);
}