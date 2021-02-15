#version 460 core
// Interpolated values from the vertex shaders
in vec3 fragColor;
in vec3 fragPos;// fragment position
in vec3 fragNormal; // fragment normal
in vec4 fragPosLightSpace; //fragment position in light space

// Ouput data
out vec3 color;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;
uniform vec3 viewPos;//view position

uniform sampler2D shadowMap;


struct DirectionLight { 
    vec3 direction; // light direction, pointing towards light
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

//https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping
float ShadowCalculation(vec4 fragPosLightSpace)
{
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // calculate bias (based on depth map resolution and slope)
    vec3 normal = normalize(fragNormal);

    // vec3 lightDir = normalize(lightPos - fragPos);
    vec3 lightDir = light.direction;
    
    float bias = max(0.04 * (1.0 - dot(normal, lightDir)), 0.004);
    // check whether current frag pos is in shadow
    // float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth  ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    
    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if(projCoords.z > 1.0)
        shadow = 0.0;
        
    return shadow;
}


void main() {
    // REF: https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/2.lighting/2.2.basic_lighting_specular/2.2.basic_lighting.fs
    // The color of each vertex will be interpolated
    // to produce the color of each fragment

    vec3 normal;
    if(fragNormal.x==0.0 && fragNormal.y==0.0 && fragNormal.z==0.0){ 
        normal = 0.8*light.direction; //if fragNormal is not set, set it to light direction *0.8
     }else{
        normal = normalize(fragNormal);
     }

    // diffuse
    float diffuse = light.diffuse*max(dot(normal,light.direction),0.0);
    
    // specular blinn-phong
    // ref: https://learnopengl.com/Advanced-Lighting/Advanced-Lighting
    vec3 viewDir = normalize(viewPos - fragPos);

    vec3 halfwayDir = normalize(light.direction + viewDir);  
    float specular = light.specular *pow(max(dot(normal, halfwayDir), 0.0), 16.0);
    // vec3 reflectDir = reflect(-light.direction, normal);
    // float specular = light.specular * pow(max(dot(viewDir, reflectDir), 0.0), 32);

    // float ratio = clamp(diffuse+ambient+specular,0,1);
    // float ratio = diffuse+light.ambient+specular;
    // calculate shadow
    float shadow = ShadowCalculation(fragPosLightSpace);
    float ratio = light.ambient+(1-shadow)*(diffuse+specular);

    // float ratio = 1-shadow;

    //Output color = color specified in the vertex shader,
    //interpolated between all 3 surrounding vertices
    color = light.color*fragColor*ratio;

    // float depth = LinearizeDepth(gl_FragCoord.z) / far; // divide by far for demonstration
    // color = vec3(depth);
}